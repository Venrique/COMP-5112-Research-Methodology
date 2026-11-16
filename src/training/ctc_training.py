"""Reproducible baseline training; validation selects checkpoints, never test.

Run ``python -m src.training.ctc_training --help`` from the project root.
Resume restores the last completed epoch, including optimizer and RNG state.
An interrupted partial epoch is repeated, not silently counted as complete.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import random
from time import perf_counter
from importlib.metadata import version

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.evaluation.recognition import corpus_metrics
from src.models.cnn_lstm_ctc import (
    CNNLSTMCTC, CNNLSTMCTCConfig, CNNLSTMCTCRecognizer,
    ctc_batch_loss, greedy_ctc_decode,
)
from src.training.ctc_data import (
    CharacterVocabulary, CTCImagePreparation, CTCLineDataset, collate_ctc_lines,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARATION = ROOT / "results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.001
    patience: int = 3
    seed: int = 5112
    gradient_clip: float = 5.0
    cpu_threads: int = 4
    device: str = "cpu"
    log_every_batches: int = 20

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience", "cpu_threads", "log_every_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def audit_alignment(dataset: CTCLineDataset) -> list[dict]:
    """Check image headers and every label before any optimizer update.

    Decode only image headers where the underlying Hugging Face dataset permits
    it. Source images are never rewritten; invalid examples are never dropped.
    """
    source = dataset.dataset
    if hasattr(source, "cast_column"):
        from datasets import Image as DatasetImage
        source = source.cast_column("image", DatasetImage(decode=False))
    rows = []
    for index in dataset.indices:
        row = source[index]
        image = row["image"]
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]) if image.get("bytes") is not None else image["path"])
            try:
                width, height = image.size
            finally:
                image.close()
        else:
            width, height = image.size
        target = dataset.vocabulary.encode(row["text"])
        prepared_width = max(1, round(width * dataset.preparation.height / height))
        required = len(target) + int((target[1:] == target[:-1]).sum())
        rows.append({"source_index": index, "sample_id": row.get("id", f"sample_{index}"),
                     "original_width": width, "original_height": height,
                     "prepared_width": prepared_width, "ctc_timesteps": prepared_width // 4,
                     "minimum_ctc_timesteps": required,
                     "valid": prepared_width >= 4 and prepared_width // 4 >= required})
    return rows


def plot_history(run_dir: str | Path) -> Path:
    """Plot recorded, completed epochs only; a one-epoch run shows one point."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    root = Path(run_dir)
    with (root / "history.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    epochs = [int(r["epoch"]) for r in rows]
    for key, label in (("train_loss", "Training"), ("validation_loss", "Validation")):
        axes[0].plot(epochs, [float(r[key]) for r in rows], marker="o", label=label)
    axes[0].set(ylabel="Mean target-length-normalized CTC loss", title="Loss on completed epochs")
    for key, label in (("validation_cer", "CER"), ("validation_wer", "WER")):
        axes[1].plot(epochs, [100 * float(r[key]) for r in rows], marker="o", label=label)
    axes[1].set(ylabel="Corpus error rate (%)", title="Validation recognition errors")
    for ax in axes:
        ax.set_xlabel("Completed epoch")
        ax.set_xticks(epochs)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    path = root / "training_curves.png"
    fig.savefig(path, dpi=150)
    return path


def load_recognizer(checkpoint_path: str | Path, device: str = "cpu") -> CNNLSTMCTCRecognizer:
    """Load project checkpoints with torch's restricted weights-only loader."""
    path = Path(checkpoint_path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("optimizer_steps", 0) < 1:
        raise ValueError("Expected a project checkpoint with actual optimizer updates")
    model = CNNLSTMCTC(CNNLSTMCTCConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state"])
    vocabulary = CharacterVocabulary(tuple(saved["vocabulary"]["characters"]))
    preparation = CTCImagePreparation(saved["preparation"]["height"])
    recognizer = CNNLSTMCTCRecognizer(model, vocabulary, preparation, device=device, trained=True)
    recognizer.checkpoint_path = str(path.resolve())
    recognizer.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    recognizer.checkpoint_epoch = saved["completed_epochs"]
    return recognizer


def fit(model: CNNLSTMCTC, train: CTCLineDataset, validation: CTCLineDataset,
        run_dir: str | Path, config: TrainingConfig, provenance: dict | None = None,
        resume: bool = False) -> dict:
    """Train a dataset-specific baseline, selecting by validation CER/WER/loss.

    Only train/validation splits are accepted. Existing runs require explicit
    resume and identical data, model, input preparation and training settings;
    only the maximum epoch limit may change. STOP files pause at batch boundaries.
    """
    if str(getattr(train.dataset, "split", "")) != "train":
        raise ValueError("Training requires the existing train split")
    if str(getattr(validation.dataset, "split", "")) != "validation":
        raise ValueError("Checkpoint selection requires validation, never test")
    if train.vocabulary != validation.vocabulary or train.preparation != validation.preparation:
        raise ValueError("Training and validation must share vocabulary and mandatory transforms")
    if model.config.num_classes != train.vocabulary.num_classes or model.config.input_height != train.preparation.height:
        raise ValueError("Model classes/height must match data preparation")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    training_config = asdict(config)
    training_config.pop("epochs")
    code_files = [Path(__file__), ROOT / "src/models/cnn_lstm_ctc.py", ROOT / "src/training/ctc_data.py"]
    identity = {"training": training_config, "model": model.to_config(),
                "vocabulary": train.vocabulary.to_config(), "input": train.preparation.to_config(),
                "train_fingerprint": getattr(train.dataset, "_fingerprint", None),
                "validation_fingerprint": getattr(validation.dataset, "_fingerprint", None),
                "train_indices": list(train.indices), "validation_indices": list(validation.indices),
                "provenance": provenance or {},
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                "versions": {p: version(p) for p in ("torch", "numpy", "Pillow", "datasets", "jiwer")}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = root / "metadata.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Existing run: resume must be explicit and its identity must match")
    elif any(root.iterdir()):
        raise ValueError("Use an empty output directory for a new training run")
    else:
        _json(manifest, {"phase": "baseline_training", "signature": signature, "identity": identity,
                         "requested_epochs": config.epochs, "python": platform.python_version(),
                         "selection": "lowest validation corpus CER, then WER, then validation loss",
                         "test_loaded": False, "optional_preprocessing": "none"})
        for name, dataset in (("train", train), ("validation", validation)):
            rows = audit_alignment(dataset)
            _csv(root / f"{name}_input_audit.csv", rows)
            invalid = [r for r in rows if not r["valid"]]
            if invalid:
                _json(root / "status.json", {"status": "blocked_invalid_inputs", "split": name,
                                             "invalid_count": len(invalid), "first_examples": invalid[:5]})
                raise ValueError(f"{len(invalid)} impossible CTC inputs in {name}; see audit, no samples dropped")
    # Never resume around a failed audit.
    for name in ("train", "validation"):
        audit_path = root / f"{name}_input_audit.csv"
        if not audit_path.is_file():
            raise ValueError("Input audit is incomplete; use a new run after fixing inputs")
        with audit_path.open(encoding="utf-8", newline="") as stream:
            if any(r["valid"] != "True" for r in csv.DictReader(stream)):
                raise ValueError("Input audit failed; do not resume this run")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history, best, bad_epochs, steps, completed = [], None, 0, 0, 0

    def checkpoint() -> dict:
        return {"format_version": 1, "signature": signature, "model_config": asdict(model.config),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "vocabulary": train.vocabulary.to_config(), "preparation": train.preparation.to_config(),
                "completed_epochs": completed, "optimizer_steps": steps,
                "history": history, "best_score": best, "bad_epochs": bad_epochs,
                "torch_rng": torch.get_rng_state(), "shuffle_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}

    last = root / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history, best = saved["history"], saved["best_score"]
        completed, steps, bad_epochs = saved["completed_epochs"], saved["optimizer_steps"], saved["bad_epochs"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        generator.set_state(saved["shuffle_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        if history:
            _csv(root / "history.csv", history)
            plot_history(root)
    else:
        _save(last, checkpoint())
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              generator=generator, num_workers=0, collate_fn=collate_ctc_lines)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False,
                                   generator=torch.Generator().manual_seed(config.seed),
                                   num_workers=0, collate_fn=collate_ctc_lines)
    try:
        while completed < config.epochs and bad_epochs < config.patience:
            epoch = completed + 1
            model.train()
            started, total_loss, seen = perf_counter(), 0.0, 0
            _json(root / "status.json", {"status": "training", "epoch": epoch,
                                         "completed_epochs": completed, "requested_epochs": config.epochs})
            for batch_index, batch in enumerate(train_loader, 1):
                if (root / "STOP").exists():
                    raise KeyboardInterrupt("STOP file requested pause")
                optimizer.zero_grad(set_to_none=True)
                scores, lengths = model(batch["images"].to(device), batch["image_widths"])
                loss = ctc_batch_loss(scores, lengths, batch, train.vocabulary.blank_id)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss; no invalid samples are skipped")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                count = len(batch["sample_ids"])
                total_loss += float(loss.detach()) * count
                seen += count
                if batch_index == 1 or batch_index % config.log_every_batches == 0 or seen == len(train):
                    elapsed = perf_counter() - started
                    progress = {"epoch": epoch, "seen": seen, "training_count": len(train),
                                "mean_loss": total_loss / seen, "seconds": elapsed,
                                "estimated_remaining_epoch_seconds": elapsed / seen * (len(train) - seen)}
                    _json(root / "progress.json", progress)
                    print(f"{root.name} epoch {epoch}/{config.epochs}: {seen}/{len(train)} "
                          f"loss={total_loss / seen:.4f}, {elapsed:.0f}s", flush=True)
            training_seconds = perf_counter() - started
            model.eval()
            started, validation_loss, predictions = perf_counter(), 0.0, []
            with torch.inference_mode():
                for batch in validation_loader:
                    if (root / "STOP").exists():
                        raise KeyboardInterrupt("STOP file requested pause")
                    scores, lengths = model(batch["images"].to(device), batch["image_widths"])
                    loss = ctc_batch_loss(scores, lengths, batch, train.vocabulary.blank_id)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite validation loss")
                    validation_loss += float(loss) * len(batch["sample_ids"])
                    texts = greedy_ctc_decode(scores, lengths, train.vocabulary)
                    for sample_id, source_index, ground_truth, text in zip(
                            batch["sample_ids"], batch["source_indices"], batch["ground_truth"], texts):
                        predictions.append({"sample_id": sample_id, "source_index": source_index,
                                            "ground_truth": ground_truth, "prediction": text})
            metrics = corpus_metrics([r["ground_truth"] for r in predictions], [r["prediction"] for r in predictions])
            row = {"epoch": epoch, "train_loss": total_loss / seen,
                   "validation_loss": validation_loss / len(validation),
                   "validation_cer": metrics["cer"], "validation_wer": metrics["wer"],
                   "train_count": seen, "validation_count": len(predictions),
                   "training_seconds": training_seconds, "validation_seconds": perf_counter() - started,
                   "optimizer_steps": steps}
            history.append(row)
            score = (row["validation_cer"], row["validation_wer"], row["validation_loss"])
            improved = best is None or score < tuple(best)
            best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
            completed = epoch
            _csv(root / f"validation_epoch_{epoch:03d}.csv", predictions)
            _save(last, checkpoint())
            if improved:
                _save(root / "best.pt", checkpoint())
            _csv(root / "history.csv", history)
            plot_history(root)
            print(f"{root.name} epoch {epoch}: validation CER={metrics['cer']:.4f}, "
                  f"WER={metrics['wer']:.4f}, best={improved}", flush=True)
    except (KeyboardInterrupt, Exception) as error:
        _json(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                     "reason": str(error), "completed_epochs": completed,
                                     "resume_policy": "restore last completed epoch; repeat partial epoch"})
        raise
    result = {"status": "early_stopped" if bad_epochs >= config.patience else "complete",
              "completed_epochs": completed, "requested_epochs": config.epochs,
              "best_validation_score": best, "best_checkpoint": "best.pt",
              "test_evaluated": False}
    _json(root / "status.json", result)
    return result


def train_dataset(name: str, output: Path, config: TrainingConfig,
                  preparation_dir: Path = DEFAULT_PREPARATION, resume: bool = False,
                  smoke_samples: int | None = None) -> dict:
    """Load only original train/validation splits using the recorded revision."""
    from src.load_dataset.iam import load_iam_dataset
    from src.load_dataset.bentham import load_bentham_dataset, REVISION

    saved = json.loads((preparation_dir / "metadata.json").read_text(encoding="utf-8"))
    reference = saved["datasets"][name]
    if name == "iam":
        training = load_iam_dataset(split="train", revision=reference["revision"])
        validation = load_iam_dataset(split="validation", revision=reference["revision"])
    elif name == "bentham":
        if REVISION != reference["revision"]:
            raise ValueError("Bentham source revision differs from preparation")
        training, validation = load_bentham_dataset("train"), load_bentham_dataset("validation")
    else:
        raise ValueError("Choose iam or bentham")
    if training._fingerprint != reference["train_fingerprint"] or validation._fingerprint != reference["validation_fingerprint"]:
        raise ValueError("Dataset fingerprints differ from preparation")
    vocabulary = CharacterVocabulary(tuple(reference["vocabulary"]["characters"]))
    # Rebuild from training only to verify the saved label mapping.
    if vocabulary != CharacterVocabulary.from_training_split(training):
        raise ValueError("Saved vocabulary differs from training-derived characters")
    preparation = CTCImagePreparation(saved["mandatory_input"]["height"])
    validation_indices = reference["validation_source_indices"]
    train_indices = None
    if smoke_samples is not None:
        if smoke_samples < 1:
            raise ValueError("smoke_samples must be positive")
        train_indices = list(range(min(smoke_samples, len(training))))
        validation_indices = validation_indices[:min(smoke_samples, len(validation_indices))]
    train = CTCLineDataset(training, vocabulary, preparation, indices=train_indices)
    validation = CTCLineDataset(validation, vocabulary, preparation, indices=validation_indices)
    torch.manual_seed(config.seed)
    model = CNNLSTMCTC(CNNLSTMCTCConfig(vocabulary.num_classes, input_height=preparation.height))
    return fit(model, train, validation, output, config, resume=resume,
               provenance={"dataset": name, "revision": reference["revision"],
                           "preparation_reference": str(preparation_dir.resolve()),
                           "phase": "smoke_test" if smoke_samples else "full_training",
                           "smoke_samples": smoke_samples})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("iam", "bentham"), default=["iam", "bentham"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-samples", type=int)
    args = parser.parse_args()
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                            cpu_threads=args.threads, device=args.device)
    for name in args.datasets:
        train_dataset(name, args.output / name, config, args.preparation, args.resume, args.smoke_samples)


if __name__ == "__main__":
    main()
