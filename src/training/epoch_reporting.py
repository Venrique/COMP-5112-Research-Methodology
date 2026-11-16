"""Document recorded validation results without touching the training algorithm.

Reports are derived from completed history and saved predictions only. This
observer does not load datasets, change model weights, or evaluate the test set.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import torch

from src.evaluation.recognition import corpus_metrics, normalize_text


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def archive_checkpoints(root: Path, signature: str, completed: set[int]) -> dict[int, str]:
    """Preserve currently available epoch weights; never invent missing ones.

    last.pt/best.pt are atomically replaced by training. Read bytes first, then
    inspect that snapshot. Existing per-epoch archives are never overwritten.
    """
    folder = root / "epoch_checkpoints"
    folder.mkdir(exist_ok=True)
    for filename in ("last.pt", "best.pt"):
        source = root / filename
        if not source.exists():
            continue
        data = source.read_bytes()
        saved = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        epoch = int(saved.get("completed_epochs", 0))
        if epoch not in completed or int(saved.get("optimizer_steps", 0)) < 1:
            continue
        if saved.get("signature") != signature:
            raise ValueError("Checkpoint signature differs from this run")
        target = folder / f"epoch_{epoch:03d}.pt"
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    return {int(path.stem.split("_")[-1]): path.relative_to(root).as_posix()
            for path in folder.glob("epoch_*.pt")}


def generate_report(run_dir: str | Path, planned_epochs: int = 20,
                    example_count: int = 3) -> Path | None:
    """Verify corpus CER/WER against each CSV and write a human-readable report.

    Only completed epochs appear. An epoch's checkpoint can be absent if it was
    overwritten before archiving began; its original scores and predictions are
    still documented. CSVs retain full-precision metrics; Markdown rounds them.
    """
    if planned_epochs < 1 or example_count < 1:
        raise ValueError("planned_epochs and example_count must be positive")
    root = Path(run_dir)
    history_path = root / "history.csv"
    if not history_path.exists():
        return None
    history = _read_csv(history_path)
    if not history:
        return None
    manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    expected_indices = identity["validation_indices"]
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError("History must contain consecutive completed epochs")
    summaries, examples = [], {}
    reference_texts = None
    previous, best = None, None
    for row in history:
        epoch = int(row["epoch"])
        filename = f"validation_epoch_{epoch:03d}.csv"
        predictions = _read_csv(root / filename)
        if [int(p["source_index"]) for p in predictions] != expected_indices:
            raise ValueError(f"Epoch {epoch} validation cohort differs from the saved protocol")
        if len(predictions) != int(row["validation_count"]):
            raise ValueError(f"Epoch {epoch} validation count mismatch")
        texts = [(p["sample_id"], p["ground_truth"]) for p in predictions]
        if reference_texts is not None and texts != reference_texts:
            raise ValueError("Sample IDs or original ground truth changed between epochs")
        reference_texts = texts
        measured = corpus_metrics([p["ground_truth"] for p in predictions], [p["prediction"] for p in predictions])
        for metric in ("cer", "wer"):
            if abs(measured[metric] - float(row[f"validation_{metric}"])) > 1e-12:
                raise ValueError(f"Epoch {epoch} saved {metric.upper()} disagrees with its predictions")
        score = (measured["cer"], measured["wer"], float(row["validation_loss"]))
        improved = best is None or score < best
        if improved:
            best, best_epoch = score, epoch
        summary = {
            "epoch": epoch, "train_loss": float(row["train_loss"]),
            "validation_loss": float(row["validation_loss"]),
            "validation_cer": measured["cer"], "validation_wer": measured["wer"],
            "validation_cer_pct": 100 * measured["cer"], "validation_wer_pct": 100 * measured["wer"],
            "cer_change_vs_previous_pp": "" if previous is None else 100 * (measured["cer"] - previous["cer"]),
            "wer_change_vs_previous_pp": "" if previous is None else 100 * (measured["wer"] - previous["wer"]),
            "exact_text_lines": sum(normalize_text(p["ground_truth"]) == normalize_text(p["prediction"]) for p in predictions),
            "empty_predictions": sum(not normalize_text(p["prediction"]) for p in predictions),
            "train_count": int(row["train_count"]), "validation_count": len(predictions),
            "training_minutes_elapsed": float(row["training_seconds"]) / 60,
            "validation_seconds": float(row["validation_seconds"]),
            "optimizer_steps": int(row["optimizer_steps"]),
            "best_so_far_at_epoch": improved, "prediction_file": filename,
        }
        previous = measured
        summaries.append(summary)
        examples[epoch] = predictions[:example_count]
    archives = archive_checkpoints(root, manifest["signature"], set(epochs))
    for summary in summaries:
        relative = archives.get(summary["epoch"], "")
        summary["epoch_checkpoint"] = relative
        summary["checkpoint_sha256"] = hashlib.sha256((root / relative).read_bytes()).hexdigest() if relative else ""
    table_path = root / "epoch_summary.csv"
    temporary = table_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    temporary.replace(table_path)
    provenance = identity.get("provenance", {})
    lines = [
        "# CNN + LSTM + CTC: epoch results", "",
        "Generated from recorded history and validation predictions. Keep manual discussion in a separate file; this report refreshes automatically.", "",
        f"- Dataset: **{provenance.get('dataset', 'recorded dataset').upper()}**; original training split, no optional preprocessing.",
        f"- Training lines: **{len(identity['train_indices']):,}**. Fixed validation lines: **{len(expected_indices)}**.",
        f"- Requested maximum: **{planned_epochs} epochs**; **{len(history)} completed**. Early stopping still applies.",
        f"- Dataset revision: `{provenance.get('revision', 'see metadata.json')}`.",
        f"- Seed: `{identity['training']['seed']}`; full model, vocabulary, source fingerprints, software versions, and original settings: [metadata.json](metadata.json).",
        "- Scores are **validation results**, not final test results. Corpus CER/WER are recomputed from each saved prediction CSV as a consistency check.",
        "- Scoring collapses whitespace and preserves case/punctuation. Lower loss, CER, and WER are better. CER is an edit rate, not a simple accuracy percentage.",
        "- Training loss is averaged while weights change; validation loss is measured afterward with fixed weights. Loss alone does not establish reading quality.",
        "- Training time is elapsed time and can include standby pauses; do not treat it as uninterrupted compute time.", "",
        "## Completed epochs", "",
        "| Epoch | Train loss | Validation loss | CER (%) | WER (%) | CER change (pp) | WER change (pp) | Exact lines | Empty outputs | Train min | Predictions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        delta_cer = "-" if row["cer_change_vs_previous_pp"] == "" else f"{row['cer_change_vs_previous_pp']:+.2f}"
        delta_wer = "-" if row["wer_change_vs_previous_pp"] == "" else f"{row['wer_change_vs_previous_pp']:+.2f}"
        lines.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['validation_loss']:.4f} | "
                     f"{row['validation_cer_pct']:.2f} | {row['validation_wer_pct']:.2f} | {delta_cer} | {delta_wer} | "
                     f"{row['exact_text_lines']}/{row['validation_count']} | {row['empty_predictions']} | "
                     f"{row['training_minutes_elapsed']:.1f} | [CSV]({row['prediction_file']}) |")
    lines += ["", "Changes are versus the previous epoch in percentage points (pp); negative means fewer errors.", "",
              f"**Best completed epoch: {best_epoch}**, selected by validation CER, then WER, then validation loss. This can differ from the latest epoch.", "",
              "[Full-precision epoch table](epoch_summary.csv) · [Original history](history.csv) · [Loss/error graphs](training_curves.png)", "",
              "## Predictions and checkpoint notes by epoch", "",
              "The same first validation sample IDs are shown each epoch; these are fixed examples, not chosen to look successful."]
    for row in summaries:
        epoch = row["epoch"]
        lines += ["", f"### Epoch {epoch}", "",
                  f"- Validation CER: **{row['validation_cer_pct']:.2f}%**; WER: **{row['validation_wer_pct']:.2f}%**.",
                  f"- Exact normalized text lines: **{row['exact_text_lines']}/{row['validation_count']}**; empty predictions: **{row['empty_predictions']}**."]
        if row["epoch_checkpoint"]:
            lines.append(f"- Archived weights and training state: [{row['epoch_checkpoint']}]({row['epoch_checkpoint']}). SHA-256 is recorded in `epoch_summary.csv`.")
        else:
            lines.append("- No separate weights archive is available for this epoch: archiving began later. Its original scores and prediction CSV are preserved; weights have not been reconstructed.")
        lines += ["", "| Sample | Correct text | Predicted text |", "| --- | --- | --- |"]
        for sample in examples[epoch]:
            lines.append(f"| {_escape(sample['sample_id'])} | {_escape(sample['ground_truth'])} | {_escape(sample['prediction']) or '(empty)'} |")
    path = root / "EPOCH_RESULTS.md"
    _write_text(path, "\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    print(generate_report(args.run_dir, args.epochs))
