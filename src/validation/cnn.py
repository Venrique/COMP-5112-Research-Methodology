"""CNN-specific individual preprocessing grids; validation only, no training."""
import hashlib
import json
from importlib import import_module
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.evaluation.recognition import corpus_metrics
from src.preprocessing import PreprocessingPipeline
from src.validation.denoising import adapt_image
from src.evaluation.selection import select_validation_configuration

TECHNIQUES = ("denoising", "deskew", "clahe", "grayscale",
              "intensity_normalization", "size_normalization")


def configurations() -> dict:
    """Reuse the prespecified TrOCR grids; do not adapt grids to test scores."""
    return {technique: import_module("src.validation." + technique).configurations()
            for technique in TECHNIQUES}


def evaluate(samples: list[dict], model, pipeline: PreprocessingPipeline,
             output_dir: Path) -> dict:
    """Save every ten predictions and refuse resume under a changed identity."""
    if not samples:
        raise ValueError("Provide nonempty validation samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"model": model.to_config(), "pipeline": pipeline.to_config(),
                "samples": [{"id": s["id"], "text": s["ground_truth"],
                             "mode": s["image"].mode, "size": s["image"].size,
                             "pixels": hashlib.sha256(s["image"].tobytes()).hexdigest()}
                            for s in samples],
                "normalization": "collapse whitespace; preserve case/punctuation"}
    encoded = json.dumps(identity, sort_keys=True)
    signature = hashlib.sha256(encoded.encode()).hexdigest()
    manifest = output_dir / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Resume identity mismatch")
    manifest.write_text(json.dumps({"signature": signature, **identity}, indent=2))
    path = output_dir / "predictions.csv"
    rows = pd.read_csv(path, keep_default_na=False).to_dict("records") if path.exists() else []
    if [r["sample_id"] for r in rows] != [s["id"] for s in samples[:len(rows)]] or len(rows) > len(samples):
        raise ValueError("Invalid checkpoint sample order")
    for i in range(len(rows), len(samples)):
        sample = samples[i]
        reference = sample["ground_truth"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Missing validation transcription")
        begin = perf_counter()
        image = pipeline(adapt_image(sample["image"]))
        preprocessing_seconds = perf_counter() - begin
        begin = perf_counter()
        prediction = model.predict(image)
        inference_seconds = perf_counter() - begin
        rows.append({"sample_id": sample["id"], "ground_truth": reference,
                     "prediction": prediction, "preprocessing_seconds": preprocessing_seconds,
                     "inference_seconds": inference_seconds,
                     **corpus_metrics([reference], [prediction])})
        if len(rows) % 10 == 0 or len(rows) == len(samples):
            temporary = path.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(path)
            print(f"{output_dir.name}: {len(rows)}/{len(samples)}", flush=True)
    summary = {"sample_count": len(rows),
               **corpus_metrics([r["ground_truth"] for r in rows], [r["prediction"] for r in rows]),
               "preprocessing_seconds": sum(r["preprocessing_seconds"] for r in rows),
               "inference_seconds": sum(r["inference_seconds"] for r in rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def select_settings(table: pd.DataFrame, grids: dict) -> dict:
    """Active CER/WER winner per technique; baseline wins equal-score overall ties."""
    selected = {}
    for technique, grid in grids.items():
        scores = table[table.technique == technique]
        active = select_validation_configuration(scores, "iam")
        winner = scores.sort_values(["cer", "wer"], kind="stable").iloc[0]
        step = grid[active["configuration"]]
        selected[technique] = {
            "configuration": active["configuration"], "parameters": step.to_config(),
            "cer": active["cer"], "wer": active["wer"],
            "winner_including_baseline": winner["configuration"],
            "improves_baseline": winner["configuration"] != "baseline"}
    return selected
