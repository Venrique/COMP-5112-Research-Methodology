"""Model-independent evaluation; raw transcriptions remain unchanged."""
from time import perf_counter
from collections.abc import Mapping, Sequence
from jiwer import cer, wer
from src.models.base import HandwritingRecognizer
from src.preprocessing import PreprocessingPipeline


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip edges; retain case and punctuation."""
    return " ".join(text.split())


def corpus_metrics(references: Sequence[str], predictions: Sequence[str]) -> dict:
    """Corpus edit counts / reference counts, never mean per-line rates."""
    if not references or len(references) != len(predictions):
        raise ValueError("Provide nonempty, equally sized references and predictions")
    refs = [normalize_text(text) for text in references]
    preds = [normalize_text(text) for text in predictions]
    if not any(refs):
        raise ValueError("Corpus reference denominator must be nonzero")
    return {"cer": cer(refs, preds), "wer": wer(refs, preds)}


def compare_recognition(samples: Sequence[dict], model: HandwritingRecognizer,
                        pipelines: Mapping[str, PreprocessingPipeline]) -> tuple[list, list]:
    """Compare all configurations on identical ordered samples.

    Timings cover preprocessing and prediction separately; model loading is
    excluded. Per-image metrics are diagnostic; summaries use corpus metrics.
    """
    if not samples or not pipelines:
        raise ValueError("Samples and pipelines must be nonempty")
    rows, summaries = [], []
    for name, pipeline in pipelines.items():
        current = []
        for sample in samples:
            start = perf_counter()
            image = pipeline(sample["image"])
            preprocessing_seconds = perf_counter() - start
            start = perf_counter()
            prediction = model.predict(image)
            seconds = perf_counter() - start
            row = {"configuration": name, "sample_id": sample["id"],
                   "ground_truth": sample["ground_truth"], "prediction": prediction,
                   "preprocessing_seconds": preprocessing_seconds,
                   "inference_seconds": seconds,
                   **corpus_metrics([sample["ground_truth"]], [prediction])}
            current.append(row)
            print(f"{name}: {len(current)}/{len(samples)}", flush=True)
        summaries.append({"configuration": name, "sample_count": len(current),
                          **corpus_metrics([r["ground_truth"] for r in current],
                                           [r["prediction"] for r in current]),
                          "inference_seconds": sum(r["inference_seconds"] for r in current)})
        rows.extend(current)
    return rows, summaries
