"""UI-independent single-line recognition helpers; no models load on import."""
from dataclasses import dataclass, field
import hashlib
import json
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from src.evaluation.recognition import corpus_metrics, normalize_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_CHECKPOINT = ROOT / "results/cnn_lstm_ctc/training/20261114_baseline/iam/best.pt"
TROCR_CHECKPOINT = "microsoft/trocr-base-handwritten"
TROCR_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"
MODEL_LABELS = {"cnn_lstm_ctc": "CNN + LSTM + CTC", "trocr": "TrOCR"}
PREPROCESSING_LABELS = {
    "grayscale": "Grayscale",
    "clahe": "Contrast enhancement (CLAHE)",
    "denoising": "Denoising",
    "deskew": "Deskewing",
    "intensity": "Intensity normalization",
    "size": "Size normalization (height 64)",
}


def build_preprocessing(selected=()):
    """Build optional steps in fixed display order using project defaults.

    Defaults are demonstration settings, not validation-selected optima.
    An empty selection preserves the decoded upload's pixels and mode.
    """
    from src.preprocessing import (
        CLAHE, Deskew, Grayscale, IntensityNormalization, NonLocalMeans,
        PreprocessingPipeline, SizeNormalization,
    )

    selected = set(selected)
    if selected - PREPROCESSING_LABELS.keys():
        raise ValueError("Unknown preprocessing technique.")
    factories = (Grayscale, CLAHE, NonLocalMeans, Deskew,
                 IntensityNormalization, SizeNormalization)
    return PreprocessingPipeline(tuple(
        factory() for key, factory in zip(PREPROCESSING_LABELS, factories)
        if key in selected
    ))


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_upload(data: bytes) -> Image.Image:
    """Return an oriented white-composited RGB copy; never rewrite uploads."""
    if not data:
        raise ValueError("Choose an image first.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Please choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                if original.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Use a PNG, JPG, or WebP image.")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Please resize this image to 20 megapixels or less.")
                if getattr(original, "n_frames", 1) != 1:
                    raise ValueError("Choose one still image, rather than an animation.")
                original.load()
                oriented = ImageOps.exif_transpose(original).convert("RGBA")
                white = Image.new("RGBA", oriented.size, "white")
                return Image.alpha_composite(white, oriented).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("This image could not be opened. Try another PNG, JPG, or WebP file.") from error


def checkpoint_token(model_key: str, checkpoint: str | Path = DEFAULT_CNN_CHECKPOINT) -> str:
    """Cache identity without loading weights; missing local files are explicit."""
    if model_key == "trocr":
        return TROCR_REVISION
    if model_key != "cnn_lstm_ctc":
        raise ValueError("Choose CNN + LSTM + CTC or TrOCR.")
    path = Path(checkpoint).resolve()
    if not path.is_file():
        return "missing:" + str(path)
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def prediction_key(data: bytes, model_key: str, token: str,
                   preprocessing: dict | None = None) -> str:
    """Bind predictions to the image, model, weights and ordered preprocessing."""
    digest = hashlib.sha256()
    digest.update(data)
    digest.update(b"\0" + model_key.encode() + b"\0" + token.encode())
    digest.update(b"\0" + json.dumps(
        preprocessing if preprocessing is not None else {"steps": []},
        sort_keys=True, separators=(",", ":"),
    ).encode())
    return digest.hexdigest()


def score_prediction(prediction: str, ground_truth: str | None) -> dict | None:
    """Use the project's existing normalization/metrics; blank reference is unscored.

    Keep case/punctuation, collapse whitespace, and never clamp rates to 100%.
    Empty predictions with a nonempty reference are valid, scored failures.
    """
    if ground_truth is None or not normalize_text(ground_truth):
        return None
    return corpus_metrics([ground_truth], [prediction])


@dataclass
class RecognizerResource:
    """Serialize access to a model shared by Streamlit resource caching."""

    recognizer: object
    lock: object = field(default_factory=RLock, repr=False)

    def recognize(self, image: Image.Image) -> dict:
        with self.lock:
            begin = perf_counter()
            prediction = self.recognizer.predict(image.copy())
            elapsed = perf_counter() - begin
            return {"prediction": prediction, "inference_seconds": elapsed,
                    "model_config": self.recognizer.to_config()}


def load_resource(model_key: str, checkpoint: str | Path = DEFAULT_CNN_CHECKPOINT) -> RecognizerResource:
    """Load only the requested CPU model; called after a user presses Predict."""
    if model_key not in MODEL_LABELS:
        raise ValueError("Unknown model selection.")
    if model_key == "cnn_lstm_ctc" and not Path(checkpoint).is_file():
        raise FileNotFoundError("CNN checkpoint not found. Restore best.pt or set HTR_CNN_CHECKPOINT.")
    import torch
    from src.models import CNNLSTMCTCRecognizer, TrOCRRecognizer

    torch.set_num_threads(4)
    if model_key == "cnn_lstm_ctc":
        model = CNNLSTMCTCRecognizer.from_checkpoint(checkpoint, device="cpu")
    else:
        model = TrOCRRecognizer(
            checkpoint=TROCR_CHECKPOINT, revision=TROCR_REVISION, device="cpu",
            generation_kwargs={"max_new_tokens": 128, "num_beams": 1, "do_sample": False},
        )
    return RecognizerResource(model)
