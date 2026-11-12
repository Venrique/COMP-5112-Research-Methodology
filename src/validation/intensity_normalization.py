"""Global min-max normalization has one active configuration, plus baseline."""
from src.preprocessing import IntensityNormalization
from src.validation.denoising import evaluate, summarize


def configurations() -> dict:
    """Evaluate intensity alone, without resizing or grayscale conversion."""
    return {"baseline": None, "intensity_normalization": IntensityNormalization()}
