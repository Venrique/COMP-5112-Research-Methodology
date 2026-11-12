"""Prespecified height contrasts, independent of intensity normalization."""
from src.preprocessing import SizeNormalization
from src.validation.denoising import evaluate, summarize


def configurations() -> dict:
    """Compare five heights on the same samples with fixed interpolation."""
    return {"baseline": None, **{
        f"size_height{height}": SizeNormalization(target_height=height)
        for height in (32, 48, 64, 96, 128)
    }}
