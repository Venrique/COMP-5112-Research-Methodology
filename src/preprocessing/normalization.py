"""Independent optional intensity and spatial normalization of PIL images."""
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class IntensityNormalization:
    """Stretch the image-wide min/max to 0/255; constant images stay unchanged.

    A single range over all RGB channels avoids independent channel stretching.
    Mode and dimensions are preserved; no grayscale conversion is implied.
    """

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")

    def __call__(self, image: Image.Image) -> Image.Image:
        # Bypass normalization and return a separate image when disabled.
        if not self.enabled:
            return image.copy()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"IntensityNormalization requires L or RGB images, got {image.mode}")
        # Use floating-point arithmetic and one shared range across all channels.
        pixels = np.asarray(image, dtype=np.float64)
        low, high = pixels.min(), pixels.max()
        # A zero range has no contrast to stretch and would divide by zero.
        if low == high:
            return image.copy()
        # Map the minimum to 0 and the maximum to 255, then round to byte values.
        result = np.rint((pixels - low) / (high - low) * 255.0).clip(0, 255)
        return Image.fromarray(result.astype(np.uint8))

    def to_config(self) -> dict:
        """Record the fixed algorithm as well as the enable flag."""
        return {"name": "intensity_normalization", **asdict(self),
                "processing": "global min-max over all channels; round to uint8; constant unchanged",
                "output_range": [0, 255]}


@dataclass(frozen=True)
class SizeNormalization:
    """Resize to target height with LANCZOS and proportional integer width.

    Width is rounded to the nearest integer (at least one pixel). No cropping
    or padding is applied. Defaults are initial settings, not selected optima.
    """

    enabled: bool = True
    target_height: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        # Require a whole number of pixels and exclude Boolean values.
        if type(self.target_height) is not int or self.target_height <= 0:
            raise ValueError("target_height must be a positive integer")

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image.copy()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"SizeNormalization requires L or RGB images, got {image.mode}")
        # Scale both axes by the same factor, allowing for whole-pixel rounding.
        # Clamp the width so even very narrow images retain at least one column.
        width = max(1, round(image.width * self.target_height / image.height))
        return image.resize((width, self.target_height), Image.Resampling.LANCZOS)

    def to_config(self) -> dict:
        """Record dimensions policy and interpolation for reproducibility."""
        return {"name": "size_normalization", **asdict(self),
                "resampling": "LANCZOS", "preserve_aspect_ratio": True,
                "width_rounding": "nearest integer, ties to even; minimum 1"}
