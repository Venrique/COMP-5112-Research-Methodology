"""Contrast enhancement for PIL images using CLAHE."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CLAHE:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    L images are processed directly. RGB images are converted to the LAB
    color space and CLAHE is applied only to luminance, preserving chroma.
    Defaults are initial settings and should be selected on validation data.
    """

    enabled: bool = True
    # Limit histogram peaks to curb noise amplification in each tile.
    clip_limit: float = 2.0
    # OpenCV takes the number of tiles across and down the image.
    tile_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if (isinstance(self.clip_limit, bool) or
                not isinstance(self.clip_limit, (int, float)) or
                not math.isfinite(self.clip_limit) or self.clip_limit <= 0):
            raise ValueError("clip_limit must be finite and positive")
        if (not isinstance(self.tile_grid_size, tuple) or len(self.tile_grid_size) != 2 or
                any(type(size) is not int or size <= 0 for size in self.tile_grid_size)):
            raise ValueError("tile_grid_size must be a pair of positive integers")

    def __call__(self, image: Image.Image) -> Image.Image:
        # Bypass contrast adjustment and return a separate image when disabled.
        if not self.enabled:
            return image.copy()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"CLAHE requires L or RGB images, got {image.mode}")

        # Lazy import keeps baseline use independent of OpenCV.
        import cv2

        # Equalize contrast locally and interpolate between neighboring tiles.
        clahe = cv2.createCLAHE(clipLimit=float(self.clip_limit),
                                tileGridSize=self.tile_grid_size)
        pixels = np.asarray(image)
        if image.mode == "L":
            result = clahe.apply(pixels)
        else:
            # LAB channel 0 holds lightness. Leave the two color channels
            # untouched during equalization, then convert back to RGB.
            lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def to_config(self) -> dict:
        """Return JSON-serializable parameters and implementation details."""
        return {"name": "clahe", **asdict(self),
                "processing": "grayscale directly; RGB via LAB luminance"}
