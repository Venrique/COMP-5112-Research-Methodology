"""Bounded projection-profile deskewing for dark handwriting on light paper."""
from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Deskew:
    """Maximize squared differences of adjacent ink-row sums.

    Otsu binarization is used only for angle estimation; output retains the
    original L/RGB pixels with bicubic rotation and a white expanded canvas.
    Search includes zero and both bounds. Ties prefer the smallest rotation.
    Defaults are initial settings, not selected using recognition results.
    This corrects line rotation, not individual character slant or perspective.
    """

    enabled: bool = True
    max_angle: float = 10.0
    angle_step: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        for name in ("max_angle", "angle_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_angle > 45 or self.angle_step > self.max_angle:
            raise ValueError("Require angle_step <= max_angle <= 45 degrees")
        if self.max_angle / self.angle_step > 1000:
            raise ValueError("Search must use at most 1000 steps per direction")

    def estimate_angle(self, image: Image.Image) -> float:
        """Return correction in Pillow degrees (positive counterclockwise)."""
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"Deskew requires L or RGB images, got {image.mode}")
        import cv2

        gray = np.asarray(image.convert("L"))
        if gray.min() == gray.max():
            return 0.0
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.count_nonzero(ink) < 10:
            return 0.0
        mask = Image.fromarray(ink)
        positive = np.append(np.arange(self.angle_step, self.max_angle, self.angle_step), self.max_angle)
        angles = sorted([0.0, *positive, *(-positive)], key=lambda a: (abs(a), a))
        best_angle, best_score = 0.0, -1.0
        for angle in angles:
            rotated = mask.rotate(float(angle), resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
            profile = np.asarray(rotated, dtype=float).sum(axis=1) / 255
            score = float(np.square(np.diff(np.pad(profile, (1, 1)))).sum())
            if score > best_score:
                best_angle, best_score = float(angle), score
        return best_angle

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image.copy()
        angle = self.estimate_angle(image)
        if angle == 0:
            return image.copy()
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True,
                            fillcolor=255 if image.mode == "L" else (255, 255, 255))

    def to_config(self) -> dict:
        """Record parameters and fixed implementation choices."""
        return {"name": "deskew", **asdict(self), "estimator": "otsu_projection_difference",
                "resampling": "bicubic", "expand": True, "fill": "white"}
