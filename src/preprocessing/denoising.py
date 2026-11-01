"""Non-local means denoising for PIL images."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NonLocalMeans:
    """OpenCV fast non-local means for 8-bit L or RGB images.

    h controls grayscale/luminance smoothing; h_color controls RGB chroma
    smoothing. Larger strengths can remove handwriting detail. These defaults
    are starting parameters, not settings selected using recognition results.
    Unsupported modes fail explicitly rather than adding an implicit transform.
    """

    enabled: bool = True
    h: float = 3.0
    h_color: float = 3.0
    template_window_size: int = 7
    search_window_size: int = 21

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        for name in ("h", "h_color"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("template_window_size", "search_window_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.search_window_size < self.template_window_size:
            raise ValueError("search_window_size must be >= template_window_size")

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image.copy()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"NonLocalMeans requires L or RGB images, got {image.mode}")
        # Lazy import keeps the no-preprocessing baseline independent of OpenCV.
        import cv2

        pixels = np.array(image)
        if image.mode == "L":
            result = cv2.fastNlMeansDenoising(
                pixels, None, self.h, self.template_window_size, self.search_window_size
            )
        else:
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            result = cv2.fastNlMeansDenoisingColored(
                bgr, None, self.h, self.h_color,
                self.template_window_size, self.search_window_size,
            )
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

    def to_config(self) -> dict:
        """Return JSON-serializable parameters, including the enabled flag."""
        return {"name": "non_local_means", **asdict(self)}
