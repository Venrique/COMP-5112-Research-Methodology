"""Optional grayscale conversion, independent of model input transforms."""
from dataclasses import asdict, dataclass

from PIL import Image


@dataclass(frozen=True)
class Grayscale:
    """Convert to 8-bit luminance; existing L images retain their pixels."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image.copy()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"Grayscale requires L or RGB images, got {image.mode}")
        return image.convert("L")

    def to_config(self) -> dict:
        return {"name": "grayscale", **asdict(self), "output_mode": "L"}
