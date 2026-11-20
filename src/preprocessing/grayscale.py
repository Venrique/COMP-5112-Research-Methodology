"""Optional grayscale conversion, independent of model input transforms."""
from dataclasses import asdict, dataclass

from PIL import Image


# Freeze the transform settings once an instance is created.
@dataclass(frozen=True)
class Grayscale:
    """Convert to 8-bit luminance; existing L images retain their pixels."""

    enabled: bool = True

    def __post_init__(self) -> None:
        # Type hints are checked separately, so enforce the Boolean flag at runtime.
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")

    def __call__(self, image: Image.Image) -> Image.Image:
        # A disabled transform returns a separate image with the same mode and pixels.
        if not self.enabled:
            return image.copy()
        # Accept only grayscale and RGB inputs when conversion is enabled.
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"Grayscale requires L or RGB images, got {image.mode}")
        # Pillow's L mode stores one 8-bit luminance value per pixel.
        return image.convert("L")

    def to_config(self) -> dict:
        # Include the enabled flag and the target mode for grayscale conversion.
        return {"name": "grayscale", **asdict(self), "output_mode": "L"}
