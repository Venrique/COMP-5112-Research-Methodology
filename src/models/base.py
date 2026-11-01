"""Common interface independent of model architecture."""
from typing import Protocol
from PIL import Image


class HandwritingRecognizer(Protocol):
    """Models own mandatory input transforms and return unnormalized text."""

    def predict(self, image: Image.Image) -> str: ...

    def to_config(self) -> dict: ...
