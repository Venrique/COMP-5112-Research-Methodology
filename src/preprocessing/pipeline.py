"""Ordered, configurable preprocessing of PIL images without modifying inputs."""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class PreprocessingStep(Protocol):
    """Interface for independently enabled steps, applied in pipeline order."""

    enabled: bool

    def __call__(self, image: Image.Image) -> Image.Image: ...

    def to_config(self) -> dict: ...


@dataclass(frozen=True)
class PreprocessingPipeline:
    """Apply enabled steps in order; no steps means a pixel-identical copy.

    Reuse one pipeline across samples. Future techniques implement
    PreprocessingStep and are added to the ordered steps tuple.
    """

    steps: tuple[PreprocessingStep, ...] = ()

    def __call__(self, image: Image.Image) -> Image.Image:
        result = image.copy()
        for step in self.steps:
            if step.enabled:
                result = step(result)
        return result

    def to_config(self) -> dict:
        """Record all configured steps in execution order, including disabled ones."""
        return {"steps": [step.to_config() for step in self.steps]}
