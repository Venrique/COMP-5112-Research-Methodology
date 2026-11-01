"""Pretrained TrOCR inference; no training or optional preprocessing."""
from copy import deepcopy
from PIL import Image


class TrOCRRecognizer:
    """Load once, then reuse for every preprocessing configuration.

    RGB conversion, resizing and normalization belong to the processor and
    always run after optional preprocessing. CPU is the portable default.
    """

    def __init__(self, checkpoint: str = "microsoft/trocr-base-handwritten",
                 revision: str = "main", device: str = "cpu",
                 generation_kwargs: dict | None = None):
        import torch
        from huggingface_hub import HfApi
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.checkpoint = checkpoint
        self.revision = HfApi().model_info(checkpoint, revision=revision).sha
        self.device = torch.device(device)
        self.generation_kwargs = {"max_new_tokens": 128, "num_beams": 1,
                                  "do_sample": False}
        self.generation_kwargs.update(deepcopy(generation_kwargs or {}))
        if self.generation_kwargs.get("do_sample"):
            raise ValueError("Use deterministic decoding for controlled comparisons")
        self.processor = TrOCRProcessor.from_pretrained(checkpoint, revision=self.revision)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint, revision=self.revision).to(self.device).eval()

    def predict(self, image: Image.Image) -> str:
        """Recognize a single text line without mutating its image."""
        import torch

        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        with torch.inference_mode():
            ids = self.model.generate(pixels.to(self.device), **self.generation_kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def to_config(self) -> dict:
        """Include resolved checkpoint, processor and decoding configuration."""
        return {"architecture": "TrOCR", "checkpoint": self.checkpoint,
                "revision": self.revision, "device": str(self.device),
                "generation_kwargs": deepcopy(self.generation_kwargs),
                "generation_config": self.model.generation_config.to_dict(),
                "image_processor": self.processor.image_processor.to_dict()}
