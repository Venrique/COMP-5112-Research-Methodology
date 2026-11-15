"""Prepare existing line datasets for CTC without changing their splits.

This module does not train a model or apply experimental preprocessing.
Training labels use the same whitespace policy as recognition evaluation.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.evaluation.recognition import normalize_text


@dataclass(frozen=True)
class CharacterVocabulary:
    """Deterministic character IDs; zero is reserved for the CTC blank.

    Unknown characters raise an error instead of silently altering the label.
    Build from training text only, then reuse unchanged for validation/test.
    """

    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.characters or any(len(c) != 1 for c in self.characters):
            raise ValueError("Provide nonempty, single-character vocabulary entries")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary characters must be unique")

    @classmethod
    def from_training_split(cls, dataset) -> "CharacterVocabulary":
        """Read training transcriptions only; do not decode dataset images."""
        if str(getattr(dataset, "split", "")) != "train":
            raise ValueError("Build the vocabulary from the train split only")
        characters = set()
        for text in dataset["text"]:
            if not isinstance(text, str) or not normalize_text(text):
                raise ValueError("Training transcriptions must be nonempty strings")
            characters.update(normalize_text(text))
        return cls(tuple(sorted(characters)))

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    def unknown_characters(self, texts: Iterable[str]) -> list[str]:
        """Report unsupported characters without extending the vocabulary."""
        known = set(self.characters)
        return sorted({c for text in texts for c in normalize_text(text)} - known)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a label, retaining case/punctuation and collapsing whitespace."""
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError("CTC labels must be nonempty strings")
        text = normalize_text(text)
        lookup = {character: index for index, character in enumerate(self.characters, 1)}
        unknown = sorted(set(text) - lookup.keys())
        if unknown:
            raise ValueError(f"Characters absent from training vocabulary: {unknown!r}")
        return torch.tensor([lookup[c] for c in text], dtype=torch.long)

    def decode_labels(self, ids: Iterable[int]) -> str:
        """Reverse label encoding, not CTC path decoding; retain repeated letters."""
        result = []
        for index in ids:
            index = int(index)
            if not 1 <= index < self.num_classes:
                raise ValueError("Label IDs must exclude blank and be in vocabulary")
            result.append(self.characters[index - 1])
        return "".join(result)

    def to_config(self) -> dict:
        return {"characters": list(self.characters), "blank_id": self.blank_id,
                "normalization": "collapse whitespace; retain case and punctuation",
                "source_split": "train"}


@dataclass(frozen=True)
class CTCImagePreparation:
    """Mandatory grayscale input, proportional resize, and pixels in [0, 1].

    Height 64 is an initial input specification, not a validation optimum.
    Width is never cropped or capped. Transparency is composited onto white.
    Apply these same transforms to baseline and every experimental treatment.
    """

    height: int = 64

    def __post_init__(self) -> None:
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("height must be a positive integer")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a Pillow image")
        rgba = image.convert("RGBA")
        gray = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("L")
        width = max(1, round(gray.width * self.height / gray.height))
        resized = gray.resize((width, self.height), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).unsqueeze(0)

    def to_config(self) -> dict:
        return {"height": self.height, "mode": "L", "preserve_aspect_ratio": True,
                "resampling": "LANCZOS", "pixel_range": [0, 1],
                "alpha_background": "white", "padding": "right, white (1.0)",
                "optional_preprocessing": "none"}


class CTCLineDataset(Dataset):
    """Wrap an existing split lazily, keeping source order and original text.

    Optional indices refer to the original split, so saved teammate sample
    selections can be reused without renaming IAM examples or splitting again.
    """

    def __init__(self, dataset, vocabulary: CharacterVocabulary,
                 preparation: CTCImagePreparation | None = None,
                 indices: Sequence[int] | None = None):
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.preparation = preparation or CTCImagePreparation()
        self.indices = tuple(range(len(dataset))) if indices is None else tuple(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Provide nonempty, unique source indices")
        if any(type(i) is not int or not 0 <= i < len(dataset) for i in self.indices):
            raise ValueError("Source indices must be integers within the existing split")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self.indices[index]
        row = self.dataset[source_index]
        targets = self.vocabulary.encode(row["text"])
        image = self.preparation(row["image"])
        # Consecutive repeated labels require separating blank time steps.
        repeats = int((targets[1:] == targets[:-1]).sum())
        return {"image": image, "targets": targets,
                "sample_id": row.get("id", f"sample_{source_index}"),
                "source_index": source_index, "ground_truth": row["text"],
                "training_text": normalize_text(row["text"]),
                "minimum_ctc_timesteps": len(targets) + repeats}


def collate_ctc_lines(samples: list[dict]) -> dict:
    """Right-pad images to batch width and concatenate unpadded CTC labels.

    image_widths are pixel widths, NOT CTC input_lengths. The future CNN must
    convert these widths to output sequence lengths and check them against
    minimum_ctc_timesteps before computing CTC loss. Never hide impossible
    alignments with zero_infinity or silently truncate labels/images.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if any(s["image"].ndim != 3 or s["image"].shape[0] != 1 for s in samples):
        raise ValueError("Images must have shape [1, height, width]")
    heights = {s["image"].shape[1] for s in samples}
    if len(heights) != 1:
        raise ValueError("All images in a batch must have the same prepared height")
    widths = torch.tensor([s["image"].shape[2] for s in samples], dtype=torch.long)
    images = torch.ones((len(samples), 1, heights.pop(), int(widths.max())), dtype=torch.float32)
    for index, sample in enumerate(samples):
        images[index, :, :, :sample["image"].shape[2]] = sample["image"]
    return {"images": images, "image_widths": widths,
            "targets": torch.cat([s["targets"] for s in samples]),
            "target_lengths": torch.tensor([len(s["targets"]) for s in samples], dtype=torch.long),
            "minimum_ctc_timesteps": torch.tensor([s["minimum_ctc_timesteps"] for s in samples], dtype=torch.long),
            "sample_ids": [s["sample_id"] for s in samples],
            "source_indices": [s["source_index"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
            "training_text": [s["training_text"] for s in samples]}
