"""RIMES French text lines with the publisher's original split boundaries."""
from datasets import load_dataset

DATASET_ID = "Teklia/RIMES-2011-line"
FRENCH_CHECKPOINT = "dj0w/trocr-french-handwriting-v5"
FRENCH_MODEL_REVISION = "6047032cbe5f3c7f17244a4c3dd21d9356f15480"


def load_rimes_dataset(split: str = "validation", revision: str | None = None):
    """Load one predefined split without changing images or French labels.

    Pass a resolved Hub commit for reproducibility. Development defaults to
    validation; test must be requested explicitly for final evaluation.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    return load_dataset(DATASET_ID, split=split, revision=revision)
