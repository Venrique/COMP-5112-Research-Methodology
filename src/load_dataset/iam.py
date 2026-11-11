from datasets import load_dataset


def load_iam_dataset(split: str = "test", revision: str | None = None):
    """
    Load the IAM line-level handwriting dataset
    from Hugging Face.

    Args:
        split: 'train', 'validation', or 'test'
        revision: Hugging Face dataset commit, tag, or branch. Use an exact
            commit and record it in experiment metadata for reproducible runs.
            None preserves the default Hugging Face revision behavior.

    Returns:
        Hugging Face Dataset containing images
        and ground-truth transcriptions.
    """

    return load_dataset(
        "Teklia/IAM-line",
        split=split,
        revision=revision,
    )


def get_sample(dataset, index):
    """
    Return an image and its ground truth.
    """

    sample = dataset[index]

    return {
        "id": f"sample_{index}",
        "image": sample["image"],
        "ground_truth": sample["text"]
    }
