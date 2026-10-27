from datasets import load_dataset


def load_iam_dataset(split="test"):
    """
    Load the IAM line-level handwriting dataset
    from Hugging Face.

    Args:
        split: 'train', 'validation', or 'test'

    Returns:
        Hugging Face Dataset containing images
        and ground-truth transcriptions.
    """

    return load_dataset(
        "Teklia/IAM-line",
        split=split
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
