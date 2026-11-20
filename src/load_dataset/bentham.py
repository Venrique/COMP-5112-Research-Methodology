"""Local Bentham R0 line images with the publisher's fixed partitions.

Download once with ``python -m src.load_dataset.bentham --download``.
Loading is offline and never extracts or changes the source files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from datasets import Dataset, Image

# Pin the Hugging Face dataset to a specific commit for reproducible downloads.
REPOSITORY = "staghado/Bentham"
REVISION = "bfd99574602e2abbe7896255b1fd5e114f68571a"
# Two parent directories above src/load_dataset is the project root.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bentham"
# Each archive name maps to the SHA-256 digest of its expected contents.
ARCHIVES = {
    "BenthamDatasetR0-GT.tbz": "369b86b0953416df93aa0daaf521c00305a73c78ad86a46a4316c4adfd7515b3",
    "BenthamDatasetR0-Images.tbz": "b389c33c0813c447304bba0f49b46d39a0a8220b1bf8254f1222ae278f8a85c7",
}


def download_bentham_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download pinned archives, verify SHA-256, and safely extract originals.

    Requires about 9 GB for archives plus extracted files. Existing archives
    are verified and reused. Extraction may overwrite files from these archives.
    """
    # Import the download helper only when preparing the local files.
    from huggingface_hub import hf_hub_download

    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Reuse local archives and verify their bytes before each extraction.
    for filename, expected in ARCHIVES.items():
        archive = root / filename
        if not archive.exists():
            hf_hub_download(REPOSITORY, filename, repo_type="dataset",
                            revision=REVISION, local_dir=root)
        with archive.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {archive}; remove it and retry")
        # A .tbz is a bzip2-compressed tar archive. The data filter rejects
        # unsafe entries such as paths that would escape the destination.
        with tarfile.open(archive, "r:bz2") as handle:
            handle.extractall(root, filter="data")
    # Record the dataset version and checksums after both archives are extracted.
    (root / "source.json").write_text(
        json.dumps({"repository": REPOSITORY, "revision": REVISION,
                    "sha256": ARCHIVES}, indent=2) + "\n", encoding="utf-8")
    return root


def load_bentham_dataset(
    split: str = "validation", data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Dataset:
    """Return an IAM-compatible Dataset with id, image and text columns.

    Preserves partition order, original image modes/pixels and transcription
    punctuation/spacing (only the terminal file newline is removed). Images
    decode lazily. No random splitting, normalization or PAGE XML matching:
    PAGE line IDs do not necessarily match the supplied line image names.
    """
    # The publisher's partition lists define which line IDs belong to each split.
    partitions = {"train": "TrainLines.lst", "validation": "ValidationLines.lst",
                  "test": "TestLines.lst"}
    if split not in partitions:
        raise ValueError(f"split must be one of {tuple(partitions)}")
    root = Path(data_dir).resolve()
    gt = root / "BenthamDatasetR0-GT"
    partition = gt / "Partitions" / partitions[split]
    if not partition.is_file():
        raise FileNotFoundError(
            f"Missing {partition}. Run python -m src.load_dataset.bentham --download")
    # Read IDs in file order and require a nonempty list of unique, bare names.
    ids = partition.read_text(encoding="utf-8").splitlines()
    if not ids or any(not item or Path(item).name != item for item in ids):
        raise ValueError(f"Empty or malformed partition: {partition}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate line IDs in {partition}")
    # Match each ID to its PNG and transcription by their common filename stem.
    images, texts = [], []
    for line_id in ids:
        image = gt / "Images" / "Lines" / f"{line_id}.png"
        transcript = gt / "Transcriptions" / f"{line_id}.txt"
        if not image.is_file() or not transcript.is_file():
            raise FileNotFoundError(f"Missing image or transcription for {line_id} in {gt}")
        images.append(str(image))
        # Remove one final newline while preserving transcription spacing.
        texts.append(transcript.read_text(encoding="utf-8").removesuffix("\n"))
    # Store image paths and let the Image feature decode them when accessed.
    return Dataset.from_dict({"id": ids, "image": images, "text": texts},
                             split=split).cast_column("image", Image())


def get_sample(dataset: Dataset, index: int) -> dict:
    """Return a stable source ID, Pillow image and unchanged ground truth."""
    row = dataset[index]
    # Expose the dataset's text column under the sample key ground_truth.
    return {"id": row["id"], "image": row["image"], "ground_truth": row["text"]}


# Run the download command only when this module is executed directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print(download_bentham_dataset(args.data_dir))
