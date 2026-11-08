"""Load local line images from a UTF-8 CSV manifest without altering files."""
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageOps


def load_local_samples(manifest: str | Path, split: str = "validation") -> list[dict]:
    """Resolve image paths relative to manifest; blank labels mean unscored.

    Decode EXIF orientation and convert to RGB for all configurations. Composite
    transparency onto white. Originals are never overwritten. Only selected
    split images are decoded; IDs and paths must be unique across the manifest.
    """
    manifest = Path(manifest).resolve()
    samples, ids, paths = [], set(), set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "image_path", "ground_truth", "split", "condition"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row; quote text containing commas")
            sample_id, relative = row["id"].strip(), row["image_path"].strip()
            if not sample_id or not relative:
                raise ValueError("Each row requires an id and image_path")
            path = (manifest.parent / relative).resolve()
            if sample_id in ids or path in paths:
                raise ValueError(f"Duplicate id or image path: {sample_id}")
            ids.add(sample_id)
            paths.add(path)
            if row["split"].strip() not in {"validation", "test"}:
                raise ValueError("Local split must be validation or test")
            if row["split"].strip() != split:
                continue
            with Image.open(path) as original:
                oriented = ImageOps.exif_transpose(original)
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
                original_mode, original_size = original.mode, original.size
            samples.append({"id": sample_id, "image": image,
                            "ground_truth": row["ground_truth"] if row["ground_truth"].strip() else None,
                            "image_path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "condition": row["condition"], "split": split,
                            "original_mode": original_mode, "original_size": original_size})
    if not samples:
        raise ValueError(f"No images selected for split {split!r}")
    return samples
