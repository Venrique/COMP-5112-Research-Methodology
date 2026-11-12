"""Offline checks for Bentham's line pairing and fixed partitions."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.load_dataset.bentham import get_sample, load_bentham_dataset


class BenthamTests(unittest.TestCase):
    def test_partitions_pairing_and_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "BenthamDatasetR0-GT"
            for folder in ("Partitions", "Images/Lines", "Transcriptions"):
                (root / folder).mkdir(parents=True)
            for split, name in (("train", "Train"), ("validation", "Validation"),
                                ("test", "Test")):
                (root / "Partitions" / f"{name}Lines.lst").write_text(split + "\n")
                Image.new("L", (12, 3), 123).save(root / "Images/Lines" / f"{split}.png")
                (root / "Transcriptions" / f"{split}.txt").write_text(
                    "  Original , text !  \n", encoding="utf-8")
                ds = load_bentham_dataset(split, directory)
                sample = get_sample(ds, 0)
                self.assertEqual(sample["id"], split)
                self.assertEqual(sample["ground_truth"], "  Original , text !  ")
                self.assertEqual(sample["image"].mode, "L")
                self.assertEqual(sample["image"].getpixel((0, 0)), 123)
                self.assertEqual(len(ds), 1)
            (root / "Partitions/ValidationLines.lst").write_text("validation\nvalidation\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_bentham_dataset(data_dir=directory)
            (root / "Partitions/ValidationLines.lst").write_text("missing\n")
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                load_bentham_dataset(data_dir=directory)

    def test_invalid_split_and_missing_download(self):
        with self.assertRaises(ValueError):
            load_bentham_dataset("dev")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "--download"):
                load_bentham_dataset(data_dir=directory)
