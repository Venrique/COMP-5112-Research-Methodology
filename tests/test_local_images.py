"""Local image ingestion and partial-label scoring without model downloads."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.load_dataset.local import load_local_samples
from src.evaluation import compare_recognition
from src.preprocessing import PreprocessingPipeline


class LocalImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "line.png"
        Image.new("RGBA", (12, 8), (0, 0, 0, 0)).save(self.image)
        self.manifest = self.root / "manifest.csv"

    def write_rows(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "image_path", "ground_truth", "split", "condition"])
            writer.writerows(rows)

    def test_decode_preserves_original_and_raw_label(self):
        original = self.image.read_bytes()
        self.write_rows([["one", "line.png", " Hello, world! ", "validation", "shadow"]])
        sample, = load_local_samples(self.manifest)
        self.assertEqual(sample["ground_truth"], " Hello, world! ")
        self.assertEqual(sample["image"].mode, "RGB")
        self.assertEqual(sample["image"].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(sample["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(self.image.read_bytes(), original)

    def test_missing_label_and_split_filter(self):
        self.write_rows([["one", "line.png", "  ", "validation", ""],
                         ["two", "not_uploaded.jpg", "test", "test", ""]])
        samples = load_local_samples(self.manifest)
        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0]["ground_truth"])

    def test_duplicate_and_missing_image_rejected(self):
        self.write_rows([["one", "line.png", "a", "validation", ""],
                         ["two", "line.png", "a", "validation", ""]])
        with self.assertRaises(ValueError):
            load_local_samples(self.manifest)
        self.write_rows([["one", "missing.jpg", "a", "validation", ""]])
        with self.assertRaises(FileNotFoundError):
            load_local_samples(self.manifest)

    def test_partial_and_unlabeled_corpus(self):
        class Stub:
            def predict(self, image):
                return "hello"

        image = Image.new("RGB", (10, 10), "white")
        samples = [{"id": "one", "image": image, "ground_truth": "hello"},
                   {"id": "two", "image": image, "ground_truth": None}]
        rows, summary = compare_recognition(samples, Stub(), {"baseline": PreprocessingPipeline()})
        self.assertIsNone(rows[1]["cer"])
        self.assertEqual(summary[0]["sample_count"], 2)
        self.assertEqual(summary[0]["scored_sample_count"], 1)
        self.assertEqual(summary[0]["cer"], 0)
        _, summary = compare_recognition(samples[1:], Stub(), {"baseline": PreprocessingPipeline()})
        self.assertEqual(summary[0]["scored_sample_count"], 0)
        self.assertIsNone(summary[0]["cer"])
