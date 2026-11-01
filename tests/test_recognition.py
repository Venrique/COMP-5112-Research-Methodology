"""Offline checks using known edit counts and a model-independent stub."""
import unittest
from PIL import Image
from src.evaluation import compare_recognition, corpus_metrics, normalize_text
from src.preprocessing import PreprocessingPipeline


class RecognitionTests(unittest.TestCase):
    def test_corpus_weighting(self):
        scores = corpus_metrics(["a", "bbbb"], ["x", "bbbb"])
        self.assertAlmostEqual(scores["cer"], 1 / 5)
        self.assertAlmostEqual(scores["wer"], 1 / 2)

    def test_normalization(self):
        self.assertEqual(normalize_text("  Hello,\n World!  "), "Hello, World!")
        self.assertGreater(corpus_metrics(["Hello"], ["hello"])["cer"], 0)

    def test_insertions_deletions_and_empty_prediction(self):
        self.assertEqual(corpus_metrics(["ab"], [""])["cer"], 1)
        self.assertAlmostEqual(corpus_metrics(["ab"], ["abc"])["cer"], .5)
        with self.assertRaises(ValueError):
            corpus_metrics([], [])
        with self.assertRaises(ValueError):
            corpus_metrics(["a"], [])

    def test_comparison_accepts_another_model_and_preserves_image(self):
        class Stub:
            def predict(self, image):
                image.putpixel((0, 0), 0)
                return "hello"

        image = Image.new("L", (8, 8), 255)
        rows, summary = compare_recognition(
            [{"id": "one", "image": image, "ground_truth": "hello"}], Stub(),
            {"baseline": PreprocessingPipeline(), "other": PreprocessingPipeline()})
        self.assertEqual(image.getpixel((0, 0)), 255)
        self.assertEqual([row["sample_id"] for row in rows], ["one", "one"])
        self.assertTrue(all(row["cer"] == 0 for row in summary))


if __name__ == "__main__":
    unittest.main()
