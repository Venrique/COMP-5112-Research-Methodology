"""Validate resume integrity and baseline-aware CNN selection offline."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import pandas as pd
from src.preprocessing import PreprocessingPipeline, Grayscale
from src.validation.cnn import evaluate, select_settings


class Model:
    def __init__(self):
        self.calls = 0

    def to_config(self):
        return {"checkpoint": "test-only-stub"}

    def predict(self, image):
        self.calls += 1
        return "hello"


class CNNValidationTests(unittest.TestCase):
    def test_resume_and_changed_samples(self):
        samples = [{"id": "a", "image": Image.new("L", (8, 8)), "ground_truth": "hello"}]
        model = Model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = evaluate(samples, model, PreprocessingPipeline(), path)
            self.assertEqual(result["cer"], 0)
            evaluate(samples, model, PreprocessingPipeline(), path)
            self.assertEqual(model.calls, 1)
            samples[0]["ground_truth"] = "different"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                evaluate(samples, model, PreprocessingPipeline(), path)

    def test_baseline_eligible_but_active_setting_retained(self):
        table = pd.DataFrame([
            {"dataset": "iam", "technique": "grayscale", "configuration": "baseline", "cer": .1, "wer": .2},
            {"dataset": "iam", "technique": "grayscale", "configuration": "grayscale", "cer": .1, "wer": .2}])
        chosen = select_settings(table, {"grayscale": {"baseline": None, "grayscale": Grayscale()}})
        self.assertEqual(chosen["grayscale"]["configuration"], "grayscale")
        self.assertEqual(chosen["grayscale"]["winner_including_baseline"], "baseline")
        self.assertFalse(chosen["grayscale"]["improves_baseline"])
