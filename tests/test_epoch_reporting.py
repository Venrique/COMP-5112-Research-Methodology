"""Reports must use real stored predictions, preserve cohorts, and mark missing weights."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.evaluation.recognition import corpus_metrics
from src.training.epoch_reporting import generate_report


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EpochReportingTests(unittest.TestCase):
    def fixture(self, root):
        metadata = {"signature": "fixture-signature", "identity": {
            "train_indices": [0, 1], "validation_indices": [6, 25],
            "training": {"seed": 5112}, "provenance": {"dataset": "iam", "revision": "fixture"}}}
        (root / "metadata.json").write_text(json.dumps(metadata))
        history = []
        for epoch, guesses in [(1, ["c", ""]), (2, ["cat", "dog"])]:
            rows = [{"sample_id": "sample_6", "source_index": 6, "ground_truth": "cat", "prediction": guesses[0]},
                    {"sample_id": "sample_25", "source_index": 25, "ground_truth": "dog", "prediction": guesses[1]}]
            scores = corpus_metrics(["cat", "dog"], guesses)
            write_csv(root / f"validation_epoch_{epoch:03d}.csv", rows)
            history.append({"epoch": epoch, "train_loss": 3 - epoch, "validation_loss": 4 - epoch,
                            "validation_cer": scores["cer"], "validation_wer": scores["wer"],
                            "train_count": 2, "validation_count": 2, "training_seconds": 60,
                            "validation_seconds": 1, "optimizer_steps": epoch})
        write_csv(root / "history.csv", history)
        torch.save({"completed_epochs": 2, "optimizer_steps": 2, "signature": "fixture-signature"}, root / "last.pt")
        return history

    def test_report_percentage_changes_fixed_examples_and_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = generate_report(root, planned_epochs=20)
            text = path.read_text(encoding="utf-8")
            self.assertIn("83.33", text)
            self.assertIn("-83.33", text)
            self.assertIn("Best completed epoch: 2", text)
            self.assertIn("No separate weights archive", text)
            self.assertTrue((root / "epoch_checkpoints/epoch_002.pt").exists())
            self.assertFalse((root / "epoch_checkpoints/epoch_001.pt").exists())
            before = (root / "epoch_checkpoints/epoch_002.pt").read_bytes()
            generate_report(root)
            self.assertEqual(before, (root / "epoch_checkpoints/epoch_002.pt").read_bytes())
            with (root / "epoch_summary.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["exact_text_lines"], "2")
            self.assertEqual(rows[0]["empty_predictions"], "1")

    def test_rejects_inconsistent_scores_and_changed_validation_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.fixture(root)
            history[1]["validation_cer"] = 0.5
            write_csv(root / "history.csv", history)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                generate_report(root)
            self.fixture(root)
            path = root / "validation_epoch_002.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["source_index"] = "7"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "cohort"):
                generate_report(root)

    def test_no_history_has_no_fabricated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(generate_report(root))
            self.assertFalse((root / "EPOCH_RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
