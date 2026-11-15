"""Tiny CPU runs verify real updates, checkpoint selection, and exact resume."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from src.models.cnn_lstm_ctc import CNNLSTMCTC, CNNLSTMCTCConfig, CNNLSTMCTCRecognizer
from src.training.ctc_data import CharacterVocabulary, CTCImagePreparation, CTCLineDataset
from src.training.ctc_training import TrainingConfig, fit


class TinySplit:
    def __init__(self, split):
        self.split = split
        self._fingerprint = "fixture-" + split
        self.rows = [{"image": Image.new("L", (48, 16), 80), "text": "ab"},
                     {"image": Image.new("L", (40, 16), 180), "text": "ba"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CTCTrainingTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        self.old_deterministic = torch.are_deterministic_algorithms_enabled()
        self.vocabulary = CharacterVocabulary(("a", "b"))
        preparation = CTCImagePreparation(16)
        self.train = CTCLineDataset(TinySplit("train"), self.vocabulary, preparation)
        self.validation = CTCLineDataset(TinySplit("validation"), self.vocabulary, preparation)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        torch.use_deterministic_algorithms(self.old_deterministic)

    def model(self):
        torch.manual_seed(11)
        return CNNLSTMCTC(CNNLSTMCTCConfig(3, input_height=16, cnn_channels=(4, 4, 8),
                                          hidden_size=8, lstm_layers=2, dropout=0.1))

    def config(self, epochs=1):
        return TrainingConfig(epochs=epochs, batch_size=2, cpu_threads=1)

    def test_real_updates_checkpoint_reload_and_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            model = self.model()
            before = model.classifier.weight.detach().clone()
            result = fit(model, self.train, self.validation, temp, self.config())
            self.assertEqual(result["status"], "complete")
            self.assertFalse(torch.equal(before, model.classifier.weight))
            root = Path(temp)
            self.assertTrue((root / "training_curves.png").is_file())
            recognizer = CNNLSTMCTCRecognizer.from_checkpoint(root / "best.pt")
            torch.testing.assert_close(recognizer.model.classifier.weight, model.classifier.weight)
            self.assertIsInstance(recognizer.predict(self.train.dataset[0]["image"]), str)
            self.assertEqual(recognizer.to_config()["checkpoint_epoch"], 1)
            self.assertEqual(len(recognizer.to_config()["checkpoint_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "resume"):
                fit(self.model(), self.train, self.validation, temp, self.config())

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            continuous = self.model()
            fit(continuous, self.train, self.validation, root / "continuous", self.config(2))
            fit(self.model(), self.train, self.validation, root / "resumed", self.config(1))
            resumed = self.model()
            fit(resumed, self.train, self.validation, root / "resumed", self.config(2), resume=True)
            for key, value in continuous.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
            saved = torch.load(root / "resumed/last.pt", weights_only=True)
            scores = [(r["validation_cer"], r["validation_wer"], r["validation_loss"])
                      for r in saved["history"]]
            self.assertEqual(tuple(saved["best_score"]), min(scores))

    def test_no_test_selection_and_no_silent_invalid_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            invalid_validation = CTCLineDataset(TinySplit("test"), self.vocabulary, CTCImagePreparation(16))
            with self.assertRaisesRegex(ValueError, "never test"):
                fit(self.model(), self.train, invalid_validation, temp, self.config())
            self.train.dataset.rows[0]["image"] = Image.new("L", (4, 16), 255)
            with self.assertRaisesRegex(ValueError, "impossible"):
                fit(self.model(), self.train, self.validation, temp, self.config())
            self.assertFalse((Path(temp) / "last.pt").exists())
            status = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(status["invalid_count"], 1)

    def test_changed_resume_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fit(self.model(), self.train, self.validation, temp, self.config())
            different = TrainingConfig(epochs=2, batch_size=1, cpu_threads=1)
            with self.assertRaisesRegex(ValueError, "identity"):
                fit(self.model(), self.train, self.validation, temp, different, resume=True)

    def test_early_stopping_and_initial_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainingConfig(epochs=4, batch_size=2, cpu_threads=1,
                                    learning_rate=1e-30, patience=1)
            # Learning rate is too small to change float32 weights, so validation ties.
            result = fit(self.model(), self.train, self.validation, temp, config)
            self.assertEqual(result["status"], "early_stopped")
            self.assertEqual(result["completed_epochs"], 2)
            best = torch.load(Path(temp) / "best.pt", weights_only=True)
            self.assertEqual(best["completed_epochs"], 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Create STOP after initialization/audits, immediately before the batch loop.
            from unittest.mock import patch
            from src.training.ctc_training import audit_alignment
            def audit_and_stop(dataset):
                rows = audit_alignment(dataset)
                if str(dataset.dataset.split) == "validation":
                    (root / "STOP").touch()
                return rows
            with patch("src.training.ctc_training.audit_alignment", side_effect=audit_and_stop):
                with self.assertRaises(KeyboardInterrupt):
                    fit(self.model(), self.train, self.validation, root, self.config())
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "interrupted")
            with self.assertRaisesRegex(ValueError, "actual optimizer"):
                CNNLSTMCTCRecognizer.from_checkpoint(root / "last.pt")


if __name__ == "__main__":
    unittest.main()
