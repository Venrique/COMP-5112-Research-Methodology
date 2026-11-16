"""Small offline architecture, CTC alignment, and decoding checks."""
import unittest

import torch
from PIL import Image

from src.models.cnn_lstm_ctc import (
    CNNLSTMCTC, CNNLSTMCTCConfig, CNNLSTMCTCRecognizer,
    ctc_batch_loss, greedy_ctc_decode,
)
from src.training.ctc_data import CharacterVocabulary


class CNNLSTMCTCTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(5112)
        return CNNLSTMCTC(CNNLSTMCTCConfig(
            num_classes=3, input_height=16, cnn_channels=(4, 8, 8),
            hidden_size=8, lstm_layers=1, dropout=0,
        ))

    def test_shapes_lengths_and_real_backward_pass(self):
        model = self.make_model()
        images = torch.rand(2, 1, 16, 47)
        scores, lengths = model(images, torch.tensor([47, 32]))
        self.assertEqual(tuple(scores.shape), (11, 2, 3))
        self.assertEqual(lengths.tolist(), [11, 8])
        torch.testing.assert_close(scores.exp().sum(2), torch.ones(11, 2))
        batch = {"targets": torch.tensor([1, 1, 2, 2, 1]),
                 "target_lengths": torch.tensor([3, 2])}
        loss = ctc_batch_loss(scores, lengths, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.cnn[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.lstm.weight_ih_l0.grad.abs().sum()), 0)

    def test_same_prediction_features_alone_and_in_padded_batch(self):
        model = self.make_model().eval()
        image = torch.rand(1, 1, 16, 29)
        other = torch.rand(1, 1, 16, 48)
        padded = torch.ones(2, 1, 16, 48)
        padded[0, :, :, :29] = image[0]
        padded[1] = other[0]
        with torch.no_grad():
            alone, _ = model(image, torch.tensor([29]))
            together, lengths = model(padded, torch.tensor([29, 48]))
        torch.testing.assert_close(alone[:, 0], together[:int(lengths[0]), 0], atol=1e-6, rtol=1e-5)

    def test_repeated_letters_blanks_empty_output_and_padding(self):
        vocabulary = CharacterVocabulary(("l", "a"))
        # First path: ll-blank-ll -> ll. Second: all blanks -> empty text.
        paths = torch.tensor([[1, 0], [1, 0], [0, 0], [1, 0], [1, 0], [2, 2]])
        scores = torch.full((6, 2, 3), -10.0)
        scores.scatter_(2, paths.unsqueeze(2), 0.0)
        self.assertEqual(greedy_ctc_decode(scores, torch.tensor([5, 5]), vocabulary), ["ll", ""])

    def test_ctc_rejects_impossible_alignments_and_blank_targets(self):
        scores = torch.randn(2, 1, 3).log_softmax(2)
        batch = {"targets": torch.tensor([1, 1]), "target_lengths": torch.tensor([2]),
                 "sample_ids": ["repeated-letter-line"]}
        with self.assertRaisesRegex(ValueError, "need 3 time steps"):
            ctc_batch_loss(scores, torch.tensor([2]), batch)
        batch["targets"] = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "exclude.*blank"):
            ctc_batch_loss(scores, torch.tensor([2]), batch)
        batch["targets"] = torch.tensor([1, 2, 1])
        with self.assertRaisesRegex(ValueError, "sum"):
            ctc_batch_loss(scores, torch.tensor([2]), batch)

    def test_input_validation_and_untrained_recognition_guard(self):
        model = self.make_model()
        for widths in (torch.tensor([3]), torch.tensor([100]), torch.tensor([8.5])):
            with self.assertRaises(ValueError):
                model(torch.rand(1, 1, 16, 20), widths)
        with self.assertRaises(ValueError):
            CNNLSTMCTCConfig(num_classes=1)
        recognizer = CNNLSTMCTCRecognizer(model, CharacterVocabulary(("a", "b")))
        with self.assertRaisesRegex(RuntimeError, "Train"):
            recognizer.predict(Image.new("L", (20, 16), 255))
        self.assertEqual(recognizer.to_config()["weights_status"], "random initialization")

    def test_recognizer_interface_and_image_preservation(self):
        # Controlled artificial weights test the adapter, not handwriting accuracy.
        model = self.make_model().train()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
        recognizer = CNNLSTMCTCRecognizer(model, CharacterVocabulary(("a", "b")), trained=True)
        image = Image.new("RGBA", (40, 16), (20, 20, 20, 128))
        before = image.tobytes()
        self.assertEqual(recognizer.predict(image), "a")
        self.assertEqual(image.tobytes(), before)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
