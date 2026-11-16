"""Offline checks for train-only character IDs and loss-compatible batches."""
import unittest

import torch
from PIL import Image

from src.training.ctc_data import (
    CharacterVocabulary, CTCImagePreparation, CTCLineDataset, collate_ctc_lines,
)


class SmallSplit:
    def __init__(self, split="train"):
        self.split = split
        self.rows = [{"image": Image.new("L", (20, 10), 0), "text": "  book  "},
                     {"image": Image.new("RGBA", (10, 10), (0, 0, 0, 0)), "text": "OK!"}]

    def __getitem__(self, index):
        return [r[index] for r in self.rows] if isinstance(index, str) else self.rows[index]

    def __len__(self):
        return len(self.rows)


class CTCDataTests(unittest.TestCase):
    def setUp(self):
        self.source = SmallSplit()
        self.vocabulary = CharacterVocabulary.from_training_split(self.source)

    def test_vocabulary_train_only_and_no_unknown_substitution(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            CharacterVocabulary.from_training_split(SmallSplit("test"))
        self.assertEqual(self.vocabulary.characters, tuple(sorted(set("bookOK!"))))
        labels = self.vocabulary.encode("  book\n")
        self.assertTrue(torch.all(labels > 0))
        self.assertEqual(self.vocabulary.decode_labels(labels), "book")
        self.assertEqual(self.vocabulary.unknown_characters(["book?", "OK!"]), ["?"])
        with self.assertRaisesRegex(ValueError, "absent"):
            self.vocabulary.encode("book?")
        with self.assertRaises(ValueError):
            self.vocabulary.encode(" ")
        with self.assertRaises(ValueError):
            self.vocabulary.decode_labels([0])

    def test_resize_alpha_and_original_preservation(self):
        original = self.source[1]["image"]
        before = original.tobytes()
        prepared = CTCImagePreparation(height=8)(original)
        self.assertEqual(tuple(prepared.shape), (1, 8, 8))
        self.assertTrue(torch.all(prepared == 1))
        self.assertEqual(original.tobytes(), before)
        self.assertEqual(original.mode, "RGBA")
        self.assertEqual(original.size, (10, 10))
        self.assertEqual(tuple(CTCImagePreparation(8)(self.source[0]["image"]).shape), (1, 8, 16))
        with self.assertRaises(ValueError):
            CTCImagePreparation(0)

    def test_padding_targets_source_ids_and_repeated_label_alignment(self):
        dataset = CTCLineDataset(self.source, self.vocabulary, CTCImagePreparation(8), indices=[1, 0])
        batch = collate_ctc_lines([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["images"].shape), (2, 1, 8, 16))
        self.assertTrue(torch.all(batch["images"][0, :, :, 8:] == 1))
        self.assertEqual(batch["image_widths"].tolist(), [8, 16])
        self.assertEqual(batch["target_lengths"].tolist(), [3, 4])
        self.assertEqual(batch["minimum_ctc_timesteps"].tolist(), [3, 5])
        self.assertEqual(batch["sample_ids"], ["sample_1", "sample_0"])
        self.assertEqual(batch["ground_truth"], ["OK!", "  book  "])
        self.assertEqual(batch["training_text"], ["OK!", "book"])
        # A real CTC loss checks concatenated targets and repeated-letter lengths.
        log_probs = torch.randn(6, 2, self.vocabulary.num_classes).log_softmax(2).requires_grad_()
        loss = torch.nn.CTCLoss(blank=0)(log_probs, batch["targets"], torch.tensor([6, 6]), batch["target_lengths"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(log_probs.grad).all())

    def test_invalid_selection_and_inconsistent_height(self):
        for indices in ([0, 0], [2], []):
            with self.assertRaises(ValueError):
                CTCLineDataset(self.source, self.vocabulary, indices=indices)
        with self.assertRaises(ValueError):
            collate_ctc_lines([])
        first = CTCLineDataset(self.source, self.vocabulary, CTCImagePreparation(8))[0]
        second = CTCLineDataset(self.source, self.vocabulary, CTCImagePreparation(16))[1]
        with self.assertRaisesRegex(ValueError, "height"):
            collate_ctc_lines([first, second])


if __name__ == "__main__":
    unittest.main()
