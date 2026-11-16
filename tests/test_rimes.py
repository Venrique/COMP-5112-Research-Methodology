"""Offline contract tests; never download the corpus during unit tests."""
import unittest
from unittest.mock import patch

from src.load_dataset.rimes import DATASET_ID, load_rimes_dataset


class RIMESLoaderTests(unittest.TestCase):
    @patch("src.load_dataset.rimes.load_dataset")
    def test_development_default(self, loader):
        self.assertIs(load_rimes_dataset(), loader.return_value)
        loader.assert_called_once_with(DATASET_ID, split="validation", revision=None)

    @patch("src.load_dataset.rimes.load_dataset")
    def test_preserves_split_and_revision(self, loader):
        for split in ("train", "validation", "test"):
            loader.reset_mock()
            load_rimes_dataset(split, revision="fixed-commit")
            loader.assert_called_once_with(DATASET_ID, split=split, revision="fixed-commit")

    @patch("src.load_dataset.rimes.load_dataset")
    def test_rejects_resplitting_and_invalid_names(self, loader):
        for split in ("test[:10]", "dev", "train+test"):
            with self.assertRaises(ValueError):
                load_rimes_dataset(split)
        loader.assert_not_called()
