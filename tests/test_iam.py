"""Offline checks for IAM split and revision forwarding."""

import unittest
from unittest.mock import patch

from src.load_dataset.iam import load_iam_dataset


class IAMLoaderTests(unittest.TestCase):
    @patch("src.load_dataset.iam.load_dataset")
    def test_existing_calls_preserve_defaults(self, load_dataset):
        result = load_iam_dataset()
        load_dataset.assert_called_once_with(
            "Teklia/IAM-line", split="test", revision=None
        )
        self.assertIs(result, load_dataset.return_value)

    @patch("src.load_dataset.iam.load_dataset")
    def test_revision_and_split_are_forwarded(self, load_dataset):
        revision = "fbdad97500ce54635c0d1ba306bf535cb40656cf"
        result = load_iam_dataset(split="validation", revision=revision)
        load_dataset.assert_called_once_with(
            "Teklia/IAM-line", split="validation", revision=revision
        )
        self.assertIs(result, load_dataset.return_value)
