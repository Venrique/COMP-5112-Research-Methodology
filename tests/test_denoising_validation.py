"""Offline controls for the denoising parameter study."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.validation.denoising import adapt_image, configurations, summarize


class ValidationTests(unittest.TestCase):
    def test_five_valid_configurations_and_baseline(self):
        configs = configurations()
        self.assertEqual(len(configs), 6)
        self.assertIsNone(configs.pop('baseline'))
        self.assertEqual(len({tuple(sorted(c.to_config().items())) for c in configs.values()}), 5)

    def test_alpha_composite_preserves_source(self):
        original = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
        adapted = adapt_image(original)
        self.assertEqual(adapted.mode, 'RGB')
        self.assertEqual(adapted.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(original.getpixel((0, 0)), (0, 0, 0, 0))

    def test_selection_equal_dataset_weight_and_baseline_eligibility(self):
        rows = [dict(dataset=d, configuration=c, cer=cer, wer=cer, sample_count=n)
                for d, c, cer, n in [('iam', 'baseline', .1, 1), ('bentham', 'baseline', .3, 100),
                                     ('iam', 'nlm', .3, 1), ('bentham', 'nlm', .2, 100)]]
        with tempfile.TemporaryDirectory() as directory:
            table, ranking, winners = summarize(rows, Path(directory))
        self.assertEqual(ranking.index[0], 'baseline')
        self.assertAlmostEqual(ranking.loc['baseline', 'cer'], .2)
        self.assertEqual(len(winners), 2)
        self.assertTrue((table[table.configuration == 'baseline'].cer_delta_vs_baseline == 0).all())
