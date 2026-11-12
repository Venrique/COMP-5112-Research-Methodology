"""Controls for the CLAHE validation grid."""
import unittest
from src.validation.clahe import configurations


class CLAHEValidationTests(unittest.TestCase):
    def test_five_distinct_one_parameter_contrasts_and_baseline(self):
        configs = configurations()
        self.assertIsNone(configs.pop('baseline'))
        pairs = {(step.clip_limit, step.tile_grid_size) for step in configs.values()}
        self.assertEqual(pairs, {(1, (8, 8)), (2, (8, 8)), (4, (8, 8)),
                                 (2, (4, 4)), (2, (16, 16))})
        for step in configs.values():
            self.assertEqual(step.to_config()['name'], 'clahe')
            self.assertTrue(step.clip_limit == 2 or step.tile_grid_size == (8, 8))
