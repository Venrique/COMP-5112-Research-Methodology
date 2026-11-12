"""Controls for the deskew validation parameter grid."""
import unittest

from src.validation.deskew import configurations


class DeskewValidationTests(unittest.TestCase):
    def test_five_distinct_one_parameter_contrasts_and_baseline(self):
        configs = configurations()
        self.assertIsNone(configs.pop('baseline'))
        pairs = {(step.max_angle, step.angle_step) for step in configs.values()}
        self.assertEqual(pairs, {(5, .5), (10, .5), (15, .5), (10, .25), (10, 1)})
        for step in configs.values():
            self.assertEqual(step.to_config()['name'], 'deskew')
            self.assertTrue(step.max_angle == 10 or step.angle_step == .5)
