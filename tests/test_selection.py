"""Dataset-specific selection must not turn an active treatment into baseline."""
import unittest
import pandas as pd
from src.evaluation.selection import select_validation_configuration


class SelectionTests(unittest.TestCase):
    def test_dataset_selection_baseline_exclusion_and_ties(self):
        table = pd.DataFrame([
            ('iam', 'baseline', 0., 0.), ('iam', 'first', .1, .2),
            ('iam', 'second', .1, .2), ('iam', 'worse_wer', .1, .3),
            ('bentham', 'first', .4, .5), ('bentham', 'second', .2, .3),
        ], columns=['dataset', 'configuration', 'cer', 'wer'])
        self.assertEqual(select_validation_configuration(table, 'iam')['configuration'], 'first')
        self.assertEqual(select_validation_configuration(table, 'bentham')['configuration'], 'second')
        with self.assertRaises(ValueError):
            select_validation_configuration(table, 'absent')
