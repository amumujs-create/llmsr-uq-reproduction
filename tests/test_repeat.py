"""Prevent changed data and corrupted saved intervals from passing replay checks."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'upstream'))
from repeat_search import require_same_data, replay_intervals


class RepeatChecks(unittest.TestCase):
    def fixture(self):
        out = {'train_X': np.arange(10).reshape(-1, 1), 'train_y': np.arange(10) / 10}
        for part in ('cal', 'id', 'ood'):
            out.update({part + '_X': np.arange(10).reshape(-1, 1),
                        part + '_y': np.arange(10) / 10,
                        part + '_mean': np.zeros(10),
                        part + '_var_struct': np.ones(10), part + '_var_noise': np.zeros(10)})
            for method in ('split', 'structural'):
                out[f'{part}_{method}_lo'] = np.full(10, -0.9)
                out[f'{part}_{method}_hi'] = np.full(10, 0.9)
        return out

    def test_changed_calibration_data_rejected(self):
        first, second = self.fixture(), self.fixture()
        self.assertEqual(len(require_same_data([first, second])), 8)
        second['cal_y'][0] = 99
        with self.assertRaisesRegex(ValueError, 'cal_y'):
            require_same_data([first, second])

    def test_saved_intervals_replayed_without_model(self):
        result = replay_intervals(self.fixture())
        self.assertTrue(result['passed'])
        self.assertEqual(result['max_absolute_difference'], 0)

    def test_wrong_or_nonfinite_saved_interval_rejected(self):
        for value in (9.0, float('nan')):
            data = self.fixture()
            data['ood_structural_hi'][0] = value
            with self.assertRaises((AssertionError, ValueError)):
                replay_intervals(data)


if __name__ == '__main__':
    unittest.main()
