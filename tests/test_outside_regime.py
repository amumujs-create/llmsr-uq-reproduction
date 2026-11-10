"""Checks for the outside-regime code (docs/research/ccpp-twin-outside-regime.md §3-§6). Synthetic data only."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
import info_axis as ia  # noqa: E402
import outside_regime as orr  # noqa: E402


class DesignChecks(unittest.TestCase):
    def test_tasks_and_grid(self):
        self.assertEqual(len(orr.TASKS), 7)
        self.assertEqual(orr.KINK_TASKS, ("evap|hot", "evap+cap_mild|hot", "cap_strong|cold"))
        self.assertEqual(len(orr.NOKINK_TASKS), 4)
        self.assertEqual(orr.DELTAS, (1.0, 2.0, 4.0))
        self.assertEqual(len(orr.VALIDATION), 6)

    def test_every_window_has_the_kink_outside(self):
        for task, (_, side, k, _) in orr.TASKS.items():
            for d in orr.DELTAS:
                for w in orr.WIDTHS:
                    lo, hi = ia.window_bounds(k, side, d, w)
                    if side == "hot":
                        self.assertLess(hi, k, task)
                    else:
                        self.assertGreater(lo, k, task)

    def test_seeds_are_new_and_distinct(self):
        self.assertNotEqual(orr.sample_seed("evap|hot", 2.0, 5.0, 150, 0), orr.sample_seed("evap|hot", 2.0, 5.0, 150, 1))
        self.assertNotEqual(orr.sample_seed("evap|hot", 2.0, 5.0, 150, 0),
                            orr.sample_seed("evap+cap_mild|hot", 2.0, 5.0, 150, 0))
        from grid import _seed
        self.assertEqual(orr.noise_seed("evap", 4.3, 3), _seed("outside1|noise|evap|4.3|3"))
        self.assertNotEqual(orr.noise_seed("evap", 4.3, 3), _seed("noise|evap|realV|4.3|3"))

    def test_description_aware_default(self):
        self.assertEqual(orr.D_AWARE["base"], [])
        self.assertEqual(orr.D_AWARE["evap"], [("above", 27.0)])
        self.assertEqual(orr.D_AWARE["cap_strong"], [("below", 8.0)])
        self.assertEqual(sorted(orr.D_AWARE["evap+cap_mild"]), sorted(orr.D_REAL))
        self.assertEqual(set(orr.D_AWARE), {t[0] for t in orr.TASKS.values()})

    def test_side_placebos(self):
        hot, cold = orr.side_placebos("hot"), orr.side_placebos("cold")
        self.assertEqual(len(hot), 10)
        self.assertEqual(len(cold), 7)
        self.assertTrue(all(st[0][0] == "above" for st in hot.values()))
        self.assertTrue(all(st[0][0] == "below" for st in cold.values()))


def frame(ha_kink, ha_nokink, d_real_nokink=2.0, d_aware_kink=0.8, oracle_kink=0.6, p_side_kink=0.9):
    """Two rows per task with the given relative RMSEs; all six validation cells share the HA values."""
    rows = []
    for task, (world, side, k, kink) in orr.TASKS.items():
        for rep in range(2):
            r = dict(task=task, world=world, side=side, k=k, delta=1.0, w=5.0, n=50, sigma=4.3, rep=rep, n_test=100,
                     R0_abs=5.0, ORACLE=oracle_kink if kink else 1.0, D_real=0.8 if kink else d_real_nokink,
                     D_aware=d_aware_kink if kink else 1.0, P_side=p_side_kink if kink else 1.5)
            for c in orr.VALIDATION:
                r[f"{c}|HA"] = ha_kink if kink else ha_nokink
                r[f"rej|{c}|HA"] = False
            rows.append(r)
    return pd.DataFrame(rows)


class VerdictChecks(unittest.TestCase):
    def test_macro_weights_tasks_equally(self):
        df = frame(0.7, 1.0)
        df = pd.concat([df, df[df.task == "base|hot"]] * 3)           # more rows in one task must not change macro
        tm = orr.task_means(df)
        self.assertAlmostEqual(orr.macro(tm, orr.TASKS)["D_real"], (3 * 0.8 + 4 * 2.0) / 7)
        self.assertAlmostEqual(orr.macro(tm, orr.KINK_TASKS)["ORACLE"], 0.6)

    def test_all_hold(self):
        v = orr.verdicts(frame(0.7, 1.0))
        self.assertTrue(v["H0"]["pass"])
        self.assertTrue(all(v[h]["pass"] for h in orr.NEED))
        self.assertTrue(v["stage1"] and v["strong"])

    def test_silent_model_beats_harmful_default_but_fails_h2(self):
        v = orr.verdicts(frame(1.0, 1.0))                              # HA == R0 everywhere
        self.assertTrue(v["H1"]["pass"] and v["H3"]["pass"])
        self.assertFalse(v["H2"]["pass"])
        self.assertFalse(v["H4"]["pass"])
        self.assertFalse(v["stage1"])

    def test_loses_to_aware_default_only(self):
        v = orr.verdicts(frame(0.85, 1.0))
        self.assertTrue(v["stage1"])
        self.assertFalse(v["H5"]["pass"])
        self.assertFalse(v["strong"])

    def test_harmful_statement_fails_h1(self):
        v = orr.verdicts(frame(0.7, 1.3))
        self.assertFalse(v["H1"]["pass"])
        self.assertFalse(v["stage1"])

    def test_premise_fails_when_default_is_harmless(self):
        v = orr.verdicts(frame(0.7, 1.0, d_real_nokink=1.0))
        self.assertFalse(v["H0"]["pass"])
        self.assertFalse(v["stage1"])

    def test_missing_validation_cell_is_an_error(self):
        df = frame(0.7, 1.0).drop(columns=[f"{orr.VALIDATION[0]}|HA"])
        with self.assertRaises(RuntimeError):
            orr.verdicts(df)


if __name__ == "__main__":
    unittest.main()
