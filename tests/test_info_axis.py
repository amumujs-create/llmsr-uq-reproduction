"""Checks for the twin information-axis code (docs/research/ccpp-twin-information-axis.md §3-§4). Synthetic data only."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
import info_axis as ia  # noqa: E402


def kinked(n=300, k=24.0, s_cold=-1.5, s_warm=0.0, lo=18.0, hi=30.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    at = rng.uniform(lo, hi, n)
    Z = rng.normal(0, 1, (n, 3))
    y = 450 + np.where(at < k, s_cold * (at - k), s_warm * (at - k)) + Z @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, noise, n)
    return at, Z, y


class WindowChecks(unittest.TestCase):
    def test_delta_sign_hot_side(self):
        lo, hi = ia.window_bounds(24.0, "hot", 2.0, 5.0)
        self.assertEqual((lo, hi), (17.0, 22.0))      # delta > 0: kink beyond the hot edge
        lo, hi = ia.window_bounds(24.0, "hot", -2.0, 5.0)
        self.assertEqual((lo, hi), (21.0, 26.0))      # delta < 0: kink inside the window
        self.assertTrue(lo < 24.0 < hi)

    def test_delta_sign_cold_side(self):
        lo, hi = ia.window_bounds(9.0, "cold", 2.0, 5.0)
        self.assertEqual((lo, hi), (11.0, 16.0))      # kink below the cold edge
        lo, hi = ia.window_bounds(9.0, "cold", -2.0, 10.0)
        self.assertEqual((lo, hi), (7.0, 17.0))

    def test_test_band_is_beyond_window_and_kink(self):
        for delta in (-6.0, 0.0, 8.0):
            lo, hi = ia.window_bounds(24.0, "hot", delta, 5.0)
            a, b = ia.test_band(24.0, "hot", lo, hi)
            self.assertAlmostEqual(b - a, 2.5)
            self.assertAlmostEqual(a, max(hi, 24.0) + 2.5)
            lo, hi = ia.window_bounds(9.0, "cold", delta, 5.0)
            a, b = ia.test_band(9.0, "cold", lo, hi)
            self.assertAlmostEqual(b - a, 2.5)
            self.assertAlmostEqual(b, min(lo, 9.0) - 2.5)

    def test_test_mask_excludes_window(self):
        at = np.linspace(0, 40, 4001)
        lo, hi = ia.window_bounds(24.0, "hot", -2.0, 5.0)
        m = ia.test_mask(at, 24.0, "hot", lo, hi)
        self.assertTrue(at[m].min() >= hi + 2.5 and at[m].max() < hi + 5.0)

    def test_specs_and_validation_cells(self):
        specs = ia.window_specs()
        self.assertEqual(sum(s["module"] == "evap" for s in specs), 7)
        self.assertEqual(sum(s["module"] == "cap_strong" for s in specs), 5)
        self.assertEqual(sum(s["module"] == "base" for s in specs), 12)
        self.assertEqual(len(ia.VALIDATION), 6)


class InformationChecks(unittest.TestCase):
    def test_kink_inside_window_gives_positive_I(self):
        at, Z, y = kinked()
        I, knot, b = ia.changepoint(at, Z, y)
        self.assertGreater(I, 0)
        self.assertAlmostEqual(knot, 24.0, delta=1.0)
        self.assertAlmostEqual(b[1], -1.5, delta=0.2)
        self.assertAlmostEqual(b[2], 0.0, delta=0.2)

    def test_linear_window_gives_nonpositive_I(self):
        at, Z, y = kinked(s_cold=-1.5, s_warm=-1.5)
        I, _, _ = ia.changepoint(at, Z, y)
        self.assertLessEqual(I, 0)

    def test_predict_cp_uses_extrapolation_side_slope(self):
        at, Z, y = kinked()
        I, knot, b = ia.changepoint(at, Z, y)
        at_test = np.array([33.0, 34.0])
        Zt = np.zeros((2, 3))
        p = ia.predict_cp(I, knot, b, "hot", at_test, Zt, np.zeros(2))
        self.assertAlmostEqual(p[1] - p[0], b[2], places=9)       # warm slope on the hot side
        p = ia.predict_cp(I, knot, b, "cold", np.array([15.0, 16.0]), Zt, np.zeros(2))
        self.assertAlmostEqual(p[1] - p[0], b[1], places=9)       # cold slope on the cold side
        p0 = np.array([1.0, 2.0])
        self.assertIs(ia.predict_cp(-1.0, knot, b, "hot", at_test, Zt, p0), p0)   # I <= 0 -> R0

    def test_switch(self):
        a, b = np.array([1.0]), np.array([2.0])
        self.assertIs(ia.switch(0.5, a, b), a)
        self.assertIs(ia.switch(0.0, a, b), b)
        self.assertIs(ia.switch(-3.0, a, b), b)


class StatementChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 150
        at = rng.uniform(14.0, 19.0, n)
        v, ap, rh = rng.uniform(40, 70, n), rng.uniform(1000, 1030, n), rng.uniform(30, 90, n)
        self.X = np.column_stack([at, v, ap, rh])
        self.y = 480 - 1.6 * at + 0.1 * v + rng.normal(0, 1.0, n)
        self.b, self.se = ia.ols(self.X, self.y)
        self.slope_phys = -self.y.mean() / (at.mean() + 273.15)
        self.edge, self.center = at.max(), 0.5 * (at.min() + at.max())
        self.at_test = np.linspace(self.edge + 2.5, self.edge + 5.0, 40)
        Xt = np.column_stack([self.at_test, np.full(40, 55.0), np.full(40, 1015.0), np.full(40, 60.0)])
        self.p0 = np.column_stack([np.ones(40), Xt]) @ self.b
        self.rh_test = Xt[:, 3]

    def test_empty_statement_is_r0_not_rejected(self):
        p, rej = ia.predict_statement([], self.p0, self.b, self.se, self.slope_phys, self.edge, self.center,
                                      self.at_test, self.rh_test)
        self.assertIs(p, self.p0)
        self.assertFalse(rej)

    def test_flat_above_edge_is_between_r0_and_flat(self):
        p, rej = ia.predict_statement([("above", self.edge)], self.p0, self.b, self.se, self.slope_phys, self.edge,
                                      self.center, self.at_test, self.rh_test)
        self.assertFalse(rej)
        y_edge = self.p0 - self.b[1] * (self.at_test - self.edge)
        half = 0.5 * (self.p0 + y_edge)                      # blend 0.5 of R0 and "flat from the edge"
        np.testing.assert_allclose(p, half, atol=1e-6)


class IntervalChecks(unittest.TestCase):
    def test_no_statement_interval_covers_monotone_lipschitz_truth(self):
        at = np.linspace(22.5, 25.0, 50)
        edge, L = 20.0, 1.6
        y_edge = np.full(50, 440.0)
        truth = 440.0 - 0.9 * (at - edge)                    # slope magnitude 0.9 < L, non-increasing
        lo, hi = ia.interval([], "hot", edge, at, y_edge, L)
        self.assertTrue(np.all((truth >= lo) & (truth <= hi)))
        np.testing.assert_allclose(hi, y_edge)
        np.testing.assert_allclose(hi - lo, L * (at - edge))
        lo, hi = ia.interval([], "cold", 10.0, np.array([5.0]), np.array([440.0]), L)
        self.assertAlmostEqual(lo[0], 440.0)
        self.assertAlmostEqual(hi[0], 440.0 + 5 * L)

    def test_statement_narrows_interval_only_on_its_side(self):
        at = np.array([23.0, 24.0, 25.0])
        y_edge, edge, L = np.full(3, 440.0), 20.0, 1.6
        lo_n, hi_n = ia.interval([], "hot", edge, at, y_edge, L)
        lo_s, hi_s = ia.interval([("above", 22.0)], "hot", edge, at, y_edge, L)
        np.testing.assert_allclose(hi_s - lo_s, np.full(3, 2.0 * L))    # distance capped at x_c - edge
        self.assertTrue(np.all(hi_s - lo_s < hi_n - lo_n))
        lo_i, hi_i = ia.interval([("above", 18.0)], "hot", edge, at, y_edge, L)
        np.testing.assert_allclose(hi_i - lo_i, 0.0)                     # x_c inside the window: width 0
        lo_o, hi_o = ia.interval([("below", 9.0)], "hot", edge, at, y_edge, L)
        np.testing.assert_allclose(hi_o - lo_o, hi_n - lo_n)             # other side: unchanged

    def test_iv_column_names_keep_cell_pipes(self):
        import pandas as pd
        df = pd.DataFrame(columns=["IV|none|cov", "IV|gemma3-27b|L1|cov", "IV|gemma3-27b|L1|width", "R0"])
        self.assertEqual(ia.iv_statement_names(df), ["gemma3-27b|L1", "none"])

    def test_interval_stats(self):
        st = ia.interval_stats(np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([1.0, 3.0]), 1.0)
        self.assertAlmostEqual(st["cov"], 0.5)
        self.assertAlmostEqual(st["width"], 2.0)
        self.assertAlmostEqual(st["mid"], np.sqrt(0.5 * (0 + 4)))


if __name__ == "__main__":
    unittest.main()
