"""Checks for the BDG2 chilled-water extrapolation code (docs/research/bdg2-regime-extrapolation.md)."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "bdg2"))
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
import collect_bdg2  # noqa: E402
import extrap  # noqa: E402
import score  # noqa: E402
from check_data import fit_5p  # noqa: E402
from parse_score import majority, parse, to_statement  # noqa: E402


def piecewise(t, tc, slope, rho=0.0, base=100.0):
    """Chilled-water-like daily load: flat (or weaker, slope*rho) below tc, slope above."""
    return base + np.where(t > tc, slope * (t - tc), rho * slope * (t - tc))


def synthetic(tc=14.0, slope=10.0, rho=0.0, noise=1.0, n=720, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-5, 35, n)
    y = piecewise(t, tc, slope, rho) + rng.normal(0, noise, n)
    return t, y


class FitChecks(unittest.TestCase):
    def test_fit_5p_recovers_cooling_balance_point(self):
        t, y = synthetic(tc=14.0)
        th, tc, b, cv, r2 = fit_5p(t, y)
        self.assertAlmostEqual(tc, 14.0, delta=1.5)
        self.assertGreater(r2, 0.9)
        self.assertAlmostEqual(b[2], 10.0, delta=0.5)


class WindowChecks(unittest.TestCase):
    def test_kink_outside_window_gives_no_information_and_oracle_gain(self):
        t, y = synthetic(tc=10.0, noise=0.5)
        cells = extrap.rows_for_world(t, y, tc=10.0)
        self.assertTrue(cells)
        outside = [c for c in cells if 0 < c["delta"] <= c["d"]]   # kink outside, test segment reaches below it
        self.assertTrue(outside)
        for c in outside:
            self.assertLessEqual(c["I"], 2.0 * np.log(c["n_win"]))   # no strong in-window evidence
            y_te = c["y_test"]
            r0 = extrap.rmse(extrap.predict_statement(c, None, 0.0), y_te)
            rcp = extrap.rmse(extrap.predict_cp(c), y_te)
            oracle = extrap.rmse(extrap.predict_statement(c, 10.0, 0.0), y_te)
            self.assertLess(oracle, r0)                                # the true statement helps
            self.assertGreaterEqual(rcp, r0 - 1e-9)                    # in-window change-point gives no gain

    def test_kink_inside_window_gives_information_and_rcp_gain(self):
        t, y = synthetic(tc=22.0, noise=0.5)
        cells = [c for c in extrap.rows_for_world(t, y, tc=22.0) if c["delta"] < -3 and c["e"] > 25.0 and c["w"] == 10.0]
        self.assertTrue(cells)
        for c in cells:
            self.assertGreater(c["I"], 0.0)
            self.assertAlmostEqual(c["knot"], 22.0, delta=2.0)
            y_te = c["y_test"]
            self.assertLess(extrap.rmse(extrap.predict_cp(c), y_te), extrap.rmse(extrap.predict_statement(c, None, 0.0), y_te))

    def test_guard_rejects_flat_statement_covering_a_sloped_window(self):
        t, y = synthetic(tc=5.0, noise=0.2)
        c = [c for c in extrap.rows_for_world(t, y, tc=5.0) if c["w"] == 10.0][0]
        p0 = extrap.predict_statement(c, None, 0.0)
        # "flat below 40" says the whole (clearly sloped) window is flat -> guard rejects -> R0 unchanged
        np.testing.assert_allclose(extrap.predict_statement(c, 40.0, 0.0), p0)
        # a boundary below the window passes the guard and changes the prediction
        self.assertGreater(np.abs(extrap.predict_statement(c, c["lo"] - 1.0, 0.0) - p0).max(), 0.0)

    def test_factor_half_sits_between_r0_and_flat(self):
        t, y = synthetic(tc=10.0, noise=0.5)
        c = [c for c in extrap.rows_for_world(t, y, tc=10.0) if 0 < c["delta"] <= c["d"]][0]
        p0 = extrap.predict_statement(c, None, 0.0)
        pf = extrap.predict_statement(c, 10.0, 0.0)
        ph = extrap.predict_statement(c, 10.0, 0.5)
        np.testing.assert_allclose(ph, 0.5 * (p0 + pf), rtol=1e-9)

    def test_run_table_and_fixed_statements(self):
        worlds = pd.DataFrame([dict(meter="A_x_1", site="A", tc=10.0, rho=0.0), dict(meter="A_x_2", site="A", tc=12.0, rho=0.1)])
        data = {"A_x_1": synthetic(tc=10.0, seed=1), "A_x_2": synthetic(tc=12.0, seed=2)}
        llm = {"m|L1|HA": {"A_x_1": (9.0, 0.0), "A_x_2": (None, 0.0)}}
        df = extrap.run(worlds, lambda m: data[m], llm)
        self.assertEqual(set(df.meter), {"A_x_1", "A_x_2"})
        for col in ("R0", "R_cp", "ORACLE", "ORACLE_half", "D", "D_site", "P|4", "P|26", "m|L1|HA"):
            self.assertIn(col, df)
        # D_site for A_x_1 is the other building's tc (12.0); an empty LLM statement equals R0
        sub = df[df.meter == "A_x_2"]
        np.testing.assert_allclose(sub["m|L1|HA"], sub.R0)
        self.assertEqual(len(df.columns[df.columns.str.startswith("P|")]), len(extrap.PLACEBO))


class CollectChecks(unittest.TestCase):
    FIELDS = {"use": "Office", "climate": {"mean": 21.6, "p5": 8.9, "p95": 31.0}, "sub_use": "Office", "sqm_bin": "5,000 to 10,000"}

    def test_descriptions_do_not_leak_shape_words_or_ids(self):
        for level in ("L1", "L2"):
            text = collect_bdg2.describe(self.FIELDS, level)
            self.assertEqual(collect_bdg2.check_no_leak(text), [], level)
            for site in ("Panther", "Fox", "Bull", "Eagle", "Hog", "Cockatoo", "Kaggle", "Genome"):
                self.assertNotIn(site, text)
        self.assertEqual(collect_bdg2.check_no_leak(collect_bdg2.describe_default()), [])
        self.assertNotIn("Primary use", collect_bdg2.describe_default())
        self.assertIn("square metres", collect_bdg2.describe(self.FIELDS, "L2"))
        self.assertNotIn("square metres", collect_bdg2.describe(self.FIELDS, "L1"))

    def test_jobs_and_mock_parse(self):
        desc = {"L1": {"L1-000": {"fields": self.FIELDS, "meters": ["A_x_1"]}},
                "L2": {"L2-000": {"fields": self.FIELDS, "meters": ["A_x_1"]}}}
        jobs = collect_bdg2.jobs(desc)
        self.assertEqual([j["level"] for j in jobs], ["M", "L1", "L2"])
        for j in jobs:
            text, _ = collect_bdg2.call_mock(j["messages"], 0.7, 3)
            p = parse(text)
            self.assertTrue(p["ok"])
        self.assertEqual(to_statement(parse(collect_bdg2.call_mock(jobs[0]["messages"], 0.0, 0)[0])["boundaries"]), [])
        stmt = to_statement(parse(collect_bdg2.call_mock(jobs[1]["messages"], 0.0, 0)[0])["boundaries"])
        self.assertEqual(stmt[0][0], "below")


class ScoreChecks(unittest.TestCase):
    def test_llm_statements_majority_to_cells(self):
        desc = {"L1": {"L1-000": {"fields": {}, "meters": ["A_x_1", "A_x_2"]}}, "L2": {"L2-000": {"fields": {}, "meters": ["A_x_1"]}}}
        answers = {"m": {("L1", "L1-000"): [[("below", 10.0)], [("below", 12.0)], []],
                         ("L2", "L2-000"): [[], [], [("below", 9.0)]],
                         ("M", "M"): [[("below", 18.0)]] * 3}}
        cells = score.llm_statements(desc, answers)
        self.assertEqual(cells["m|L1|HA"]["A_x_1"], (11.0, 0.0))        # median of the two 'below' answers
        self.assertEqual(cells["m|L1|HA_half"]["A_x_2"], (11.0, 0.5))
        self.assertEqual(cells["m|L2|HA"]["A_x_1"], (None, 0.0))        # 'below' named by 1 of 3 -> empty
        self.assertEqual(cells["m|M"]["A_x_2"], (18.0, 0.0))

    def test_bins_and_summary(self):
        df = pd.DataFrame({"I": [-1, -1, -1, -2, -2, -2, 1, 2, 3.0], "delta": [-1, -1, -1, 2, 2, 2, 0, 0, 0.0],
                           "R0": np.ones(9), "ORACLE": np.full(9, 0.8), "X": np.array([0.9] * 3 + [1.0] * 3 + [1.2] * 3)})
        d = score.add_bins(df)
        self.assertEqual(list(d.I_bin), ["inside_hidden"] * 3 + ["outside"] * 3 + ["evidence"] * 3)
        self.assertEqual(score.bin_counts(d), {"evidence": 3, "outside": 3, "inside_hidden": 3})
        s = score.summarize(d, ["X"])
        self.assertAlmostEqual(s.loc["X", "rel_inside_hidden"], 0.9)
        self.assertAlmostEqual(s.loc["X", "rel_outside"], 1.0)
        self.assertAlmostEqual(s.loc["X", "harm"], 1 / 3)
        self.assertAlmostEqual(s.loc["X", "filled"], (0.1 * 3 + 0 - 0.2 * 3) / 9 / 0.2)

    def test_verdicts_known_tables(self):
        def tab(**cols):
            return pd.DataFrame(cols).T
        fixed = tab(D=dict(rel_mean=0.97, filled=0.3, rel_outside=0.95, rel_evidence=1.0),
                    R_cp=dict(rel_mean=0.99, rel_outside=1.0, rel_evidence=0.9),
                    **{"P|4": dict(rel_mean=1.02), "P|6": dict(rel_mean=1.0)})
        by_cell = {"fixed": fixed}
        for tag in score.VALIDATION_TAGS:
            by_cell[f"{tag}|M"] = tab(M=dict(rel_mean=1.0))
            for lv in score.LEVELS:
                by_cell[f"{tag}|{lv}|HA"] = tab(HA=dict(rel_mean=0.95, harm=0.01, rel_outside=0.9, rel_evidence=0.99))
        struct = pd.DataFrame([dict(tag=t, level=lv, abs_err=e) for t in score.VALIDATION_TAGS
                               for lv, e in (("L1", 5.0), ("L2", 3.0))])
        v = score.verdicts(None, by_cell, struct)
        self.assertTrue(all(v[k] for k in ("B1", "B2", "B3", "B4", "B5", "B6", "B7")), v)
        # D better than HA -> B3 fails; R_cp without gain where the window shows the kink -> B6 fails
        fixed.loc["D", "rel_mean"] = 0.90
        fixed.loc["R_cp", "rel_evidence"] = 1.0
        v2 = score.verdicts(None, by_cell, struct)
        self.assertFalse(v2["B3"])
        self.assertFalse(v2["B6"])
        self.assertIn("status", score.verdicts(None, {"fixed": fixed}, struct))


if __name__ == "__main__":
    unittest.main()
