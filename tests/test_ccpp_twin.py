"""Physical sanity checks for the CCPP twin simulator."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'ccpp_twin'))
from ccpp_twin import TwinConfig, simulate, wet_bulb_c, with_modules
from regime_rules import compose, from_mechanisms, statements


def sweep(config, at=np.linspace(2, 37, 36)):
    n = len(at)
    return at, simulate(at, np.full(n, 1013.0), np.full(n, 70.0), np.full(n, 45.0), config)


class TwinChecks(unittest.TestCase):
    def test_output_positive_and_in_plant_range(self):
        _, out = sweep(TwinConfig())
        self.assertTrue(np.all(out['pe'] > 0))
        self.assertTrue(380 < out['pe'].mean() < 520)

    def test_output_decreases_with_ambient_temperature(self):
        _, out = sweep(TwinConfig())
        self.assertTrue(np.all(np.diff(out['pe']) < 0))

    def test_components_sum_to_net_power(self):
        cfg = TwinConfig()
        _, out = sweep(cfg)
        np.testing.assert_allclose(out['p_gt'] + out['p_st'] - cfg.aux_mw, out['pe_clean'])

    def test_worse_vacuum_lowers_output(self):
        a = simulate([15.0], [1013.0], [70.0], [40.0])['pe']
        b = simulate([15.0], [1013.0], [70.0], [70.0])['pe']
        self.assertLess(b[0], a[0])

    def test_cold_cap_flattens_cold_end_only(self):
        at, base = sweep(TwinConfig())
        _, capped = sweep(with_modules(cold_cap=True))
        cold, mid = at < 6, (at > 18) & (at < 24)
        self.assertLess(np.abs(np.diff(capped['pe'][cold])).mean(), np.abs(np.diff(base['pe'][cold])).mean())
        np.testing.assert_allclose(capped['pe'][mid], base['pe'][mid], atol=0.5)

    def test_evap_cooling_flattens_hot_end_only(self):
        at, base = sweep(TwinConfig())
        _, cooled = sweep(with_modules(evap_cooling=True))
        hot, cold = at > 30, at < 10
        self.assertLess(np.abs(np.diff(cooled['pe'][hot])).mean(), np.abs(np.diff(base['pe'][hot])).mean())
        np.testing.assert_allclose(cooled['pe'][cold], base['pe'][cold], atol=0.5)

    def test_noise_is_reproducible_by_seed(self):
        cfg = TwinConfig(noise_sd_mw=4.0)
        a = sweep(cfg)[1]['pe']
        b = simulate(np.linspace(2, 37, 36), np.full(36, 1013.0), np.full(36, 70.0), np.full(36, 45.0), cfg, seed=7)['pe']
        c = simulate(np.linspace(2, 37, 36), np.full(36, 1013.0), np.full(36, 70.0), np.full(36, 45.0), cfg, seed=7)['pe']
        np.testing.assert_array_equal(b, c)
        self.assertEqual(a.shape, b.shape)

    def test_wet_bulb_below_dry_bulb(self):
        self.assertTrue(np.all(wet_bulb_c(np.array([10.0, 25.0, 35.0]), np.array([40.0, 60.0, 30.0])) < [10.0, 25.0, 35.0]))

    # Anti-icing module (docs/research/ccpp-anti-icing-world.md §3, §9 step 1)
    def test_anti_icing_off_leaves_other_worlds_unchanged(self):
        rng = np.random.default_rng(0)
        x = [rng.uniform(lo, hi, 2000) for lo, hi in ((1, 37), (993, 1033), (25, 100), (25, 81))]
        for kw in ({}, {'evap_cooling': True}, {'cold_cap': True, 'gt_cap_mw': 360.0},
                   {'cold_cap': True, 'evap_cooling': True, 'gt_cap_mw': 360.0}):
            a = simulate(*x, with_modules(**kw))['pe_clean']
            b = simulate(*x, with_modules(anti_icing=False, **kw))['pe_clean']
            np.testing.assert_array_equal(a, b)
            self.assertFalse(simulate(*x, with_modules(**kw))['intake_heating_k'].any())

    def test_anti_icing_reverses_cold_humid_slope(self):
        at = np.linspace(2, 12, 21)
        n = len(at)
        base = simulate(at, np.full(n, 1013.0), np.full(n, 90.0), np.full(n, 40.0))['pe_clean']
        ice = simulate(at, np.full(n, 1013.0), np.full(n, 90.0), np.full(n, 40.0), with_modules(anti_icing=True))
        cold, warm = at < 6.5, at > 7.5
        self.assertTrue(np.all(np.diff(ice['pe_clean'][cold]) > 0))   # colder -> more heating -> less output
        np.testing.assert_allclose(ice['pe_clean'][warm], base[warm], atol=1e-9)
        self.assertTrue(np.all(ice['pe_clean'][cold] < base[cold]))
        np.testing.assert_allclose(ice['intake_heating_k'][at <= 7], 7 - at[at <= 7], atol=0.01)  # gate ~0.999 at RH 90

    def test_anti_icing_needs_humidity(self):
        dry = simulate([3.0], [1013.0], [40.0], [40.0], with_modules(anti_icing=True))
        base = simulate([3.0], [1013.0], [40.0], [40.0])
        self.assertLess(dry['intake_heating_k'][0], 0.01)
        self.assertAlmostEqual(dry['pe_clean'][0], base['pe_clean'][0], delta=0.05)


class RowLevelAssembly(unittest.TestCase):
    """Humidity-conditioned boundaries (ccpp-anti-icing-world.md §5, §9 step 3)."""
    at = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 2.0, 4.0])
    rh = np.array([90.0, 90.0, 90.0, 90.0, 90.0, 40.0, 40.0])

    def args(self, stmt, rh=None):
        return compose(stmt, 15.0, self.at, -2.0, -1.5, 0.0, False, rh)

    def test_unconditioned_statement_ignores_rh(self):
        st = [("below", 6.0)]
        np.testing.assert_array_equal(self.args(st), self.args(st, self.rh))

    def test_rh_condition_applies_only_to_humid_rows(self):
        got = self.args([("below", 7.0, 70.0)], self.rh)
        humid, dry = self.rh > 70, self.rh <= 70
        np.testing.assert_allclose(got[humid], self.args([("below", 7.0)])[humid])
        np.testing.assert_allclose(got[dry], self.args([])[dry])

    def test_union_with_cap(self):
        got = self.args([("below", 7.0, 70.0), ("below", 6.0)], self.rh)
        humid, dry = self.rh > 70, self.rh <= 70
        np.testing.assert_allclose(got[humid], self.args([("below", 7.0)])[humid])
        np.testing.assert_allclose(got[dry], self.args([("below", 6.0)])[dry])

    def test_condition_requires_rh(self):
        with self.assertRaises(ValueError):
            self.args([("below", 7.0, 70.0)])

    def test_from_mechanisms(self):
        got = from_mechanisms([{"mechanism": "anti_icing", "side": "below", "at_c": 7, "rh_above": 70},
                               {"mechanism": "power_cap", "at_c": 6},
                               {"mechanism": "evap_cooling", "side": "above", "at_c": 24},
                               {"mechanism": "power_cap", "at_c": None}])
        self.assertEqual(got, [("below", 7.0, 70.0), ("below", 6.0), ("above", 24.0)])

    def test_ice_truth_statements(self):
        st = statements("ice+cap_mild")
        self.assertEqual(st["correct"], [("below", 7.0, 70.0), ("below", 6.0)])
        self.assertEqual(st["miss_ice"], [("below", 6.0)])
        self.assertEqual(st["shift+3"], [("below", 10.0, 70.0), ("below", 9.0)])
        self.assertEqual(statements("cap_mild")["correct"], [("below", 6.0)])


if __name__ == '__main__':
    unittest.main()
