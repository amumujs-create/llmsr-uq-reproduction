"""Physical sanity checks for the CCPP twin simulator."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'ccpp_twin'))
from ccpp_twin import TwinConfig, simulate, wet_bulb_c, with_modules


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


if __name__ == '__main__':
    unittest.main()
