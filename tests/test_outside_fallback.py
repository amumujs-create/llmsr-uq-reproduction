"""Checks for HA_fb (docs/research/ccpp-twin-outside-fallback.md §2). Synthetic answers only."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
import outside_fallback as fb  # noqa: E402
import outside_regime as orr  # noqa: E402


def answer(gt=False, ice=False, evap=False, exh=False):
    return {"gt_generator_limit": {"reached_in_range": gt}, "inlet_anti_icing_heating": {"active": ice},
            "inlet_evaporative_cooling": {"active": evap}, "exhaust_or_steam_side_limit": {"active": exh}}


class FallbackChecks(unittest.TestCase):
    def test_majority_claim_without_boundary_gets_default(self):
        objs = [answer(gt=True)] * 6 + [answer()] * 4
        self.assertEqual(fb.fallback([], objs), [("below", 8.0)])

    def test_exact_half_is_not_a_majority(self):
        objs = [answer(gt=True)] * 5 + [answer()] * 5
        self.assertEqual(fb.fallback([], objs), [])

    def test_existing_boundary_keeps_the_llm_position(self):
        objs = [answer(gt=True, evap=True)] * 10
        self.assertEqual(fb.fallback([("above", 24.0)], objs), [("below", 8.0), ("above", 24.0)])
        self.assertEqual(fb.fallback([("below", 6.9), ("above", 24.0)], objs), [("below", 6.9), ("above", 24.0)])

    def test_all_four_devices_and_one_default_per_side(self):
        objs = [answer(gt=True, ice=True, evap=True, exh=True)] * 10
        self.assertEqual(fb.fallback([], objs), [("below", 8.0), ("above", 27.0)])
        self.assertEqual(fb.fallback([], [answer(ice=True)] * 10), [("below", 8.0)])
        self.assertEqual(fb.fallback([], [answer(exh=True)] * 10), [("above", 27.0)])

    def test_restricted_variant_ignores_other_devices(self):
        objs = [answer(evap=True, exh=True)] * 10
        self.assertEqual(fb.fallback([], objs, ("gt_limit",)), [])

    def test_no_s2_answers_is_ha(self):
        self.assertEqual(fb.fallback([("above", 25.0)], []), [("above", 25.0)])

    def test_claims_need_literal_true(self):
        self.assertEqual(fb.claims({"gt_generator_limit": {"reached_in_range": "yes"},
                                    "inlet_evaporative_cooling": None}), set())

    def test_with_fallback_touches_only_ha(self):
        ext = {"cap_strong": {"m|L1|HA": [], "m|L1|S0maj": [], "D_real": [("below", 8.0), ("above", 27.0)]}}
        raw = {("m", "L1", "cap_strong"): [answer(gt=True)] * 10}
        out = fb.with_fallback(ext, raw)
        self.assertEqual(out["cap_strong"]["m|L1|HA"], [("below", 8.0)])
        self.assertEqual(out["cap_strong"]["m|L1|S0maj"], [])
        self.assertEqual(ext["cap_strong"]["m|L1|HA"], [])                 # input not modified

    def test_seed_prefix_is_new(self):
        self.assertNotEqual(orr.sample_seed("evap|hot", 2.0, 5.0, 150, 0, fb.SEED_PREFIX),
                            orr.sample_seed("evap|hot", 2.0, 5.0, 150, 0))
        self.assertNotEqual(orr.noise_seed("evap", 4.3, 0, fb.SEED_PREFIX), orr.noise_seed("evap", 4.3, 0))

    def test_default_positions_are_the_d_real_numbers(self):
        self.assertEqual(sorted(fb.DEFAULT_AT.items()), sorted((s, a) for s, a in orr.D_REAL))


if __name__ == "__main__":
    unittest.main()
