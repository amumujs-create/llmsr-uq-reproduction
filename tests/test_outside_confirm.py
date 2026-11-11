"""Checks for the outside-regime confirmation (docs/research/ccpp-twin-outside-confirmation.md). Synthetic answers only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
import collect_newworld as cn  # noqa: E402
import info_axis as ia  # noqa: E402
import outside_confirm as oc  # noqa: E402
from elicit import S2_QUESTION  # noqa: E402
from prompts import QUESTION, check_no_leak  # noqa: E402


def s0_answer(bounds):
    return json.dumps({"boundaries": [{"side": s, "at_c": a, "confidence": 0.7} for s, a in bounds], "rationale": "x"})


def s2_answer(gt=False, sens=-2.2, lim=357.0, evap=None):
    return json.dumps({"gt_generator_limit": {"reached_in_range": gt, "gt_mw_at_15c": 343, "limit_mw": lim,
                                              "gt_mw_per_degc": sens},
                       "inlet_anti_icing_heating": {"active": False, "below_c": None},
                       "inlet_evaporative_cooling": {"active": evap is not None, "above_c": evap},
                       "exhaust_or_steam_side_limit": {"active": False, "above_c": None}})


def write_runs(root, tag):
    """A tiny synthetic collection for one model: main (S0), strategy (S2) and new-world files."""
    root = Path(root)
    (root / "strategy").mkdir(parents=True)
    (root / "newworld").mkdir()
    s0 = {"base": [], "cap_mild": [], "cap_strong": [], "evap": [("above", 25.0)], "evap+cap_mild": [("above", 25.0)]}
    s2 = {"base": s2_answer(), "cap_mild": s2_answer(gt=True, sens=-0.5), "cap_strong": s2_answer(gt=True, sens=-0.5),
          "evap": s2_answer(evap=25.0), "evap+cap_mild": s2_answer(gt=True, sens=-0.5, evap=25.0)}
    with open(root / f"{tag}.jsonl", "w") as f, open(root / "strategy" / f"{tag}.jsonl", "w") as g:
        for world in s0:
            for level in ("L1", "L2"):
                for k in range(3):
                    base = {"tag": tag, "model": tag, "world": world, "level": level, "data": "D0", "error": None,
                            "temperature": 0.7, "sample": k}
                    f.write(json.dumps({**base, "id": f"{tag}|{world}|{level}|{k}", "response": s0_answer(s0[world])}) + "\n")
                    g.write(json.dumps({**base, "id": f"{tag}|S2|{world}|{level}|{k}", "strategy": "S2",
                                        "response": s2[world]}) + "\n")
    with open(root / "newworld" / f"{tag}.jsonl", "w") as f:
        for world, bounds, s2a in (("chill", [("above", 21.0)], s2_answer(evap=21.0)), ("gen_hot", [], s2_answer())):
            for level in ("L1", "L2"):
                for k in range(3):
                    base = {"tag": tag, "model": tag, "stage": "newworld", "world": world, "level": level, "data": "D0",
                            "error": None, "temperature": 0.7, "sample": k}
                    f.write(json.dumps({**base, "id": f"{tag}|S0|{world}|{level}|{k}", "strategy": "S0",
                                        "response": s0_answer(bounds)}) + "\n")
                    f.write(json.dumps({**base, "id": f"{tag}|S2|{world}|{level}|{k}", "strategy": "S2",
                                        "response": s2a}) + "\n")
                f.write(json.dumps({**base, "id": f"{tag}|S0|{world}|{level}|bad", "strategy": "S0",
                                    "response": "not json"}) + "\n")


class CollectionChecks(unittest.TestCase):
    def test_call_count_and_ids(self):
        t = cn.tasks("m")
        self.assertEqual(len(t), cn.N_CALLS)
        self.assertEqual(cn.N_CALLS, 128)
        self.assertEqual(len({x[0] for x in t}), 128)
        self.assertEqual(sum(x[5] == 0.0 for x in t), 8)                 # one temperature-0 call per prompt

    def test_prompts_reuse_the_registered_wording(self):
        for world in cn.NEW_WORLDS:
            for level in cn.LEVELS:
                self.assertEqual(check_no_leak(cn.describe_new(world, level)), [])
                self.assertTrue(cn.messages(world, level, "S0")[1]["content"].endswith(QUESTION))
                self.assertTrue(cn.messages(world, level, "S2")[1]["content"].endswith(S2_QUESTION))
        self.assertIn("about 20 degC", cn.describe_new("chill", "L2"))
        self.assertNotIn("20 degC", cn.describe_new("chill", "L1"))
        self.assertNotIn("anti-icing", cn.describe_new("chill", "L2"))

    def test_models_file(self):
        tags = oc.model_tags(ROOT / "analysis" / "llm_statements" / "models_new.tsv")
        self.assertEqual(len(tags), 4)
        old = oc.model_tags(ROOT / "analysis" / "llm_statements" / "models.tsv")
        self.assertFalse(set(tags) & set(old))


class DesignChecks(unittest.TestCase):
    def test_tasks(self):
        self.assertEqual(len(oc.KINK), 4)
        self.assertEqual(len(oc.NOKINK), 4)
        self.assertEqual(oc.SCOPE, ("gen_hot|hot",))
        self.assertIn("chill|hot", oc.KINK)
        self.assertEqual(oc.SEED_PREFIX, "outside3")
        for task, (world, side, k, _) in oc.TASKS.items():
            self.assertIn(world, oc.TRUTH_ALL)
            self.assertIn(world, oc.D_AWARE)
            self.assertIn(world, oc.D_READ)
            for d in (1.0, 2.0, 4.0):
                lo, hi = ia.window_bounds(k, side, d, 10.0)
                self.assertTrue(hi < k if side == "hot" else lo > k)

    def test_defaults(self):
        self.assertEqual(oc.D_AWARE["chill"], [("above", 27.0)])
        self.assertEqual(oc.D_READ["chill"], [("above", 20.0)])
        self.assertEqual(oc.D_READ["evap"], [("above", 24.0)])
        self.assertEqual(oc.D_READ["cap_strong"], oc.D_AWARE["cap_strong"])
        self.assertEqual(oc.NEW_TRUTH["gen_hot"], [])

    def test_two_thirds(self):
        self.assertEqual([oc.two_thirds(n) for n in (2, 3, 4, 6, 8)], [2, 2, 3, 4, 6])


class LoaderChecks(unittest.TestCase):
    def test_statements_from_synthetic_collection(self):
        with tempfile.TemporaryDirectory() as d:
            write_runs(d, "m")
            done, counts = oc.complete_tags(d, ["m", "absent"])
            self.assertEqual(done, [])                                   # too few lines to count as complete
            self.assertEqual(counts["absent"], {"main": 0, "strategy": 0, "newworld": 0})
            stm, claims, fail = oc.statements(d, ["m"])
        self.assertEqual(stm["cap_strong"]["m|L1|HA"], [])               # sensitivity too small: position gated out
        self.assertEqual(stm["cap_strong"]["m|L1|HA_fb_gt"], [("below", 8.0)])
        self.assertEqual(stm["base"]["m|L1|HA_fb_gt"], [])
        self.assertEqual(stm["evap"]["m|L2|HA_fb_gt"], [("above", 25.0)])
        self.assertEqual(stm["evap+cap_mild"]["m|L1|HA_fb_gt"], [("below", 8.0), ("above", 25.0)])
        self.assertEqual(stm["chill"]["m|L1|HA_fb_gt"], [("above", 21.0)])   # S2 evap slot corroborated by S0
        self.assertEqual(stm["gen_hot"]["m|L2|HA_fb_gt"], [])
        self.assertEqual(claims["m|L1|cap_strong"]["gt_limit"], 1.0)
        self.assertEqual(fail["m|S0"], [4, 16])                          # one unparsed S0 answer per world and level


def frame(cells, ha_kink=0.7, ha_nokink=1.0, ha_scope=1.0, d_aware=0.85, d_read=0.7):
    rows = []
    for task, (world, side, k, kind) in oc.TASKS.items():
        for rep in range(2):
            kk = kind == "kink"
            r = dict(task=task, kind=kind, world=world, side=side, k=k, delta=1.0, w=5.0, n=50, sigma=4.3, rep=rep,
                     n_test=100, R0_abs=5.0, ORACLE=0.6 if kk else 1.0, D_real=0.8 if kk else 2.0,
                     D_aware=d_aware if kk else 1.0, D_read=d_read if kk else 1.0, P_side=0.9 if kk else 1.5)
            for c in cells:
                for var in oc.VARIANTS:
                    r[f"{c}|{var}"] = ha_kink if kk else (ha_scope if kind == "scope" else ha_nokink)
            rows.append(r)
    return pd.DataFrame(rows)


CELLS = [f"{t}|{lv}" for t in ("a", "b", "c") for lv in ("L1", "L2")]


class VerdictChecks(unittest.TestCase):
    def test_all_hold(self):
        v = oc.verdicts(frame(CELLS), CELLS)
        self.assertTrue(v["stage1"] and v["strong"])
        self.assertEqual((v["H5"]["of"], v["H6"]["of"]), (3, 3))
        self.assertEqual((v["H1"]["need"], v["H2"]["need"]), (5, 4))

    def test_scope_task_is_not_in_the_primary_verdict(self):
        v = oc.verdicts(frame(CELLS, ha_scope=2.0), CELLS)
        self.assertTrue(v["stage1"])
        self.assertTrue(v["S"]["predicted_harm_in_half_or_more"])
        self.assertEqual(v["per_cell"]["a|L1"]["harm"], 0.0)

    def test_l1_vs_default_and_l2_vs_reading(self):
        v = oc.verdicts(frame(CELLS, ha_kink=0.8), CELLS)                # beats D_aware 0.85, misses D_read 0.7 + 0.03
        self.assertTrue(v["H5"]["pass"])
        self.assertFalse(v["H6"]["pass"])
        self.assertTrue(v["stage1"])
        self.assertFalse(v["strong"])
        self.assertIsNone(v["per_cell"]["a|L1"]["H6"])
        self.assertIsNone(v["per_cell"]["a|L2"]["H5"])

    def test_h6_margin(self):
        self.assertTrue(oc.verdicts(frame(CELLS, ha_kink=0.72), CELLS)["H6"]["pass"])

    def test_harm_fails_stage1(self):
        v = oc.verdicts(frame(CELLS, ha_nokink=1.3), CELLS)
        self.assertFalse(v["H1"]["pass"])
        self.assertFalse(v["stage1"])

    def test_silence_fails_h2(self):
        v = oc.verdicts(frame(CELLS, ha_kink=1.0), CELLS)
        self.assertTrue(v["H1"]["pass"] and v["H3"]["pass"])
        self.assertFalse(v["H2"]["pass"])
        self.assertFalse(v["stage1"])

    def test_one_model_gives_no_verdict(self):
        cells = ["a|L1", "a|L2"]
        self.assertIn("no_verdict", oc.verdicts(frame(cells), cells))


if __name__ == "__main__":
    unittest.main()
