"""Checks for regime-statement prompts, parsing and scoring."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'llm_statements'))
import hashlib  # noqa: E402
import json  # noqa: E402

from parse_score import agreement, majority, parse, parse_mech, score, to_statement  # noqa: E402
from elicit import assemble_s2, messages as s_messages  # noqa: E402
from prompts import (ICE_WORLDS, L1_ICE, L2_ICE, NO_ICE, QUESTION, QUESTION_MECH, SETTINGS, WORLDS,  # noqa: E402
                     build_prompt, check_no_leak, describe)
from collect import call_mock, jobs, jobs_ice  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'ccpp_twin'))
from regime_rules import from_mechanisms  # noqa: E402

LLM_DIR = Path(__file__).resolve().parents[1] / 'analysis' / 'llm_statements'

SUMMARY = {'n': 146, 'at_lo': 4.4, 'at_hi': 6.9, 'slope': -0.3, 'se': 0.5}


class PromptChecks(unittest.TestCase):
    def test_descriptions_do_not_name_the_shape(self):
        for world in WORLDS:
            for level in ('L0', 'L1', 'L2'):
                self.assertEqual(check_no_leak(describe(world, level)), [], (world, level))

    def test_l0_is_world_independent(self):
        self.assertEqual(len({describe(w, 'L0') for w in WORLDS}), 1)

    def test_l2_numbers_match_world(self):
        self.assertIn('364 MW', describe('cap_mild', 'L2'))
        self.assertIn('357 MW', describe('cap_strong', 'L2'))
        self.assertIn('24 degC', describe('evap', 'L2'))
        self.assertNotIn('limited to', describe('base', 'L2'))

    def test_window_summary_and_question_present(self):
        msg = build_prompt('evap', 'L1', SUMMARY)[-1]['content']
        self.assertIn('146 hourly records', msg)
        self.assertTrue(msg.endswith(QUESTION))
        self.assertEqual(len(SETTINGS), 4)


class AntiIcingPromptChecks(unittest.TestCase):
    """ccpp-anti-icing-world.md §6, §9 step 4."""

    def test_original_prompts_unchanged(self):
        # SHA-256 of every main-collection and S1/S2 prompt before the anti-icing wording was added (2026-11-05).
        summ = {tuple(k.split('|')): v for k, v in json.load(open(LLM_DIR / 'window_summaries.json')).items()}
        main = json.dumps([j['messages'] for j in jobs(summ)])
        strat = json.dumps([s_messages(w, l, st) for st in ('S1', 'S2') for l in ('L1', 'L2') for w in WORLDS])
        self.assertEqual(hashlib.sha256(main.encode()).hexdigest(),
                         'c6d876810d1915c30e48ca2f79b31b93e1662192c0dd7ac16ef063dd3c2849f8')
        self.assertEqual(hashlib.sha256(strat.encode()).hexdigest(),
                         '5ae32b711e884c14aa905dc8b72169c0fa8460c347a72d9a067b16670c2e0df0')

    def test_no_leak_in_new_wording(self):
        self.assertEqual(check_no_leak(QUESTION_MECH), [])
        for w in WORLDS + ICE_WORLDS:
            for lv in ('L1', 'L2'):
                self.assertEqual(check_no_leak(describe(w, lv, ice_note=True)), [], (w, lv))

    def test_l0_same_for_all_seven_worlds(self):
        self.assertEqual(len({describe(w, 'L0', ice_note=True) for w in WORLDS + ICE_WORLDS}), 1)

    def test_ice_note_only_on_original_worlds(self):
        for w in WORLDS:
            self.assertIn(NO_ICE, describe(w, 'L1', ice_note=True))
            self.assertNotIn(NO_ICE, describe(w, 'L1'))
            self.assertEqual(describe(w, 'L2', ice_note=True).count(NO_ICE), 1)
        for w in ICE_WORLDS:
            d1, d2 = describe(w, 'L1'), describe(w, 'L2')
            self.assertIn(L1_ICE, d1)
            self.assertNotIn(NO_ICE, d2)
            self.assertNotIn('no inlet air conditioning', d1)
            self.assertNotIn(L2_ICE, d1)
            self.assertIn(L2_ICE, d2)

    def test_l2_ice_numbers(self):
        self.assertIn('7 degC', describe('ice', 'L2'))
        self.assertIn('70%', describe('ice', 'L2'))
        self.assertNotIn('limited to', describe('ice', 'L2'))
        self.assertIn('364 MW', describe('ice+cap_mild', 'L2'))
        self.assertIn('rated limit', describe('ice+cap_mild', 'L1'))
        for w in ICE_WORLDS:  # the bleed loss and output change are not given (§6)
            self.assertNotIn('0.25', describe(w, 'L2'))
            self.assertNotIn('bleed', describe(w, 'L2').split(L1_ICE)[-1])

    def test_question_choice(self):
        for w in ICE_WORLDS:
            self.assertTrue(build_prompt(w, 'L1')[-1]['content'].endswith(QUESTION_MECH))
        for w in WORLDS:
            self.assertTrue(build_prompt(w, 'L1', ice_note=True)[-1]['content'].endswith(QUESTION))


class IceWindowSummaryChecks(unittest.TestCase):
    """ccpp-anti-icing-world.md §9 step 5: window summaries for the ice worlds."""
    summ = json.load(open(LLM_DIR / 'window_summaries.json'))

    def test_every_ice_world_and_setting_present(self):
        for w in ICE_WORLDS:
            for st in SETTINGS:
                self.assertEqual(self.summ[f'{w}|{st}']['n'], 146, (w, st))

    def test_same_training_rows_as_original_worlds(self):
        for w in ICE_WORLDS:
            for st in SETTINGS:
                for k in ('n', 'at_lo', 'at_hi'):
                    self.assertEqual(self.summ[f'{w}|{st}'][k], self.summ[f'base|{st}'][k])

    def test_d1_prompts_do_not_leak(self):
        for w in ICE_WORLDS:
            for st in SETTINGS:
                for lv in ('L1', 'L2'):
                    msg = build_prompt(w, lv, self.summ[f'{w}|{st}'])[-1]['content']
                    self.assertEqual(check_no_leak(msg), [], (w, st, lv))
                    self.assertTrue(msg.endswith(QUESTION_MECH))


class IceStageJobChecks(unittest.TestCase):
    """ccpp-anti-icing-world.md §6 call budget and §9 step 6."""
    summ = {tuple(k.split('|')): v for k, v in json.load(open(LLM_DIR / 'window_summaries.json')).items()}

    def test_job_count_and_budget(self):
        js = jobs_ice(self.summ)
        self.assertEqual(len(js), 30)
        self.assertEqual(len(js) * 21, 630)
        ice = [j for j in js if j['world'] in ICE_WORLDS]
        self.assertEqual(len(ice), 2 * 2 * 5)
        orig = [j for j in js if j['world'] in WORLDS]
        self.assertEqual({(j['level'], j['data']) for j in orig}, {('L1', 'D0'), ('L2', 'D0')})
        self.assertEqual(len({(j['world'], j['level'], j['data'], j['setting']) for j in js}), 30)

    def test_ice_stage_prompts(self):
        for j in jobs_ice(self.summ):
            msg = j['messages'][-1]['content']
            self.assertEqual(check_no_leak(msg), [])
            if j['world'] in WORLDS:
                self.assertIn(NO_ICE, msg)
                self.assertTrue(msg.endswith(QUESTION))
            else:
                self.assertTrue(msg.endswith(QUESTION_MECH))

    def test_mock_answers_parse(self):
        for j in jobs_ice(self.summ):
            text, _ = call_mock(j['messages'], 0.7, 3)
            ok = (parse_mech(text) if j['world'] in ICE_WORLDS else parse(text))['ok']
            self.assertTrue(ok, j['world'])


class ParseMechChecks(unittest.TestCase):
    def test_two_cold_causes_kept(self):
        r = parse_mech('{"boundaries": ['
                       '{"side": "below", "at_c": 7, "rh_above": 70, "mechanism": "anti_icing", "output_trend": "falls"},'
                       '{"side": "below", "at_c": 6, "rh_above": null, "mechanism": "power_cap",'
                       ' "output_trend": "about the same"},'
                       '{"side": "below", "at_c": 4, "mechanism": "power_cap"}]}')
        self.assertTrue(r['ok'])
        self.assertEqual(r['duplicates'], 1)
        self.assertEqual([(i['mechanism'], i['at_c'], i['rh_above'], i['output_trend']) for i in r['items']],
                         [('anti_icing', 7.0, 70.0, 'falls'), ('power_cap', 6.0, None, 'about the same')])
        self.assertEqual(from_mechanisms(r['items']), [('below', 7.0, 70.0), ('below', 6.0)])

    def test_old_format_and_unknown_values(self):
        r = parse_mech('{"boundaries": [{"side": "above", "at_c": 25, "confidence": 0.5}]}')
        self.assertEqual(r['items'][0]['mechanism'], 'other')
        self.assertIsNone(r['items'][0]['rh_above'])
        r = parse_mech('{"boundaries": [{"side": "below", "at_c": 5, "mechanism": "Anti-Icing", "rh_above": "null",'
                       ' "output_trend": "drops"}]}')
        self.assertEqual((r['items'][0]['mechanism'], r['items'][0]['rh_above'], r['items'][0]['output_trend']),
                         ('anti_icing', None, None))
        r = parse_mech('{"boundaries": [{"side": "below", "at_c": 5, "mechanism": "frost guard"}]}')
        self.assertEqual(r['items'][0]['mechanism'], 'other')

    def test_failures(self):
        self.assertEqual(parse_mech('none')['error'], 'no_json')
        self.assertFalse(parse_mech('{"boundaries": [{"side": "left", "at_c": 3}]}')['ok'])
        self.assertTrue(parse_mech('{"boundaries": []}')['ok'])


class ParseScoreChecks(unittest.TestCase):
    def test_parse_valid_and_innermost(self):
        r = parse('text {"boundaries": [{"side": "below", "at_c": 5, "confidence": 0.4},'
                  ' {"side": "below", "at_c": 8}], "rationale": "x"} trailing')
        self.assertTrue(r['ok'])
        self.assertEqual(r['duplicates'], 1)
        self.assertEqual(to_statement(r['boundaries']), [('below', 8.0)])

    def test_parse_skips_think_block(self):
        r = parse('<think>maybe {"boundaries": [{"side": "above", "at_c": 30}]}</think>{"boundaries": []}')
        self.assertTrue(r['ok'])
        self.assertEqual(r['boundaries'], [])

    def test_parse_failures_recorded(self):
        self.assertEqual(parse('no json here')['error'], 'no_json')
        self.assertFalse(parse('{"boundaries": [{"side": "left", "at_c": 3}]}')['ok'])
        self.assertTrue(parse('{"boundaries": []}')['ok'])

    def test_score_categories(self):
        self.assertTrue(score('base', [])['all_correct'])
        self.assertEqual(score('base', [('above', 25.0)])['hot'], 'false_alarm')
        self.assertEqual(score('cap_strong', [])['cold'], 'miss')
        self.assertEqual(score('cap_strong', [('below', 13.5)])['cold'], 'location')
        self.assertTrue(score('evap+cap_mild', [('below', 8.0), ('above', 22.0)])['all_correct'])

    def test_majority_and_agreement(self):
        sts = [[('above', 22.0)], [('above', 26.0)], [('above', 24.0), ('below', 5.0)], []]
        m = majority(sts)
        self.assertEqual(m, [('above', 24.0)])
        self.assertTrue(agreement([('above', 25.0)], m))
        self.assertFalse(agreement([('above', 25.0), ('below', 5.0)], m))


class StrategyChecks(unittest.TestCase):
    def test_strategy_prompts_do_not_leak(self):
        for s in ('S1', 'S2'):
            for w in WORLDS:
                for lv in ('L1', 'L2'):
                    self.assertEqual(check_no_leak(s_messages(w, lv, s)[1]['content']), [], (s, w, lv))

    def test_s2_assembly(self):
        r = assemble_s2('{"gt_generator_limit": {"reached_in_range": true, "gt_mw_at_15c": 343, "limit_mw": 364,'
                        ' "gt_mw_per_degc": -2.2}, "inlet_anti_icing_heating": {"active": true, "below_c": 4},'
                        ' "inlet_evaporative_cooling": {"active": true, "above_c": 24}}')
        self.assertEqual([s for s, _ in r['stmt']], ['below', 'above'])
        self.assertAlmostEqual(r['stmt'][0][1], 15 - 21 / 2.2)
        self.assertEqual(assemble_s2('{"gt_generator_limit": {"reached_in_range": false}}')['stmt'], [])
        self.assertFalse(assemble_s2('nothing')['ok'])


class RealPosthocChecks(unittest.TestCase):
    """real_posthoc.py summary (ccpp-real-data.md §10): regret, interval and win share on a known input."""

    def test_summarize_known_cells(self):
        import pandas as pd
        from real_posthoc import placebo_statements, summarize
        st = placebo_statements()
        self.assertEqual(st["above 30"], [("above", 30.0)])
        self.assertEqual(st["POOL(LLM, above 29)"], [("above", 29.0)])
        # statement A is 1 MW better than R0 in every cell, B is 2 MW worse in one of two cells.
        rows = []
        for i, setting in enumerate(["hot->cold", "mid->hot"]):
            for rep in range(2):
                rows.append({"statement": "A", "setting": setting, "w": 2.5, "d": 10, "rep": rep,
                             "R0_data": 5.0, "blend_0.5": 4.0})
                rows.append({"statement": "B", "setting": setting, "w": 2.5, "d": 10, "rep": rep,
                             "R0_data": 5.0, "blend_0.5": 5.0 + 2.0 * i})
        t, _ = summarize(pd.DataFrame(rows), boot=200)
        self.assertAlmostEqual(t.loc["A", "diff"], -1.0)
        self.assertAlmostEqual(t.loc["A", "lo"], -1.0)
        self.assertAlmostEqual(t.loc["A", "win"], 1.0)
        self.assertAlmostEqual(t.loc["B", "diff"], 1.0)
        self.assertAlmostEqual(t.loc["B", "win"], 0.0)
        self.assertAlmostEqual(t.loc["B", "harm_cells"], 0.5)


class ShapeStageChecks(unittest.TestCase):
    """Shape stage (ccpp-shape-elicitation.md §3-§4): prompts, parser, scorer on answers with a known truth."""

    def test_jobs_and_prompts(self):
        from collect import jobs_shape
        from prompts import QUESTION_SHAPE, SHAPE_WORLDS, check_no_leak
        js = jobs_shape()
        self.assertEqual(len(js), 12)
        self.assertEqual({j["world"] for j in js}, set(SHAPE_WORLDS))
        self.assertEqual(check_no_leak(QUESTION_SHAPE), [])
        for j in js:
            text = j["messages"][-1]["content"]
            self.assertTrue(text.endswith(QUESTION_SHAPE))
            self.assertEqual(check_no_leak(text.split("Question:")[0]), [])
            self.assertEqual("no inlet anti-icing system" in text, j["world"] != "ice")
        self.assertNotIn("more weakly", QUESTION_SHAPE)

    def test_parse_shape(self):
        from parse_score import parse_shape
        p = parse_shape('{"boundaries": [{"side": "below", "at_c": 6, "mechanism": "power_cap", '
                        '"response_beyond": "near zero"}, {"side": "below", "at_c": 4, "mechanism": "power_cap", '
                        '"response_beyond": "weaker"}, {"side": "above", "at_c": 24, "mechanism": "evap-cooling", '
                        '"response_beyond": "flatter"}]}')
        self.assertTrue(p["ok"])
        self.assertEqual(p["duplicates"], 1)
        by = {i["mechanism"]: i for i in p["items"]}
        self.assertEqual(by["power_cap"]["at_c"], 6.0)               # innermost cold boundary kept
        self.assertEqual(by["power_cap"]["response_beyond"], "near_zero")
        self.assertIsNone(by["evap_cooling"]["response_beyond"])     # unknown value -> None (scored wrong)
        self.assertFalse(parse_shape('{"boundaries": [{"side": "below", "at_c": null}]}')["ok"])

    def test_scorer_on_mock_answers(self):
        import tempfile
        from collect import call_mock, jobs_shape
        import score_shape as ss
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "shape").mkdir()
            for tag in ss.TAGS:
                with open(Path(d) / "shape" / f"{tag}.jsonl", "w") as f:
                    for j in jobs_shape():
                        for k in range(21):
                            text, _ = call_mock(j["messages"], 0.7, k)
                            f.write(json.dumps({"world": j["world"], "level": j["level"], "response": text,
                                                "error": None}) + "\n")
            rows = ss.load(Path(d))
        # mock: power_cap near_zero (correct), evap weaker (correct), anti_icing near_zero (wrong, truth opposite)
        v = ss.verdicts(rows)
        cap = v["P1"]["by_model"]["gpt-oss-120b"]
        self.assertEqual(cap["n"], 126)
        self.assertAlmostEqual(cap["shape_rate"], 1.0)
        self.assertAlmostEqual(v["P3"]["by_model"]["gpt-oss-120b"]["shape_rate"], 1.0)
        self.assertEqual(v["P2"]["by_model"]["gpt-oss-120b"]["n"], 21)
        self.assertAlmostEqual(v["P2"]["by_model"]["gpt-oss-120b"]["shape_rate"], 0.0)
        self.assertTrue(v["P1"]["pass"])
        self.assertTrue(v["P2"]["pass"])
        self.assertFalse(v["P3"]["pass"])
        self.assertAlmostEqual(v["P5"]["cap"], 1.0)
        rep = ss.reports(rows)["gpt-oss-120b"]
        self.assertEqual(rep["base_false_alarm_rate"], 0.0)
        self.assertEqual(rep["parse_failures"], 0)

    def test_location_rule(self):
        import score_shape as ss
        it = {"mechanism": "power_cap", "side": "below", "at_c": 11.5, "response_beyond": "near_zero"}
        self.assertIsNotNone(ss.loc_item([it], "cap_strong", "cap"))   # |11.5 - 9| <= 3
        self.assertIsNone(ss.loc_item([it], "cap_mild", "cap"))        # |11.5 - 6| > 3
        self.assertIsNone(ss.loc_item([dict(it, at_c=12.5)], "cap_strong", "cap"))  # 3.5 > 3
        self.assertIsNone(ss.loc_item([dict(it, mechanism="other")], "cap_strong", "cap"))
        self.assertIsNone(ss.loc_item([it], "evap", "cap"))            # world without a cap


class Shape2StageChecks(unittest.TestCase):
    """Shape stage 2 (ccpp-shape-elicitation-2.md §2-§4): twin truth, prompts, parser, scorer on known answers."""

    def test_twin_truth_shapes(self):
        import numpy as np
        from ccpp_twin import simulate, with_modules
        ap, v = 1012.94, 52.08   # CCPP medians (AP, V)

        def slope(cfg, lo, hi, rh=50.0):
            a = np.linspace(lo, hi, 21)
            pe = simulate(a, np.full(21, ap), np.full(21, rh), np.full(21, v), cfg)["pe_clean"]
            return np.polyfit(a, pe, 1)[0]
        base, chill, gen = with_modules(), with_modules(inlet_chill=True), with_modules(hot_gen_limit=True)
        normal = slope(base, 10, 18)
        self.assertLess(normal, -2.0)
        for cfg in (chill, gen):                              # inside the boundary the worlds match base
            self.assertAlmostEqual(slope(cfg, 10, 16), slope(base, 10, 16), delta=0.1)
        self.assertLess(abs(slope(chill, 22, 35)), 0.1 * abs(normal))             # near_zero
        self.assertLess(slope(gen, 30, 37), 1.2 * slope(base, 30, 37))             # same_or_stronger (steeper)
        x = np.array([5.0, 15.0, 25.0, 35.0])
        same = simulate(x, np.full(4, ap), np.full(4, 50.0), np.full(4, v))["pe"]
        self.assertTrue(np.allclose(same, simulate(x, np.full(4, ap), np.full(4, 50.0), np.full(4, v),
                                                   with_modules())["pe"]))   # default config unchanged

    def test_jobs_and_prompts(self):
        from collect import jobs_shape, jobs_shape2
        from prompts import QUESTION_SHAPE, QUESTION_SHAPE2, SHAPE2_WORLDS, check_no_leak
        js = jobs_shape2()
        self.assertEqual(len(js), 10)
        self.assertEqual({j["world"] for j in js}, set(SHAPE2_WORLDS))
        self.assertEqual(check_no_leak(QUESTION_SHAPE2), [])
        self.assertEqual(QUESTION_SHAPE2.replace(' or "inlet_chilling"', "")
                         .replace(", inlet_chilling is an inlet chiller", ""), QUESTION_SHAPE)
        for j in js:
            text = j["messages"][-1]["content"]
            self.assertTrue(text.endswith(QUESTION_SHAPE2))
            desc = text.split("Question:")[0]
            self.assertEqual(check_no_leak(desc), [])
            self.assertIn("no inlet anti-icing system", desc)
            for word in ("near zero", "barely", "constant", "steeper", "faster", "stronger", "weaker"):
                self.assertNotIn(word, desc.lower(), (j["world"], word))
        for j in jobs_shape():                               # stage-1 prompts unchanged
            self.assertTrue(j["messages"][-1]["content"].endswith(QUESTION_SHAPE))

    def test_parse_mechanisms(self):
        from parse_score import MECHANISMS2, parse_shape
        t = '{"boundaries": [{"side": "above", "at_c": 20, "mechanism": "inlet chilling", "response_beyond": "near_zero"}]}'
        self.assertEqual(parse_shape(t)["items"][0]["mechanism"], "other")          # stage-1 parsing unchanged
        self.assertEqual(parse_shape(t, MECHANISMS2)["items"][0]["mechanism"], "inlet_chilling")

    def test_scorer_on_mock_answers(self):
        import tempfile
        from collect import call_mock, jobs_shape2
        import score_shape2 as s2
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "shape2").mkdir()
            for tag in s2.TAGS:
                with open(Path(d) / "shape2" / f"{tag}.jsonl", "w") as f:
                    for j in jobs_shape2():
                        for k in range(21):
                            text, _ = call_mock(j["messages"], 0.7, k)
                            f.write(json.dumps({"world": j["world"], "level": j["level"], "response": text,
                                                "error": None}) + "\n")
            rows = s2.load(Path(d))
        # mock: chill near_zero (correct), gen_hot power_cap near_zero (wrong), evap weaker, cap near_zero (correct)
        v = s2.verdicts(rows)
        m = "gpt-oss-120b"
        self.assertEqual(v["Q1"]["by_model"][m]["n"], 42)
        self.assertEqual(v["Q2"]["by_model"][m]["n"], 21)                           # gen_hot judged on L2 only
        self.assertAlmostEqual(v["Q1"]["by_model"][m]["shape_rate"], 1.0)
        self.assertAlmostEqual(v["Q2"]["by_model"][m]["shape_rate"], 0.0)
        self.assertEqual(v["Q3"]["by_model"][m]["responses"], {"near_zero": v["Q3"]["by_model"][m]["n"]})
        self.assertEqual((v["Q1"]["pass"], v["Q2"]["pass"], v["Q3"]["pass"], v["Q4"]["pass"]),
                         (False, True, False, True))
        self.assertAlmostEqual(v["Q5"]["evap"], 1.0)
        rep = s2.reports(rows)[m]
        self.assertEqual(rep["base_false_alarm_rate"], 0.0)
        self.assertAlmostEqual(rep["by_level"]["cap"]["L1"]["shape_rate"], 1.0)

    def test_location_rule(self):
        import score_shape2 as s2
        it = {"mechanism": "inlet_chilling", "side": "above", "at_c": 22.5, "response_beyond": "near_zero"}
        self.assertIsNotNone(s2.loc_item([it], "chill"))
        self.assertIsNone(s2.loc_item([dict(it, at_c=23.5)], "chill"))              # 3.5 > 3
        self.assertIsNone(s2.loc_item([dict(it, mechanism="evap_cooling")], "chill"))
        self.assertIsNotNone(s2.loc_item([dict(it, mechanism="evap_cooling")], "chill", any_mechanism=True))
        self.assertIsNone(s2.loc_item([dict(it, side="below")], "chill"))


if __name__ == '__main__':
    unittest.main()


class IceScoringChecks(unittest.TestCase):
    """score_ice.py (anti-icing §16): known inputs -> known verdict pieces."""

    def setUp(self):
        import score_ice
        self.s = score_ice

    def item(self, mech, side, at, rh=None, trend=None):
        return {"mechanism": mech, "side": side, "at_c": at, "rh_above": rh, "output_trend": trend, "confidence": 0.8}

    def test_null_location_detected_and_parse_fails(self):
        txt = '{"boundaries": [{"side": "below", "at_c": null, "mechanism": "anti_icing", "output_trend": "falls"}]}'
        self.assertFalse(parse_mech(txt)["ok"])
        self.assertTrue(self.s.null_location(txt))
        self.assertFalse(self.s.null_location('{"boundaries": [{"side": "below", "at_c": 0. nine}]}'))
        self.assertFalse(self.s.null_location('{"boundaries": [{"side": "below", "at_c": 7}]}'))

    def test_anti_icing_and_cap_tolerance(self):
        self.assertTrue(self.s.ai_ok([self.item("anti_icing", "below", 10.0)]))
        self.assertFalse(self.s.ai_ok([self.item("anti_icing", "below", 10.5)]))
        self.assertFalse(self.s.ai_ok([self.item("power_cap", "below", 7.0)]))
        self.assertFalse(self.s.ai_ok([self.item("anti_icing", "above", 7.0)]))
        self.assertTrue(self.s.cap_ok([self.item("power_cap", "below", 3.0)]))
        self.assertFalse(self.s.cap_ok([self.item("power_cap", "below", 2.9)]))

    def test_mech_majority(self):
        a = [self.item("anti_icing", "below", 7.0, 70)]
        b = [self.item("anti_icing", "below", 5.0, None), self.item("power_cap", "below", 6.0)]
        maj = self.s.mech_majority([a, a, b])
        self.assertEqual(maj, [{"side": "below", "mechanism": "anti_icing", "at_c": 7.0, "rh_above": 70.0}])
        maj = self.s.mech_majority([a, b, b])   # rh given by 1 of 3 anti-icing items -> no humidity condition
        self.assertEqual(maj[0]["rh_above"], None)
        self.assertEqual([m["mechanism"] for m in maj], ["anti_icing", "power_cap"])
        self.assertEqual(self.s.mech_majority([a, [], []]), [])

    def test_cold_band_alarm(self):
        self.assertTrue(self.s.cold_band_alarm([("below", 8.0)]))
        self.assertFalse(self.s.cold_band_alarm([("below", 8.5), ("above", 3.0)]))


class RealValidationChecks(unittest.TestCase):
    """real_validation.py and risk_limits observed mode (ccpp-real-data.md §3-§7)."""

    def setUp(self):
        import real_validation
        self.rv = real_validation

    def test_recognition_is_word_level(self):
        self.assertIsNone(self.rv.RECOGNITION.search("reducing the sensitivity"))
        self.assertIsNotNone(self.rv.RECOGNITION.search("this looks like the UCI CCPP data"))
        self.assertIsNotNone(self.rv.RECOGNITION.search("Tufekci (2014)"))

    def test_pool_and_d1_statements(self):
        ans = {t: {("D0", None): [[("above", 30.0)]] * 3} for t in self.rv.TAGS}
        ans["gpt-oss-120b"] = {("D0", None): [[("below", 8.0)]] * 3, ("D1", "mid->hot"): [[], [], [("above", 25.0)]]}
        st = self.rv.build_statements(ans)
        self.assertEqual(st["gpt-oss-120b|S0maj"], [("below", 8.0)])
        self.assertEqual(st["POOL|POOL"], [("above", 30.0)])       # 4 of 5 models name the hot side
        self.assertEqual(st["gpt-oss-120b|D1maj|mid->hot"], [])

    def test_d1_rows_only_on_own_setting(self):
        import pandas as pd
        cells = pd.DataFrame({"statement": ["m|D1maj|mid->hot"] * 2 + ["m|S0maj"] * 2,
                              "setting": ["mid->hot", "hot->cold"] * 2})
        out = self.rv.rows_for_rules(cells)
        self.assertEqual(sorted(out[out.rule == "m|D1maj"].setting), ["mid->hot"])
        self.assertEqual(len(out[out.rule == "m|S0maj"]), 2)

    def test_observed_mode_linear_output_gives_zero_r0(self):
        import numpy as np
        from risk_limits import run
        rng = np.random.default_rng(0)
        n = 9568
        x = {"AT": rng.uniform(2, 37, n), "V": rng.uniform(25, 81, n), "AP": rng.uniform(993, 1033, n),
             "RH": rng.uniform(25, 100, n)}
        pe = 500 - 2.0 * x["AT"] - 0.3 * x["V"] + 0.05 * x["AP"] - 0.1 * x["RH"]
        cells = run(x, 1, external={"real": {"none": []}}, observed=pe)
        self.assertEqual(set(cells.module), {"real"})
        self.assertLess(cells.R0_data.max(), 1e-6)
        self.assertLess((cells["blend_0.5"] - cells.R0_data).abs().max(), 1e-9)   # empty statement = R0


class WorldBlindControlChecks(unittest.TestCase):
    """World-blind controls (ccpp-world-blind-controls.md §2-§5): statement sets, structural ceiling, verdict rule."""

    def test_statement_sets(self):
        from controls import DOMAIN, placebo_statements
        st = placebo_statements()
        self.assertEqual(len(st), 17)
        self.assertEqual(st["P|above 18"], [("above", 18.0)])
        self.assertEqual(st["P|below 16"], [("below", 16.0)])
        self.assertEqual(DOMAIN["D_real"], [("below", 8.0), ("above", 27.0)])

    def test_world_blind_structural_ceiling(self):
        from controls import DOMAIN, placebo_statements, world_blind_structural_ceiling
        c = world_blind_structural_ceiling({**placebo_statements(), **DOMAIN})
        self.assertEqual(len(c), 20)
        self.assertLessEqual(max(c.values()), 0.4)          # §4: no world-blind statement fits all five worlds
        self.assertEqual(c["P|below 6"], 0.4)               # cap_mild and cap_strong only

    def test_model_defaults_and_external(self):
        from controls import build_external, model_defaults
        s0 = {("m", "L1", "base"): [[("above", 25.0)], [("above", 26.0)], []],
              ("m", "L1", "evap"): [[("above", 24.0)]] * 3}
        self.assertEqual(model_defaults(s0), {"m|L1": [("above", 25.5)]})
        ext, meta, m = build_external(s0, {})
        self.assertEqual(set(ext), {"base", "cap_mild", "cap_strong", "evap", "evap+cap_mild"})
        self.assertEqual(ext["cap_mild"]["m|L1|M"], [("above", 25.5)])    # same in every world
        self.assertEqual(ext["evap"]["m|L1|HA"], [("above", 24.0)])         # HA is world-specific (S0 majority, no S2)
        self.assertEqual(ext["cap_mild"]["m|L1|HA"], [])
        self.assertEqual(ext["evap"]["D_real"], [("below", 8.0), ("above", 27.0)])
        self.assertEqual(meta["m|L1"]["split"], "dev")

    def test_verdicts_known_table(self):
        import pandas as pd
        from controls import verdicts
        rows = []
        for i in range(6):
            ha = 7.5 if i < 4 else 8.5   # HA beats D_real, M and P_mean in 4 of 6 cells; harm never worse
            rows.append({"cell": f"c{i}", "split": "validation", "HA": ha, "D_real": 8.0, "M": 8.0, "P_mean": 8.0,
                         "P_best": 7.0, "HA_harm": 0.0, "P_harm_mean": 0.1, "D_real_cold": 7.0, "D_real_hot": 9.0,
                         "n_P_beaten": 10})
        t = pd.DataFrame(rows).set_index("cell")
        v = verdicts(t)
        self.assertEqual(v["wins"], {"K1": 4, "K2": 4, "K3": 4, "K4": 6})
        self.assertTrue(v["K1"] and v["K2"] and v["K3"] and v["K4"])
        t.loc["c3", "HA"] = 8.5                                  # 3 of 6 -> K1-K3 fail
        v = verdicts(t)
        self.assertFalse(v["K1"] or v["K2"] or v["K3"])
        self.assertEqual(v["report"]["HA_lt_D_real_hot"], 6)


class ShapeAssemblyChecks(unittest.TestCase):
    """HA_shape (ccpp-shape-assembly.md §2): per-boundary slope factors in compose, shape lookup, verdict rule."""

    def test_compose_scalar_unchanged_and_per_boundary(self):
        import numpy as np
        from regime_rules import compose
        stmt = [("below", 6.0), ("above", 24.0)]
        at = np.linspace(-5.0, 35.0, 81)
        e, sd, sp = 15.0, -2.5, -2.6
        ref = compose(stmt, e, at, sd, sp, 0.0, False)
        self.assertTrue(np.allclose(compose(stmt, e, at, sd, sp, [0.0, 0.0], False), ref))     # list == scalar
        half = compose(stmt, e, at, sd, sp, 0.5, False)
        self.assertTrue(np.allclose(compose(stmt, e, at, sd, sp, [0.5, 0.5], False), half))
        mixed = compose(stmt, e, at, sd, sp, [0.0, 0.5], False)
        cold, hot = at < 6.0, at > 24.0
        self.assertTrue(np.allclose(mixed[cold], ref[cold]))                                    # cold side: factor 0
        self.assertTrue(np.allclose(mixed[hot], half[hot]))                                     # hot side: factor 0.5
        self.assertTrue(np.allclose(mixed[~cold & ~hot], ref[~cold & ~hot]))
        with self.assertRaises(ValueError):
            compose(stmt, e, at, sd, sp, [0.0], False)
        # humidity-conditioned boundary keeps its own factor
        ice = [("below", 7.0, 70.0), ("above", 24.0)]
        rh = np.where(at < 0, 90.0, 40.0)
        out = compose(ice, e, at, sd, sp, [-1.0, 0.5], False, rh)
        ref_ice = compose(ice, e, at, sd, sp, 0.5, False, rh)            # same grouping and grid as the scalar path
        self.assertTrue(np.allclose(out[hot], ref_ice[hot]))             # hot rows: only the 0.5 boundary applies
        self.assertFalse(np.allclose(out[at < 0], ref_ice[at < 0]))      # wet cold rows: factor -1 differs from 0.5

    def test_shape_lookup_and_factors(self):
        from shape_assembly import FACTOR, factors_for, shape_at
        items = [{"side": "above", "at_c": 25.0, "response_beyond": "weaker"},
                 {"side": "above", "at_c": 23.0, "response_beyond": "weaker"},
                 {"side": "above", "at_c": 24.0, "response_beyond": "same_or_stronger"},
                 {"side": "above", "at_c": 30.0, "response_beyond": "near_zero"},     # outside 3 degC
                 {"side": "below", "at_c": 6.0, "response_beyond": None}]              # unknown answer ignored
        self.assertEqual(shape_at(items, "above", 24.0), ("weaker", 3))
        self.assertEqual(shape_at(items, "below", 6.0), (None, 0))
        self.assertEqual(shape_at(items + [{"side": "above", "at_c": 24.0, "response_beyond": "same_or_stronger"}],
                                  "above", 24.0), (None, 4))                            # tie -> default
        facs, labs = factors_for([("below", 6.0), ("above", 24.0)], items)
        self.assertEqual(facs, [0.0, FACTOR["weaker"]])
        self.assertEqual(labs, [None, "weaker"])

    def test_verdicts_known_table(self):
        import pandas as pd
        from shape_assembly import verdicts
        rows = []
        for c in ["gemma3-27b|L1", "gemma3-27b|L2", "gpt-oss-20b|L1", "gpt-oss-20b|L2", "llama33-70b|L1", "llama33-70b|L2"]:
            llama = c.startswith("llama")
            rows.append({"cell": c, "split": "validation", "HA": 7.7, "HA_half": 7.6, "SHAPE_ORACLE": 7.5,
                         "HA_shape": 7.9 if llama else 7.6, "HA_shape_harm": 0.02})
        v = verdicts(pd.DataFrame(rows).set_index("cell"))
        self.assertEqual(v["wins"], {"J1": 6, "J2": 2, "J3": 4, "J4": 0})
        self.assertTrue(v["J1"] and v["J2"] and v["J3"] and v["J4"])
        t = pd.DataFrame(rows).set_index("cell")
        t.loc[~t.index.str.startswith("llama"), "HA_shape"] = 7.4   # LLM shape beats fixed 0.5 in 4 cells -> J4 fails
        self.assertFalse(verdicts(t)["J4"])
