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
