"""Checks for regime-statement prompts, parsing and scoring."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'llm_statements'))
from parse_score import agreement, majority, parse, score, to_statement  # noqa: E402
from elicit import assemble_s2, messages as s_messages  # noqa: E402
from prompts import QUESTION, SETTINGS, WORLDS, build_prompt, check_no_leak, describe  # noqa: E402

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


if __name__ == '__main__':
    unittest.main()
