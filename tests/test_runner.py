"""Offline checks for partition integrity and durable failure reporting."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_runner as runner


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'lsr_synth_phys_osc_po{i}' for i in range(44)]
        self.parser = runner.build_parser()

    def test_default_is_prespecified_smoke_problem(self):
        args = self.parser.parse_args([])
        self.assertEqual(runner.select_problem_ids(self.ids, args), {'lsr_synth_phys_osc_po10'})
        self.assertEqual(args.iterations, 125)

    def test_frozen_dev_and_heldout_partition_exactly(self):
        dev = runner.select_problem_ids(self.ids, self.parser.parse_args(['--dev']))
        heldout = runner.select_problem_ids(self.ids, self.parser.parse_args(['--heldout']))
        frozen = json.loads((runner.UPSTREAM / 'dev_set.json').read_text())['subsets']['lsr_synth_phys_osc']
        self.assertEqual(dev, set(frozen))
        self.assertEqual(len(dev), 5)
        self.assertEqual(len(heldout), 39)
        self.assertFalse(dev & heldout)
        self.assertEqual(dev | heldout, set(self.ids))

    def test_unknown_problem_is_not_silently_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'Selected IDs not found'):
            runner.select_problem_ids(self.ids, self.parser.parse_args(['--only', 'unknown']))

    def test_missing_dev_id_is_error_even_for_heldout(self):
        available = [v for v in self.ids if v != 'lsr_synth_phys_osc_po25']
        with self.assertRaisesRegex(RuntimeError, 'Frozen dev IDs missing'):
            runner.select_problem_ids(available, self.parser.parse_args(['--heldout']))

    def test_selectors_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(['--dev', '--heldout'])

    def test_different_subset_requires_explicit_selection(self):
        args = self.parser.parse_args(['--subset', 'lsr_synth_matsci'])
        with self.assertRaisesRegex(ValueError, 'Choose --only'):
            runner.select_problem_ids(['lsr_synth_matsci_matsci0'], args)


class FailureReportingTests(unittest.TestCase):
    def pipe(self):
        def no_best_expression():
            raise AssertionError('An empty buffer must not be queried for a winning equation')

        return SimpleNamespace(buffer=SimpleNamespace(items=[]), history=[],
                               conversation_log=lambda: [], search_diagnostics=lambda: {},
                               best_expression=no_best_expression)

    def test_artifact_error_does_not_replace_original_failure(self):
        record = {'status': 'failed', 'error_type': 'ConnectionError', 'error': 'endpoint unavailable'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'save_buffer', side_effect=OSError('disk error')):
            runner.finalize_pipeline(self.pipe(), Path(d), record)
            self.assertTrue((Path(d) / 'conversations.json').exists())
        self.assertEqual(record['error_type'], 'ConnectionError')
        self.assertEqual(record['error'], 'endpoint unavailable')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['n_buffer_members'], 0)
        self.assertIsNone(record['best_expression'])
        self.assertEqual(record['finalization_errors'][0]['stage'], 'buffer')

    def test_missing_required_artifacts_prevents_success(self):
        record = {'status': 'completed'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'save_buffer', side_effect=OSError('disk error')):
            runner.finalize_pipeline(self.pipe(), Path(d), record)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['error_type'], 'ArtifactFinalizationError')


if __name__ == '__main__':
    unittest.main()
