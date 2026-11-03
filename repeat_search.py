#!/usr/bin/env python3
"""Repeat A1 with identical split/fit seeds; measure stochastic search variation.

No method changes and no LLM seed injection. This measures fresh backend draws
under fixed local conditions, not a proof that all randomness is isolated.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def require_same_data(arrays):
    import numpy as np
    keys = ['train_X', 'train_y', 'cal_X', 'cal_y', 'id_X', 'id_y', 'ood_X', 'ood_y']
    reference = arrays[0]
    for other in arrays[1:]:
        for key in keys:
            if not np.array_equal(reference[key], other[key]):
                raise ValueError('Repeated runs do not have identical data: ' + key)
    return {key: hashlib.sha256(reference[key].tobytes()).hexdigest() for key in keys}


def replay_intervals(arrays):
    """Recompute calibration/intervals from saved moments, without refit or LLM."""
    import numpy as np
    from bma_conformal import SplitConformal, StructuralConformal
    split, structural = SplitConformal(alpha=0.1), StructuralConformal(alpha=0.1)
    split.fit(arrays['cal_y'], arrays['cal_mean'])
    cal_comp = {k: arrays['cal_' + k] for k in ('var_struct', 'var_noise')}
    structural.fit_normalized(arrays['cal_y'], arrays['cal_mean'], structural.sigma(cal_comp))
    maximum = 0.0
    for part in ('cal', 'id', 'ood'):
        comp = {k: arrays[part + '_' + k] for k in ('var_struct', 'var_noise')}
        intervals = {
            'split': split.predict_interval(arrays[part + '_mean']),
            'structural': structural.predict_interval_scaled(arrays[part + '_mean'], structural.sigma(comp)),
        }
        for method, bounds in intervals.items():
            for side, recomputed in zip(('lo', 'hi'), bounds):
                saved = arrays[f'{part}_{method}_{side}']
                if not np.all(np.isfinite(saved)) or not np.all(np.isfinite(recomputed)):
                    raise ValueError('Nonfinite interval in replay')
                np.testing.assert_allclose(recomputed, saved, rtol=1e-10, atol=1e-12)
                maximum = max(maximum, float(np.max(np.abs(recomputed - saved))))
    return {'passed': True, 'max_absolute_difference': maximum,
            'scope': 'Saved prediction moments -> calibration and intervals; no coefficient refit, no LLM.'}


def summarize(directory, runs, instance_id):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'upstream'))
    from paired_runner import write_json
    rows, arrays, configurations = [], [], []
    for number in range(runs):
        problem = directory / f'run_{number:02d}' / instance_id
        result_path = problem / 'result.json'
        if not result_path.exists():
            rows.append({'run': number, 'status': 'missing_result'})
            continue
        result = json.loads(result_path.read_text())
        row = {'run': number, 'status': result['status']}
        rows.append(row)
        if result['status'] != 'completed':
            row['error'] = result.get('error')
            continue
        with np.load(problem / 'predictions.npz', allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        row.update(best_expression=result['best_expression'], elapsed_seconds=result['elapsed_seconds'],
                   completed_rounds=result['completed_rounds'], calls=result['calls'],
                   replay=replay_intervals(data))
        for part in ('id', 'ood'):
            evaluation = result['evaluation']['splits'][part]
            row[part] = {'nmse': evaluation['point']['nmse'], 'intervals': {
                method: {key: values[key] for key in (
                    'coverage_all_rows_nonfinite_counted_as_failure', 'width_train_sd', 'interval_score_train_sd')}
                for method, values in evaluation['intervals'].items()}}
        events = [json.loads(line) for line in (problem / 'llm_calls.jsonl').read_text().splitlines()]
        requests = [e for e in events if e['event'] == 'request']
        responses = [e for e in events if e['event'] == 'response']
        row['first_request_sha256'] = hashlib.sha256(json.dumps(requests[0]['kwargs'], sort_keys=True).encode()).hexdigest()
        row['first_response_sha256'] = hashlib.sha256(responses[0]['response']['choices'][0]['message']['content'].encode()).hexdigest()
        arrays.append(data)
        configurations.append(json.loads((problem / 'config.json').read_text()))
    report = {'requested_runs': runs, 'completed_runs': len(arrays), 'runs': rows,
              'scope': 'Fixed data and fitting settings; fresh stochastic LLM search. Descriptive pilot, not a paper reproduction or significance test.',
              'limitations': ['Backend randomness is not seeded by original code.',
                             'Wall-clock fitting limits can change restart counts on overloaded hardware.',
                             'Between-run spread is descriptive, not a calibrated predictive interval.',
                             'Replay validates saved moments to intervals, not refitting equation coefficients.']}
    if arrays:
        report['data_hashes'] = require_same_data(arrays)
        if any(c != configurations[0] for c in configurations[1:]):
            raise ValueError('Fitting/search configs differ between runs')
        report['identical_local_config'] = True
        completed = [r for r in rows if r['status'] == 'completed']
        report['identical_first_requests'] = len({r['first_request_sha256'] for r in completed}) == 1
        report['distinct_first_responses'] = len({r['first_response_sha256'] for r in completed})
    if len(arrays) >= 2:
        scale = float(np.std(arrays[0]['train_y']))
        means = np.stack([a['ood_mean'] for a in arrays])
        spreads = np.stack([np.sqrt(np.maximum(a['ood_var_struct'], 0)) for a in arrays])
        if not np.isfinite(means).all() or not np.isfinite(spreads).all() or scale <= 0:
            raise ValueError('Invalid values for between-search comparison')
        report['ood_search_variation'] = {
            'rms_between_run_mean_sd_train_units': float(np.sqrt(np.mean(np.var(means, axis=0, ddof=1))) / scale),
            'rms_within_run_structural_sd_train_units': float(np.sqrt(np.mean(spreads**2)) / scale),
            'note': 'The two spreads describe different sources; their ratio is not a coverage test.'}
    write_json(directory / 'repeat_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=125)
    parser.add_argument('--seed', type=int, default=0, help='Kept IDENTICAL in every repeat; controls data and fitting.')
    parser.add_argument('--only', default='lsr_synth_phys_osc_po10', help='One prespecified problem ID')
    parser.add_argument('--subset', default='lsr_synth_phys_osc')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--backend-label', default='unverified-backend')
    parser.add_argument('--backend-details-json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.runs < 2 or args.iterations < 1 or ',' in args.only:
        parser.error('Use at least two repeats, positive iterations and one problem ID.')
    directory = (args.output_dir or ROOT / 'outputs' / ('repeat_A1_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'repeat_plan.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    exits = []
    for number in range(args.runs):
        command = [sys.executable, str(ROOT / 'paired_runner.py'), '--seed', str(args.seed),
                   '--only', args.only, '--subset', args.subset, '--iterations', str(args.iterations),
                   '--model', args.model, '--base-url', args.base_url, '--backend-label', args.backend_label,
                   '--output-dir', str(directory / f'run_{number:02d}')]
        if args.backend_details_json:
            command.extend(['--backend-details-json', str(args.backend_details_json.resolve())])
        with (directory / f'run_{number:02d}.log').open('w') as log:
            outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        exits.append(outcome.returncode)
        print(json.dumps({'repeat': number, 'exit_code': outcome.returncode, 'output_dir': str(directory)}), flush=True)
        if outcome.returncode == 130 or outcome.returncode < 0:
            break
    report = summarize(directory, args.runs, args.only)
    print(json.dumps({'completed': report['completed_runs'], 'requested': args.runs,
                      'summary': str(directory / 'repeat_summary.json')}), flush=True)
    return 0 if len(exits) == args.runs and all(code == 0 for code in exits) else 1


if __name__ == '__main__':
    raise SystemExit(main())
