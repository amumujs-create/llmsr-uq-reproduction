#!/usr/bin/env python3
"""Pinned A1 search, then paired original split/structural conformal evaluation.

This wrapper never edits upstream. Backend substitutions are explicit metadata;
a dev smoke run is not a reproduction of the paper's held-out aggregate.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream'
PINNED_COMMIT = 'b58630470144a3da6944c8f45bdbef63af38d460'
DEFAULT_PROBLEM = 'lsr_synth_phys_osc_po10'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    """Strict JSON; nonfinite raw numeric state remains available in NPZ files."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def verify_sources():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    if manifest['commit'] != PINNED_COMMIT:
        raise RuntimeError('Unexpected upstream commit')
    files = []
    for entry in manifest['files']:
        path = UPSTREAM / entry['path']
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != entry['git_blob_sha'] or len(data) != entry['bytes']:
            raise RuntimeError('Pinned source differs: ' + entry['path'])
        files.append({'path': entry['path'], 'git_blob_sha': blob, 'sha256': hashlib.sha256(data).hexdigest()})
    return {'commit': PINNED_COMMIT, 'files': files}


def select_problem_ids(available_ids, args):
    """Select before any search; never replace missing or failed tasks."""
    available = set(available_ids)
    if len(available) != len(available_ids):
        raise RuntimeError('Duplicate instance IDs in the pinned subset')
    membership = json.loads((UPSTREAM / 'dev_set.json').read_text())['subsets']
    if args.subset not in membership:
        raise ValueError('Unsupported subset: ' + args.subset)
    if args.dev or args.heldout:
        dev_ids = set(membership[args.subset])
        missing_dev = dev_ids - available
        if missing_dev:
            raise RuntimeError('Frozen dev IDs missing: ' + repr(sorted(missing_dev)))
        selected = dev_ids if args.dev else available - dev_ids
    elif args.only is not None:
        selected = {s.strip() for s in args.only.split(',') if s.strip()}
        if not selected:
            raise ValueError('--only must contain at least one instance ID')
    else:
        if args.subset != 'lsr_synth_phys_osc':
            raise ValueError('Choose --only, --dev, or --heldout for this subset')
        selected = {DEFAULT_PROBLEM}
    missing = selected - available
    if missing or not selected:
        raise RuntimeError('Selected IDs not found: ' + repr(sorted(missing)))
    return selected


def load_local_problems(args):
    """Same row order, RNG and private split functions as upstream load_subset."""
    import numpy as np
    import pyarrow.parquet as pq
    from benchmark.loader import Problem, Split, _as_1d, _as_2d, _subsample, _carve_calibration
    from benchmark.expressions import parse_expression, free_constants, is_numeric

    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    relative = args.subset + '/train-00000-of-00001.parquet'
    entry = next(f for f in manifest['files'] if f['path'] == relative)
    path = ROOT / 'data' / relative
    digest = sha256(path)
    if digest != entry['sha256'] or path.stat().st_size != entry['bytes']:
        raise RuntimeError('Pinned data file differs: ' + relative)
    rows = pq.read_table(path).to_pylist()
    selected = select_problem_ids([rec['instance_id'] for rec in rows], args)
    found = []
    for i, rec in enumerate(rows):
        if rec['instance_id'] not in selected:
            continue
        rng = np.random.default_rng(args.seed + i)
        names = [str(v) for v in rec['input_vars']]
        full = Split(_as_2d(rec['train_input']), _as_1d(rec['train_output']))
        capped = _subsample(full, 4000, rng)
        train, cal = _carve_calibration(capped.X, capped.y, 0.2, rng)
        test = Split(_as_2d(rec['test_input']), _as_1d(rec['test_output']))
        ood = (Split(_as_2d(rec['ood_input']), _as_1d(rec['ood_output']))
               if rec.get('ood_input') is not None else None)
        try:
            expr = parse_expression(rec['gt_expression'], names)
            consts, numeric = free_constants(expr, names), is_numeric(expr, names)
        except Exception:
            consts, numeric = [], False
        problem = Problem(instance_id=rec['instance_id'], subset=args.subset,
                          description=rec['description'], input_vars=names,
                          output_var=str(rec['output_vars'][0]), gt_expression=rec['gt_expression'],
                          train=train, cal=cal, test=test, ood=ood, full_train=full,
                          gt_free_constants=consts, gt_is_numeric=numeric)
        found.append((problem, i))
    missing = selected - {p.instance_id for p, _ in found}
    if missing or not found:
        raise RuntimeError('Selected IDs not found: ' + repr(sorted(missing)))
    provenance = {'repository': manifest['repository'], 'revision': manifest['revision'],
                  'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    return found, provenance


def install_call_log(generator, path):
    """Observe original SDK invocations, preserving all original kwargs/behavior.

    SDK-internal HTTP retries are not visible here; these are not billed-call counts.
    """
    original = generator.client.chat.completions.create
    counts = {'sdk_invocations': 0, 'sdk_successes': 0, 'sdk_errors': 0}

    def recorded(*positional, **kwargs):
        counts['sdk_invocations'] += 1
        index, started = counts['sdk_invocations'], time.monotonic()
        append_json(path, {'event': 'request', 'sdk_invocation': index,
                           'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'kwargs': kwargs})
        try:
            response = original(*positional, **kwargs)
        except BaseException as exc:
            counts['sdk_errors'] += 1
            append_json(path, {'event': 'error', 'sdk_invocation': index,
                               'elapsed_seconds': time.monotonic() - started,
                               'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        counts['sdk_successes'] += 1
        append_json(path, {'event': 'response', 'sdk_invocation': index,
                           'elapsed_seconds': time.monotonic() - started,
                           'response': response.model_dump(mode='json')})
        return response

    generator.client.chat.completions.create = recorded
    return counts


def save_buffer(pipe, directory):
    import numpy as np
    items, raw = [], {}
    logs = np.array([i.log_weight for i in pipe.buffer.items], dtype=float)
    weights = np.exp(logs - logs.max()) if logs.size else logs
    if weights.size:
        weights /= weights.sum()
    for index, item in enumerate(pipe.buffer.items):
        names = list(item.program.param_names)
        raw[f'member_{index}_theta'] = np.array([item.fit.theta_map[n] for n in names])
        raw[f'member_{index}_cov_theta'] = np.asarray(item.fit.cov_theta)
        items.append({'index': index, 'expression': item.program.expr,
                      'parameter_order': names, 'signature': item.signature,
                      'family': item.family, 'source_note': item.source_note,
                      'tempered_weight': item.weight, 'log_weight': item.log_weight,
                      'untempered_weight_all_buffer_members': weights[index],
                      'fit': asdict(item.fit)})
    np.savez_compressed(directory / 'buffer_numeric.npz', **raw)
    write_json(directory / 'buffer.json', {
        'temperature': pipe.buffer.temperature, 'weighting': pipe.cfg.weighting,
        'note': 'BMA renormalizes weights over usable members on each queried split; weights here cover the full buffer.',
        'members': items, 'scalers': pipe.scalers()})


def finalize_pipeline(pipe, directory, record):
    """Keep the original failure even if diagnostic serialization also fails."""
    if pipe is None:
        return
    errors = []

    def attempt(label, action):
        try:
            return action()
        except Exception as exc:
            errors.append({'stage': label, 'error_type': type(exc).__name__, 'error': str(exc)})
            return None

    attempt('conversations', lambda: write_json(directory / 'conversations.json', pipe.conversation_log()))
    attempt('buffer', lambda: save_buffer(pipe, directory))
    record['search'] = attempt('search_diagnostics', lambda: pipe.search_diagnostics())
    record['completed_rounds'] = attempt('completed_rounds', lambda: len(pipe.history))
    record['n_buffer_members'] = attempt('n_buffer_members', lambda: len(pipe.buffer.items))
    record['best_expression'] = (attempt('best_expression', lambda: pipe.best_expression())
                                 if record['n_buffer_members'] else None)
    if errors:
        record['finalization_errors'] = errors
        if record['status'] != 'failed':
            record.update(status='failed', error_type='ArtifactFinalizationError',
                          error='Failed to preserve one or more required diagnostic artifacts')


def evaluate_paired(pipe, problem, directory):
    import numpy as np
    from bma_conformal import SplitConformal, StructuralConformal
    from benchmark.metrics import evaluate_point, calibration_report, normalized_width, interval_score, attains

    assert not pipe.cfg.standardize, 'This paper-protocol wrapper uses physical units.'
    comp_cal = pipe.bma.predict_components(problem.cal.X)
    split = SplitConformal(alpha=pipe.cfg.alpha)
    split.fit(problem.cal.y, comp_cal['mean'])
    structural = StructuralConformal(alpha=pipe.cfg.alpha)
    structural.fit_normalized(problem.cal.y, comp_cal['mean'], structural.sigma(comp_cal))
    if not np.isclose(structural.q, pipe.conformal.q, equal_nan=True):
        raise AssertionError('Independent original structural calibration disagrees with fitted pipeline')
    arrays = {'train_X': problem.train.X, 'train_y': problem.train.y}
    metrics = {}
    for name, data in [('cal', problem.cal), ('id', problem.test), ('ood', problem.ood)]:
        if data is None:
            continue
        comp = comp_cal if name == 'cal' else pipe.bma.predict_components(data.X)
        arrays.update({f'{name}_X': data.X, f'{name}_y': data.y})
        for component, values in comp.items():
            arrays[f'{name}_{component}'] = np.asarray(values)
        sigma = structural.sigma(comp)
        arrays[f'{name}_structural_sigma'] = sigma
        moments_finite = {key: int(np.count_nonzero(~np.isfinite(np.asarray(val)))) for key, val in comp.items()}
        result = {'n': len(data), 'point': evaluate_point(data.y, comp['mean']),
                  'moment_nonfinite_counts': moments_finite, 'intervals': {}}
        for label, (lo, hi) in {
            'split': split.predict_interval(comp['mean']),
            'structural': structural.predict_interval_scaled(comp['mean'], sigma),
        }.items():
            arrays[f'{name}_{label}_lo'], arrays[f'{name}_{label}_hi'] = lo, hi
            original = calibration_report(data.y, lo, hi, alpha=pipe.cfg.alpha, normalise_by=data.y)
            finite = np.isfinite(data.y) & np.isfinite(lo) & np.isfinite(hi)
            result['intervals'][label] = {
                'coverage_upstream_finite_rows': original['coverage'],
                'coverage_all_rows_nonfinite_counted_as_failure': float(np.mean(finite & (data.y >= lo) & (data.y <= hi))),
                'attains_088_using_upstream_coverage': attains(original['coverage'], alpha=pipe.cfg.alpha, eps=0.02),
                'width_train_sd': normalized_width(lo, hi, problem.train.y),
                'width_eval_split_sd_upstream': original['mean_width'],
                'interval_score_train_sd': interval_score(data.y, lo, hi, alpha=pipe.cfg.alpha, y_train=problem.train.y),
                'interval_score_raw': interval_score(data.y, lo, hi, alpha=pipe.cfg.alpha),
                'nonfinite_target': int(np.count_nonzero(~np.isfinite(data.y))),
                'nonfinite_lo': int(np.count_nonzero(~np.isfinite(lo))),
                'nonfinite_hi': int(np.count_nonzero(~np.isfinite(hi))),
                'finite_rows_used_by_upstream': int(finite.sum()),
            }
        metrics[name] = result
    np.savez_compressed(directory / 'predictions.npz', **arrays)
    return {'calibration_quantiles': {'split': split.q, 'structural': structural.q},
            'same_point_predictor': True, 'train_target_sd': float(np.std(problem.train.y)),
            'structural_scale': 'sqrt(var_struct + 1e-6*var_noise)', 'splits': metrics}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', '--llm_model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--base-url', '--llm_base_url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--backend-label', default='local-backend-pilot')
    parser.add_argument('--backend-details-json', type=Path)
    parser.add_argument('--subset', default='lsr_synth_phys_osc',
                        choices=('lsr_synth_bio_pop_growth', 'lsr_synth_chem_react',
                                 'lsr_synth_matsci', 'lsr_synth_phys_osc'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--only', help='Comma-separated instance IDs; default is physics po10')
    selection.add_argument('--dev', action='store_true', help='Frozen five development problems in --subset')
    selection.add_argument('--heldout', action='store_true', help='All non-development problems in --subset')
    parser.add_argument('--list-problems', '--dry-run', action='store_true',
                        help='Verify source/data, print selected split sizes, and exit without LLM calls or output files')
    parser.add_argument('--iterations', type=int, default=125)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--restarts', type=int, default=6)
    parser.add_argument('--buffer-size', type=int, default=24)
    parser.add_argument('--fit-time-budget', type=float, default=120.0)
    parser.add_argument('--problem-timeout', type=int, default=10800)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.iterations, args.batch, args.restarts, args.buffer_size) < 1:
        parser.error('iterations, batch, restarts and buffer-size must be positive')
    if args.fit_time_budget <= 0 or args.problem_timeout <= 0:
        parser.error('Time budgets must be positive')
    source = verify_sources()
    sys.path.insert(0, str(UPSTREAM))
    import threadcap  # noqa: F401, before numpy imports
    problems, data_provenance = load_local_problems(args)
    if args.list_problems:
        print(json.dumps({'event': 'verified_selection', 'subset': args.subset, 'seed': args.seed,
                          'problems': [{'instance_id': p.instance_id, 'row_index': i,
                                        'split_seed': args.seed + i, 'train': len(p.train),
                                        'cal': len(p.cal), 'id': len(p.test),
                                        'ood': len(p.ood) if p.ood is not None else 0}
                                       for p, i in problems]}, indent=2))
        return 0
    from llmsr_pipeline import PipelineConfig, LLMSRUQPipeline
    from diagnostics import A1Feedback
    from run_benchmark import build_generator, build_prompt, _problem_seed, problem_deadline, ProblemTimeout
    from utils import set_seed
    set_seed(args.seed)
    directory = (args.output_dir or ROOT / 'outputs' / ('paired_A1_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ('numpy', 'scipy', 'torch', 'sympy', 'scikit-learn', 'pyarrow', 'datasets', 'openai'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'args': vars(args),
        'wrapper_sha256': sha256(__file__), 'source': source, 'data': data_provenance,
        'python': sys.version, 'platform': platform.platform(), 'versions': versions,
        'backend_details': json.loads(args.backend_details_json.read_text()) if args.backend_details_json else None,
        'selected_instance_ids': [p.instance_id for p, _ in problems],
        'interpretation': 'Selected-subset A1 run; selection alone does not establish paper reproduction. Backend, quantization, and budget differences must be disclosed.',
        'seed_limit': 'Original proposer does not send seed to backend; local seeds control data splits/fitting. Exact LLM replay requires saved responses.',
        'budget_limit': 'SDK invocations are logged; SDK-internal HTTP retries are not counted.',
    }
    write_json(directory / 'run_metadata.json', metadata)
    print(json.dumps({'event': 'started', 'output_dir': str(directory), 'problems': len(problems), 'iterations': args.iterations}), flush=True)
    failed = 0
    for problem, row_index in problems:
        subdir = directory / problem.instance_id
        subdir.mkdir()
        cfg = PipelineConfig(iterations=args.iterations, batch_hypotheses=args.batch,
                             n_restarts=args.restarts, max_nfev=2000, fit_time_budget=args.fit_time_budget,
                             buffer_size=args.buffer_size, conformal='structural', weighting='hybrid',
                             standardize=False, fallback=False, rung='A1', ablation='A1',
                             llm_calls_budget=args.iterations, seed=_problem_seed(args.seed, problem.instance_id),
                             base_seed=args.seed, verbose=False)
        write_json(subdir / 'config.json', asdict(cfg))
        record = {'instance_id': problem.instance_id, 'subset': problem.subset, 'row_index': row_index,
                  'split_seed': args.seed + row_index, 'problem_seed': cfg.seed, 'backend_label': args.backend_label,
                  'requested_proposer_rounds': args.iterations, 'status': 'starting'}
        write_json(subdir / 'result.json', record)
        started, pipe, calls, interrupted = time.monotonic(), None, {}, False
        try:
            generator = build_generator(cfg, 'local', args.model, args.base_url)
            calls = install_call_log(generator, subdir / 'llm_calls.jsonl')
            pipe = LLMSRUQPipeline(cfg, generator, feedback_fn=A1Feedback(problem.input_vars))
            with (subdir / 'pipeline_stdout.log').open('w') as log, contextlib.redirect_stdout(log):
                with problem_deadline(args.problem_timeout):
                    pipe.fit(problem.train.X, problem.train.y, problem.cal.X, problem.cal.y,
                             var_names=problem.input_vars, base_prompt=build_prompt(problem), X_target_for_cal=None)
                    if not pipe.buffer.items:
                        raise RuntimeError('No candidate entered the buffer')
                    record['evaluation'] = evaluate_paired(pipe, problem, subdir)
            record['status'] = 'completed'
        except (Exception, ProblemTimeout, KeyboardInterrupt) as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            interrupted = isinstance(exc, KeyboardInterrupt)
            (subdir / 'traceback.txt').write_text(traceback.format_exc())
        finally:
            finalize_pipeline(pipe, subdir, record)
            record.update(elapsed_seconds=time.monotonic() - started, calls=calls)
            record['artifact_hashes'] = {p.name: sha256(p) for p in subdir.iterdir() if p.is_file() and p.name != 'result.json'}
            write_json(subdir / 'result.json', record)
            append_json(directory / 'results.jsonl', record)
            failed += int(record['status'] != 'completed')
            print(json.dumps(clean({'event': 'problem_finished', 'instance_id': problem.instance_id,
                                    'status': record['status'], 'completed_rounds': record.get('completed_rounds'),
                                    'elapsed_seconds': record['elapsed_seconds'], 'result': str(subdir / 'result.json')})), flush=True)
        if interrupted:
            return 130
    print(json.dumps({'event': 'finished', 'failed': failed, 'total': len(problems), 'output_dir': str(directory)}), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
