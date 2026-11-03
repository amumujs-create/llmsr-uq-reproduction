#!/usr/bin/env python3
"""Split recorded run time into LLM-call time and the rest (fitting, evaluation).

Reads result.json and llm_calls.jsonl under an output folder; calls no LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(result_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    llm_seconds, responses, errors = 0.0, 0, 0
    calls_path = result_path.with_name('llm_calls.jsonl')
    if calls_path.is_file():
        for line in calls_path.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') in ('response', 'error'):
                llm_seconds += event.get('elapsed_seconds', 0.0)
                responses += event['event'] == 'response'
                errors += event['event'] == 'error'
    total = result.get('elapsed_seconds')
    return {'run': str(result_path.parent), 'status': result.get('status'), 'total_seconds': total,
            'llm_seconds': llm_seconds, 'other_seconds': None if total is None else total - llm_seconds,
            'llm_responses': responses, 'llm_errors': errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--project-runs', type=int, default=545, help='Runs to project (default 109 problems x 5)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel runs assumed for the projection')
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted(args.output_dir.rglob('result.json'))]
    if not rows:
        raise SystemExit(f'No result.json under {args.output_dir}')
    for row in rows:
        print(json.dumps(row))
    done = [r for r in rows if r['status'] == 'completed' and r['total_seconds'] is not None]
    if not done:
        raise SystemExit('No completed run; nothing to project')
    mean = sum(r['total_seconds'] for r in done) / len(done)
    print(json.dumps({'completed_runs': len(done), 'all_runs': len(rows),
                      'mean_hours_per_run': mean / 3600,
                      'mean_llm_share': sum(r['llm_seconds'] for r in done) / sum(r['total_seconds'] for r in done),
                      'projected_runs': args.project_runs, 'assumed_workers': args.workers,
                      'projected_wall_days': mean * args.project_runs / args.workers / 86400}))
    print('Projection assumes the same per-run time when runs share the server; check this, not assume it.')


if __name__ == '__main__':
    main()
