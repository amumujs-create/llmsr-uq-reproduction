"""Tool-hose twin: synthetic statements x risk-limit rules, before any real LLM answer (2026-11-12).

Usage: python3 analysis/robot_twin/hose_statements.py data/robot/h2017.urdf [--reps 20]

The target is a scalar: the usable J6 limit k (deg) where the hose turns taut.
Overshooting k (predicting more room than there is) stretches the hose; undershooting
wastes workspace, so errors are reported both ways.

Task B (condition extrapolation, main): the p=0 sweep is fully seen (k0 known, hose length
known from it); predict k1 at full pressure. Statement = claimed length change eps_s at full
pressure. Truth worlds eps in {-4, -2, 0, +2} % (trade literature range +2 .. -4 %).
A small-angle window at full pressure is seen with torque noise sigma (the checkable part).

Task A (angle extrapolation): only a slack window [k-d-w, k-d] of the p=0 sweep is seen.
Statement = claimed k_s. The checkable part: k_s must lie beyond the window.
R0 here = the window end (stay inside the tested range, never overshoots). The data-only
trend guess (quadratic zero crossing of the axial force) is reported as rule 'trend'.

Rules on D = k_statement - k_R0 (as in ccpp-risk-limit.md):
  R0 data/default, R2 full trust, blend_0.5, budget_5/10 deg, safe (accept only narrowing),
  gate (accept only if the window supports the statement), and in Task B est (choose eps
  from the window alone, no statement).
Twin assumptions as in hose.py; the predictor knows the kinematics and mount points (K1:
with these, Task B with a correct eps is geometry, the unknown is eps itself).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose  # noqa: E402
from h2017_dynamics import load_urdf  # noqa: E402

POSES = {'j5_30': (0, 30, 90, 0, 30, 0), 'j5_60': (0, 30, 90, 0, 60, 0), 'j5_90': (0, 30, 90, 0, 90, 0)}
SLACKS = [1.05, 1.10, 1.20, 1.35]
EPS_WORLDS = [-0.04, -0.02, 0.0, 0.02]
SIGMAS = [0.2, 1.0]                    # J6 torque noise [Nm] (assumption; Doosan estimator noise unknown)
GATE_CHI2 = 4.0                        # accept if SSE improves by > GATE_CHI2 * sigma^2
OVERSHOOT_DEG = 2.0
BUDGETS = [5.0, 10.0]
RULES = ['R0', 'R2', 'blend_0.5'] + [f'budget_{b:g}' for b in BUDGETS] + ['safe', 'gate']


def cell_eps(pname, slack, eps):
    """HoseCell at full pressure with length change eps (hose.EPS_PRESSURE * p = eps)."""
    return hose.HoseCell(pname, POSES[pname], slack, eps / hose.EPS_PRESSURE)


def apply_rules(k0, ks, accepted):
    d = ks - k0
    out = {'R0': k0, 'R2': ks, 'blend_0.5': k0 + 0.5 * d, 'safe': k0 + min(d, 0.0),
           'gate': ks if accepted else k0}
    for b in BUDGETS:
        out[f'budget_{b:g}'] = k0 + float(np.clip(d, -b, b))
    return out


# ---------------------------------------------------------------- Task B
def statements_b(eps):
    st = {'correct': eps, 'miss': 0.0, 'double': 2 * eps, 'half': 0.5 * eps,
          'default_shrink': -0.04}
    st['sign_flip'] = -eps if eps != 0 else 0.02   # in the eps=0 world: a false 'it lengthens'
    st['false_shrink'] = -0.04 if eps >= 0 else None
    return {k: v for k, v in st.items() if v is not None}


def task_b(chain, reps, rng):
    rows = []
    cache = {}

    def k_and_window(pname, s, e, a):
        key = (pname, s, round(e, 6))
        if key not in cache:
            c = cell_eps(pname, s, e)
            cache[key] = (min(c.taut_angle(chain), 360.0), c.torque(chain, a))   # beyond the sweep -> 360
        return cache[key]

    for pname in POSES:
        for s in SLACKS:
            k0 = hose.HoseCell(pname, POSES[pname], s, 0.0).taut_angle(chain)
            if not np.isfinite(k0):
                continue
            a = np.arange(5.0, max(5.0, 0.5 * k0) + 0.01, 2.5)      # small-angle window at full pressure
            grid_eps = sorted(set(EPS_WORLDS) | {0.04, -0.08, 0.01, -0.01, -0.03})
            for eps in EPS_WORLDS:
                k1, tau_true = k_and_window(pname, s, eps, a)
                for sigma in SIGMAS:
                    for rep in range(reps):
                        y = tau_true + rng.normal(0, sigma, len(a))
                        sse = {e: float(np.sum((y - k_and_window(pname, s, e, a)[1]) ** 2)) for e in grid_eps}
                        e_hat = min(sse, key=sse.get)
                        k_est = k_and_window(pname, s, e_hat, a)[0]
                        for sname, e_s in statements_b(eps).items():
                            ks = k_and_window(pname, s, e_s, a)[0]
                            if e_s not in sse:
                                sse[e_s] = float(np.sum((y - k_and_window(pname, s, e_s, a)[1]) ** 2))
                            accepted = sse[e_s] < sse[0.0] - GATE_CHI2 * sigma ** 2
                            preds = apply_rules(k0, ks, accepted)
                            preds['est'] = k_est
                            for rule, kh in preds.items():
                                rows.append(dict(task='B', pose=pname, slack=s, world=eps, sigma=sigma, rep=rep,
                                                 statement=sname, rule=rule, k_true=k1, k_pred=kh, k_R0=k0,
                                                 accepted=bool(accepted)))
    return rows


# ---------------------------------------------------------------- Task A
def r0_trend(chain, cell, a, tau):
    """Data-only guess: quadratic fit of axial force (tau / dl/dq) to its zero crossing."""
    q = np.deg2rad(np.array(cell.base_pose_deg, dtype=float))
    dl = []
    for ai in a:
        qq = q.copy(); qq[5] = np.deg2rad(ai)
        dq = np.zeros(6); dq[5] = 1e-5
        dl.append((hose.path_length(chain, qq + dq) - hose.path_length(chain, qq - dq)) / 2e-5)
    f = tau / np.array(dl)
    roots = np.roots(np.polyfit(a, f, 2))
    ahead = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > a[-1]]
    return min(min(ahead), 360.0) if ahead else 360.0


def task_a(chain, reps, rng):
    rows = []
    for pname in POSES:
        for s in SLACKS:
            cell = hose.HoseCell(pname, POSES[pname], s, 0.0)
            k = cell.taut_angle(chain)
            if not np.isfinite(k):
                continue
            for d in (10.0, 20.0):
                for w in (20.0, 40.0):
                    lo, hi = k - d - w, k - d
                    if lo < 5.0:
                        continue
                    a = np.arange(lo, hi + 0.01, 2.0)
                    tau_true = cell.torque(chain, a)
                    for sigma in SIGMAS:
                        for rep in range(reps):
                            y = tau_true + rng.normal(0, sigma, len(a))
                            k0 = hi
                            k_trend = r0_trend(chain, cell, a, y)
                            sts = {'correct': k, 'early_10': k - 10, 'late_10': k + 10,
                                   'early_20': k - 20, 'late_20': k + 20, 'inside_window': hi - 5}
                            for sname, ks in sts.items():
                                accepted = ks > hi            # the checkable part: no taut seen in the window
                                preds = apply_rules(k0, ks, accepted)
                                preds['trend'] = k_trend
                                for rule, kh in preds.items():
                                    rows.append(dict(task='A', pose=pname, slack=s, world=f'd{d:g}w{w:g}', sigma=sigma,
                                                     rep=rep, statement=sname, rule=rule, k_true=k, k_pred=kh,
                                                     k_R0=k0, accepted=bool(accepted)))
    return rows


# ---------------------------------------------------------------- summary
def summarize(rows, task, wrong):
    out = {}
    rs = [r for r in rows if r['task'] == task]
    rules = sorted({r['rule'] for r in rs}, key=lambda x: (RULES + ['est', 'trend']).index(x))
    for sigma in SIGMAS:
        tab = {}
        for st in sorted({r['statement'] for r in rs}):
            for rule in rules:
                sel = [r for r in rs if r['sigma'] == sigma and r['statement'] == st and r['rule'] == rule]
                err = np.array([r['k_pred'] - r['k_true'] for r in sel])
                add = np.array([abs(r['k_pred'] - r['k_true']) - abs(r['k_R0'] - r['k_true']) for r in sel])
                tab[(st, rule)] = dict(mae=float(np.mean(np.abs(err))), over=float(np.mean(err > OVERSHOOT_DEG)),
                                       max_add=float(np.max(add)),
                                       acc=float(np.mean([r['accepted'] for r in sel])))
        frontier = {}
        for rule in rules:
            lc = tab[('correct', rule)]['mae']
            lw = np.mean([tab[(w, rule)]['mae'] for w in wrong])
            frontier[rule] = dict(L_correct=lc, L_wrong=float(lw))
        r0c, r0w = frontier['R0']['L_correct'], frontier['R0']['L_wrong']
        for rule, f in frontier.items():
            ps = [p for p in np.round(np.arange(0, 1.0001, 0.05), 2)
                  if p * f['L_correct'] + (1 - p) * f['L_wrong'] < p * r0c + (1 - p) * r0w - 1e-9]
            f['p_star'] = float(min(ps)) if ps else None
        out[sigma] = dict(table=tab, frontier=frontier)
    return out


def report(summary, task, title, rules_order):
    lines = [f'== Task {task}: {title}']
    for sigma, sm in summary.items():
        lines.append(f'-- torque noise sigma = {sigma} Nm. cells: MAE deg / overshoot rate / max added error vs R0 (deg)')
        sts = sorted({st for st, _ in sm['table']})
        rules = [r for r in rules_order if ('correct', r) in sm['table']]
        lines.append(f"   {'statement':15s}" + ''.join(f'{r:>20s}' for r in rules))
        for st in sts:
            cells = []
            for r in rules:
                c = sm['table'][(st, r)]
                cells.append(f"{c['mae']:6.1f}/{c['over']:4.2f}/{c['max_add']:5.1f}")
            lines.append(f'   {st:15s}' + ''.join(f'{x:>20s}' for x in cells)
                         + f"   gate acc {sm['table'][(st, 'gate')]['acc']:.2f}")
        lines.append('   break-even reliability p* (strict wrong set), L(correct), L(wrong):')
        for r in rules:
            f = sm['frontier'][r]
            lines.append(f"     {r:12s} p* {str(f['p_star']):>5s}   L_correct {f['L_correct']:6.1f}   L_wrong {f['L_wrong']:6.1f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urdf')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'runs' / 'hose_statements'))
    args = ap.parse_args()
    chain = load_urdf(args.urdf)
    rng = np.random.default_rng(20261112)
    rows = task_b(chain, args.reps, rng) + task_a(chain, args.reps, rng)
    sb = summarize(rows, 'B', wrong=['sign_flip'])
    sa = summarize(rows, 'A', wrong=['early_20', 'late_20'])
    lines = [f'reps {args.reps}, gate chi2 {GATE_CHI2}, overshoot = predicted k > true k + {OVERSHOOT_DEG} deg',
             f'pressure worlds eps {EPS_WORLDS}; hose EI {hose.EI_HOSE} N m^2', '']
    lines += report(sb, 'B', 'predict k at full pressure (statement = length change eps)', RULES + ['est'])
    lines += [''] + report(sa, 'A', 'predict k from a slack window (statement = k)', RULES + ['trend'])
    text = '\n'.join(lines)
    print(text)
    out = Path(args.out)
    out.with_suffix('.txt').write_text(text + '\n')
    js = {t: {str(sg): {'frontier': v['frontier'],
                        'table': {f'{st}|{r}': c for (st, r), c in v['table'].items()}}
              for sg, v in s.items()} for t, s in (('B', sb), ('A', sa))}
    out.with_suffix('.json').write_text(json.dumps(js, indent=1))


if __name__ == '__main__':
    main()
