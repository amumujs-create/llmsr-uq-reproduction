"""Tool-hose twin, Task B sweeps around hose_statements.py (exploration, 2026-11-12).

Usage: python3 analysis/robot_twin/hose_statements_sweep.py data/robot/h2017.urdf [--reps 20]

Task B as in hose_statements.py (predict the J6 taut angle k1 at full pressure; statement =
length change eps), varied along:
  E1 torque noise sigma
  E2 size of the pressurized small-angle window (fraction of k0)
  E3 model mismatch: the truth differs from the predictor's hose model
     (torque x1.25 = cable lateral force, validate_hose_cable.py; hose EI x3 or /3)
  E4 k0 and the hose length estimated from a noisy p=0 sweep instead of known
  E5 asymmetric cost: loss = undershoot + c * overshoot (deg), best rule by reliability p
Strict wrong statement = sign_flip, as in hose_statements.py. All thresholds are assumptions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose  # noqa: E402
from h2017_dynamics import load_urdf  # noqa: E402
from hose_statements import EPS_WORLDS, GATE_CHI2, OVERSHOOT_DEG, POSES, SLACKS, apply_rules, statements_b  # noqa: E402

GRID_EPS = sorted(set(EPS_WORLDS) | {0.04, -0.08, 0.01, -0.01, -0.03})
SLACK_REL = np.arange(-0.03, 0.03001, 0.0025)     # hose-length search for E4 (relative)
KEEP = ['correct', 'sign_flip', 'default_shrink']
RULES = ['R0', 'R2', 'blend_0.5', 'budget_5', 'budget_10', 'safe', 'gate', 'est']


class Model:
    """Cached k and torque for (pose, slack, eps, EI); torque on a given angle grid."""

    def __init__(self, chain):
        self.chain, self.cache = chain, {}

    def get(self, pname, s, eps, a, ei=None):
        ei = hose.EI_HOSE if ei is None else ei
        key = (pname, round(s, 8), round(eps, 6), ei, a[0], a[-1], len(a))
        if key not in self.cache:
            old = hose.EI_HOSE
            try:
                hose.EI_HOSE = ei
                c = hose.HoseCell(pname, POSES[pname], s, eps / hose.EPS_PRESSURE)
                self.cache[key] = (min(c.taut_angle(self.chain), 360.0), c.torque(self.chain, a))
            finally:
                hose.EI_HOSE = old
        return self.cache[key]


def run_b(m, reps, rng, sigma, win_frac=0.5, truth_ei=None, truth_scale=1.0, estimate_k0=False):
    rows = []
    for pname in POSES:
        for s in SLACKS:
            k0_true = m.get(pname, s, 0.0, np.array([0.0]))[0]
            if k0_true >= 360.0:
                continue
            a = np.arange(5.0, max(5.0, win_frac * k0_true) + 0.01, 2.5)
            a0 = np.arange(0.0, min(k0_true + 30.0, 360.0) + 0.01, 2.0)
            for eps in EPS_WORLDS:
                k1, tau1 = m.get(pname, s, eps, a, truth_ei)
                tau1 = truth_scale * tau1
                tau0 = truth_scale * m.get(pname, s, 0.0, a0, truth_ei)[1]
                for rep in range(reps):
                    if estimate_k0:      # fit the hose length from the noisy p=0 sweep
                        y0 = tau0 + rng.normal(0, sigma, len(a0))
                        cands = [s * (1 + d) for d in SLACK_REL]
                        s_hat = min(cands, key=lambda c: np.sum((y0 - m.get(pname, c, 0.0, a0)[1]) ** 2))
                    else:
                        s_hat = s
                    k0 = m.get(pname, s_hat, 0.0, a)[0]
                    y = tau1 + rng.normal(0, sigma, len(a))
                    sse = {e: float(np.sum((y - m.get(pname, s_hat, e, a)[1]) ** 2)) for e in GRID_EPS}
                    k_est = m.get(pname, s_hat, min(sse, key=sse.get), a)[0]
                    for sname, e_s in statements_b(eps).items():
                        if sname not in KEEP:
                            continue
                        ks = m.get(pname, s_hat, e_s, a)[0]
                        accepted = sse[e_s] < sse[0.0] - GATE_CHI2 * sigma ** 2
                        preds = apply_rules(k0, ks, accepted)
                        preds['est'] = k_est
                        for rule, kh in preds.items():
                            rows.append((sname, rule, kh - k1, abs(k0 - k1), accepted))
    return rows


def stats(rows):
    out = {}
    for st in KEEP:
        for rule in RULES:
            sel = [r for r in rows if r[0] == st and r[1] == rule]
            err = np.array([r[2] for r in sel])
            out[(st, rule)] = dict(mae=float(np.mean(np.abs(err))), over=float(np.mean(err > OVERSHOOT_DEG)),
                                   acc=float(np.mean([r[4] for r in sel])), err=err)
    return out


def p_star(st, rule, cost=None):
    def loss(s, r):
        e = st[(s, r)]['err']
        return float(np.mean(np.abs(e))) if cost is None else float(np.mean(np.maximum(-e, 0) + cost * np.maximum(e, 0)))
    lc, lw, rc, rw = loss('correct', rule), loss('sign_flip', rule), loss('correct', 'R0'), loss('sign_flip', 'R0')
    ps = [p for p in np.round(np.arange(0, 1.0001, 0.05), 2) if p * lc + (1 - p) * lw < p * rc + (1 - p) * rw - 1e-9]
    return (min(ps) if ps else None), lc, lw


def line(name, st):
    acc_c, acc_w = st[('correct', 'gate')]['acc'], st[('sign_flip', 'gate')]['acc']
    g = st[('correct', 'gate')]; e = st[('correct', 'est')]; r0 = st[('correct', 'R0')]
    pg = p_star(st, 'gate')[0]
    return (f'   {name:22s} gate acc correct {acc_c:4.2f} sign_flip {acc_w:4.2f} | MAE correct: R0 {r0["mae"]:5.1f} '
            f'gate {g["mae"]:5.1f} est {e["mae"]:5.1f} (over {e["over"]:.2f}) | default_shrink gate '
            f'{st[("default_shrink", "gate")]["mae"]:5.1f} | p* gate {pg}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urdf')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'runs' / 'hose_statements_sweep.txt'))
    args = ap.parse_args()
    m = Model(load_urdf(args.urdf))
    rng = np.random.default_rng(20261113)
    L = [f'reps {args.reps}; gate chi2 {GATE_CHI2}; strict wrong = sign_flip; nominal hose EI {hose.EI_HOSE}', '']

    L.append('E1 torque noise (window 0.5 k0, model exact, k0 known)')
    for sg in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        L.append(line(f'sigma {sg}', stats(run_b(m, args.reps, rng, sg))))

    L.append('\nE2 pressurized window size (sigma 0.5)')
    for wf in (0.25, 0.5, 0.75, 0.9):
        L.append(line(f'window {wf} k0', stats(run_b(m, args.reps, rng, 0.5, win_frac=wf))))

    L.append('\nE3 model mismatch (sigma 0.2, window 0.5 k0); truth differs, predictor uses hose.py nominal')
    for name, kw in (('exact', {}), ('torque x1.25', dict(truth_scale=1.25)),
                     ('EI x3', dict(truth_ei=3 * hose.EI_HOSE)), ('EI /3', dict(truth_ei=hose.EI_HOSE / 3))):
        L.append(line(name, stats(run_b(m, args.reps, rng, 0.2, **kw))))

    L.append('\nE4 k0 and hose length fitted from a noisy p=0 sweep (window 0.5 k0)')
    for name, kw in (('sigma 0.2', dict(sigma=0.2)), ('sigma 1.0', dict(sigma=1.0)),
                     ('sigma 0.2, torque x1.25', dict(sigma=0.2, truth_scale=1.25)),
                     ('sigma 0.2, EI x3', dict(sigma=0.2, truth_ei=3 * hose.EI_HOSE))):
        st = stats(run_b(m, args.reps, rng, estimate_k0=True, **kw))
        L.append(line(name, st) + f" | R2 correct {st[('correct', 'R2')]['mae']:4.1f}")

    L.append('\nE5 asymmetric cost: loss = undershoot + c * overshoot; best rule by reliability p (k0 known)')
    for sg in (0.2, 1.0):
        st = stats(run_b(m, args.reps, rng, sg))
        for c in (1, 3, 10):
            ps = {r: p_star(st, r, cost=c)[0] for r in RULES}
            best = []
            for p in (0.1, 0.3, 0.5, 0.7, 0.9):
                exp = {}
                for r in RULES:
                    _, lc, lw = p_star(st, r, cost=c)
                    exp[r] = p * lc + (1 - p) * lw
                b = min(exp, key=exp.get)
                best.append(f'p{p}:{b}({exp[b]:.1f})')
            L.append(f'   sigma {sg} c {c:2d}: p* ' + ' '.join(f'{r}={ps[r]}' for r in RULES if r != 'R0'))
            L.append('                  best ' + '  '.join(best))
    text = '\n'.join(L)
    print(text)
    Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
