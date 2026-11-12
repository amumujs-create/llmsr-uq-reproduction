"""Mismatch-aware gates for the hose pressure statement (Task B, exploration 2026-11-12).

Usage: python3 analysis/robot_twin/hose_gate_mismatch.py data/robot/h2017.urdf [--reps 20]
       (reads runs/hose_cable_truth.npz from hose_cable_truth.py for the cable truth)

Problem (hose_statements_sweep.py E3): the nominal gate compares eps_s with eps = 0 using
hose.py as is; when the true hose differs from the model, the gate picks statements that
happen to explain the model error (wrong-sign acceptance 17-62 %).

Gates compared (accept the statement if the pressurized window SSE improves on eps = 0
by more than GATE_CHI2 * variance):
  gate_nom       nominal model, hose length fitted on the p=0 sweep (as E4)
  gate_cal       calibrate on the p=0 sweep, where eps = 0 is known: torque = c1*A + c2*B,
                 A = slack-thrust torque per unit EI, B = taut torque (so c1 = gain*EI, c2 = gain),
                 plus the hose length (+-5 %); then test eps_s with c1, c2 fixed. c2 is bounded to
                 GAIN_BOUNDS: with c2 free, a longer hose with a steeper taut rise fits a hard
                 cable wall equally well and the length runs to the search bound.
  gate_cal_infl  gate_cal, variance = max(sigma^2, calibrated p=0 residual^2 on the window angles)
  est_nom / est_cal   choose eps from the window alone with the nominal / calibrated model
Truths: exact, torque x1.25, EI x3, EI /3 (inside the calibration family),
shape (E_C x2, ALPHA_SHIFT 0: outside it), cable (MuJoCo elastic cable: outside it; the cable
scene has no wrist wrap, so the predictor uses R_WRAP = 0 there), and cable+wrap (same cable
truth, predictor keeps R_WRAP = 0.06: a geometry mismatch of 2.5-20 deg in k).
Only the cells present in the cable file are used, for every truth.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose  # noqa: E402
from h2017_dynamics import load_urdf  # noqa: E402
from hose_statements import GATE_CHI2, OVERSHOOT_DEG, POSES, apply_rules, statements_b  # noqa: E402

GRID_EPS = [-0.08, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
SLACK_REL = np.arange(-0.05, 0.05001, 0.0025)
GAIN_BOUNDS = (0.5, 2.0)   # torque-gain prior for the taut part; unbounded, length and taut gain trade off
KEEP = ['correct', 'sign_flip', 'default_shrink']
RULES = ['R0', 'R2', 'safe', 'gate_nom', 'gate_cal', 'gate_cal_infl', 'est_nom', 'est_cal']
TAUT_N = 20.0          # cable taut threshold, as validate_hose_cable.py
TRUTHS = {'exact': {}, 'torque x1.25': dict(scale=1.25), 'EI x3': dict(EI_HOSE=3 * hose.EI_HOSE),
          'EI /3': dict(EI_HOSE=hose.EI_HOSE / 3), 'shape': dict(E_C=2 * hose.E_C, ALPHA_SHIFT=0.0),
          'cable': None, 'cable+wrap': None}
PREDICTOR_WRAP = {'cable': 0.0}      # predictor geometry override per truth


@contextmanager
def hose_globals(**kw):
    old = {k: getattr(hose, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(hose, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(hose, k, v)


class Basis:
    """Nominal-shape torque split: A (slack thrust per unit EI), B (taut part), k; cached."""

    def __init__(self, chain):
        self.chain, self.cache = chain, {}

    def get(self, pname, s, eps, a):
        key = (pname, round(s, 8), round(eps, 6), a[0], a[-1], len(a), hose.R_WRAP)
        if key not in self.cache:
            c = hose.HoseCell(pname, POSES[pname], s, eps / hose.EPS_PRESSURE)
            with hose_globals(EI_HOSE=1.0):
                base = c.torque(self.chain, a, regime=False)
                full = c.torque(self.chain, a)
            self.cache[key] = (base, full - base, min(c.taut_angle(self.chain), 360.0))
        return self.cache[key]


def model_truth(chain, pname, s, eps, a, spec):
    spec = dict(spec); scale = spec.pop('scale', 1.0)
    with hose_globals(**spec):
        c = hose.HoseCell(pname, POSES[pname], s, eps / hose.EPS_PRESSURE)
        return scale * c.torque(chain, a), min(c.taut_angle(chain), 360.0)


def cable_truth(sweeps, pname, s, eps, a):
    r = sweeps[f'{pname}|{s}|{eps}']
    k = r[r[:, 1] > TAUT_N][0, 0]
    ok = a <= r[-1, 0]
    return np.interp(a[ok], r[:, 0], r[:, 2]), k, ok


def fit_two(y, A, B):
    """Least squares y ~ c1 A + c2 B with c1, c2 >= 0 (two-variable NNLS by enumeration)."""
    best = None
    M = np.column_stack([A, B])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    cands = [c] if np.all(c >= 0) else []
    for j in (0, 1):
        den = float(M[:, j] @ M[:, j])
        cj = np.zeros(2); cj[j] = max(float(M[:, j] @ y) / den, 0.0) if den > 0 else 0.0
        cands.append(cj)
    for cj in cands:
        rss = float(np.sum((y - M @ cj) ** 2))
        if best is None or rss < best[0]:
            best = (rss, cj)
    return best


def fit_gains(y, A, B, lo=GAIN_BOUNDS[0], hi=GAIN_BOUNDS[1]):
    """Least squares y ~ c1 A + c2 B with c1 >= 0 and lo <= c2 <= hi (active-set enumeration)."""
    M = np.column_stack([A, B])
    cands = []
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    if c[0] >= 0 and lo <= c[1] <= hi:
        cands.append(c)
    den = float(A @ A)
    for c2 in (lo, hi):                                   # c2 on a bound, c1 free (>= 0)
        c1 = max(float(A @ (y - c2 * B)) / den, 0.0) if den > 0 else 0.0
        cands.append(np.array([c1, c2]))
    dbb = float(B @ B)                                    # c1 = 0, c2 free within bounds
    c2 = min(max(float(B @ y) / dbb, lo), hi) if dbb > 0 else lo
    cands.append(np.array([0.0, c2]))
    best = min(cands, key=lambda cc: float(np.sum((y - M @ cc) ** 2)))
    return float(np.sum((y - M @ best) ** 2)), best


def default_statements(eps):
    return {k: v for k, v in statements_b(eps).items() if k in KEEP}


def run(chain, sweeps, reps, rng, sigma, truth, statements_fn=default_statements):
    """Rows (statement name, rule, k_pred - k_true, gate accepted). statements_fn(eps_true) -> {name: eps_s};
    eps_s values outside GRID_EPS are added to the gate comparison (score_hose.py feeds LLM statements here)."""
    with hose_globals(R_WRAP=PREDICTOR_WRAP.get(truth, hose.R_WRAP)):
        return _run(chain, sweeps, reps, rng, sigma, truth, statements_fn)


def _run(chain, sweeps, reps, rng, sigma, truth, statements_fn):
    bas = Basis(chain)
    rows = []
    cells = sorted({tuple(k.split('|')[:2]) for k in sweeps.files})
    for pname, s_txt in cells:
        s = float(s_txt)
        k0_nom = bas.get(pname, s, 0.0, np.array([0.0]))[2]
        a = np.arange(5.0, max(5.0, 0.5 * k0_nom) + 0.01, 2.5)
        a0 = np.arange(0.0, min(k0_nom + 30.0, 360.0) + 0.01, 2.0)
        if truth.startswith('cable'):
            tau0, _, ok0 = cable_truth(sweeps, pname, s, 0.0, a0)
            a0 = a0[ok0]
        else:
            tau0 = model_truth(chain, pname, s, 0.0, a0, TRUTHS[truth])[0]
        for eps in sorted({float(k.split('|')[2]) for k in sweeps.files}):
            if truth.startswith('cable'):
                tau1, k1, ok = cable_truth(sweeps, pname, s, eps, a)
                assert ok.all(), 'cable sweep shorter than the window'
            else:
                tau1, k1 = model_truth(chain, pname, s, eps, a, TRUTHS[truth])
            for rep in range(reps):
                y0 = tau0 + rng.normal(0, sigma, len(a0))
                y = tau1 + rng.normal(0, sigma, len(a))
                cands = [s * (1 + d) for d in SLACK_REL]
                # nominal: c1 = EI_HOSE, c2 = 1
                def nom_rss(c):
                    A0, B0, _ = bas.get(pname, c, 0.0, a0)
                    return float(np.sum((y0 - hose.EI_HOSE * A0 - B0) ** 2))
                s_nom = min(cands, key=nom_rss)
                # calibrated: c1, c2 free (>= 0) per candidate length
                fits = {c: fit_gains(y0, *bas.get(pname, c, 0.0, a0)[:2]) for c in cands}
                s_cal = min(fits, key=lambda c: fits[c][0])
                c1, c2 = fits[s_cal][1]
                A0c, B0c, _ = bas.get(pname, s_cal, 0.0, a0)
                res_w = np.interp(a, a0, y0 - c1 * A0c - c2 * B0c)   # p=0 misfit at the window angles
                var_infl = max(sigma ** 2, float(np.mean(res_w ** 2)))

                def sse_nom(e):
                    A, B, _ = bas.get(pname, s_nom, e, a)
                    return float(np.sum((y - hose.EI_HOSE * A - B) ** 2))

                def sse_cal(e):
                    A, B, _ = bas.get(pname, s_cal, e, a)
                    return float(np.sum((y - c1 * A - c2 * B) ** 2))
                Sn = {e: sse_nom(e) for e in GRID_EPS}
                Sc = {e: sse_cal(e) for e in GRID_EPS}
                k_nom = lambda e: bas.get(pname, s_nom, e, a)[2]  # noqa: E731
                k_cal = lambda e: bas.get(pname, s_cal, e, a)[2]  # noqa: E731
                est_nom, est_cal = k_nom(min(Sn, key=Sn.get)), k_cal(min(Sc, key=Sc.get))
                for sname, e_s in statements_fn(eps).items():
                    if e_s not in Sn:
                        Sn[e_s], Sc[e_s] = sse_nom(e_s), sse_cal(e_s)
                    acc = {'gate_nom': Sn[e_s] < Sn[0.0] - GATE_CHI2 * sigma ** 2,
                           'gate_cal': Sc[e_s] < Sc[0.0] - GATE_CHI2 * sigma ** 2,
                           'gate_cal_infl': Sc[e_s] < Sc[0.0] - GATE_CHI2 * var_infl}
                    base = apply_rules(k_cal(0.0), k_cal(e_s), False)
                    preds = {'R0': base['R0'], 'R2': base['R2'], 'safe': base['safe'],
                             'gate_nom': k_nom(e_s) if acc['gate_nom'] else k_nom(0.0),
                             'gate_cal': k_cal(e_s) if acc['gate_cal'] else k_cal(0.0),
                             'gate_cal_infl': k_cal(e_s) if acc['gate_cal_infl'] else k_cal(0.0),
                             'est_nom': est_nom, 'est_cal': est_cal}
                    for rule, kh in preds.items():
                        rows.append((sname, rule, kh - k1, acc.get(rule, False)))
    return rows


def summarize(rows):
    out = {}
    for st in KEEP:
        for rule in RULES:
            e = np.array([r[2] for r in rows if r[0] == st and r[1] == rule])
            out[(st, rule)] = dict(mae=float(np.mean(np.abs(e))), over=float(np.mean(e > OVERSHOOT_DEG)),
                                   acc=float(np.mean([r[3] for r in rows if r[0] == st and r[1] == rule])))
    for rule in RULES:
        lc, lw = out[('correct', rule)]['mae'], out[('sign_flip', rule)]['mae']
        rc, rw = out[('correct', 'R0')]['mae'], out[('sign_flip', 'R0')]['mae']
        ps = [p for p in np.round(np.arange(0, 1.0001, 0.05), 2) if p * lc + (1 - p) * lw < p * rc + (1 - p) * rw - 1e-9]
        out[('p*', rule)] = min(ps) if ps else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urdf')
    ap.add_argument('--reps', type=int, default=20)
    here = Path(__file__).resolve().parent
    ap.add_argument('--cable', default=str(here / 'runs' / 'hose_cable_truth.npz'))
    ap.add_argument('--out', default=str(here / 'runs' / 'hose_gate_mismatch.txt'))
    args = ap.parse_args()
    chain = load_urdf(args.urdf)
    sweeps = np.load(args.cable)
    rng = np.random.default_rng(20261114)
    L = [f'reps {args.reps}; gate chi2 {GATE_CHI2}; window 0.5 k0; strict wrong = sign_flip; '
         f'{len(sweeps.files) // 4} cells (those in the cable file)', '']
    for sigma in (0.2, 1.0):
        L.append(f'== torque noise sigma {sigma} Nm')
        L.append(f"   {'truth':13s} | accept correct / sign_flip: nom, cal, cal_infl | MAE correct: R0 gate_nom gate_cal "
                 f"gate_cal_infl est_nom est_cal safe R2 | p*: gate_nom gate_cal gate_cal_infl")
        for truth in TRUTHS:
            st = summarize(run(chain, sweeps, args.reps, rng, sigma, truth))
            accs = ' '.join(f"{st[('correct', g)]['acc']:.2f}/{st[('sign_flip', g)]['acc']:.2f}"
                            for g in ('gate_nom', 'gate_cal', 'gate_cal_infl'))
            maes = ' '.join(f"{st[('correct', r)]['mae']:5.1f}" for r in
                            ('R0', 'gate_nom', 'gate_cal', 'gate_cal_infl', 'est_nom', 'est_cal', 'safe', 'R2'))
            ps = ' '.join(str(st[('p*', g)]) for g in ('gate_nom', 'gate_cal', 'gate_cal_infl'))
            L.append(f'   {truth:13s} | {accs} | {maes} | {ps}')
            L.append(f"   {'':13s}   default_shrink accept nom/cal/infl "
                     + ' '.join(f"{st[('default_shrink', g)]['acc']:.2f}" for g in ('gate_nom', 'gate_cal', 'gate_cal_infl'))
                     + f" | sign_flip MAE gate_cal {st[('sign_flip', 'gate_cal')]['mae']:.1f} "
                     f"infl {st[('sign_flip', 'gate_cal_infl')]['mae']:.1f} R0 {st[('sign_flip', 'R0')]['mae']:.1f}")
        L.append('')
    text = '\n'.join(L)
    print(text)
    Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
