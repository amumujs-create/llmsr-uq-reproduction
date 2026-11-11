"""Design exploration for the H2017 saturation twin (no LLM answers, no rule scoring).

Usage: python3 analysis/robot_twin/explore_saturation.py data/robot/h2017.urdf

Questions this answers, with noise-free twin values only:
  1. Does the boundary s* move with payload and pose (so one spec-sheet default cannot be right)?
  2. Are there clean windows: no regime trace inside the training window
     (leak) but a large extrapolation error beyond it (breakage)?
Thresholds below are design choices, to be fixed at registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2017_dynamics import load_urdf, rnea, with_payload  # noqa: E402
from saturation import (PAYLOAD_OFFSET, TWIN_TORQUE_LIMITS, Move, build_cell)  # noqa: E402

MOVES = [
    Move('lift_mid', (0, 0, 90, 0, 0, 0), (0, 60, 90, 0, 0, 0)),          # J2 raise, elbow bent
    Move('lift_reach', (0, 0, 0, 0, 0, 0), (0, 70, 0, 0, 0, 0)),          # J2 raise, arm straight
    Move('elbow_out', (0, 30, 120, 0, 0, 0), (0, 30, 30, 0, 0, 0)),       # J3 extend
    Move('swing_reach', (-60, 60, 30, 0, 0, 0), (60, 60, 30, 0, 0, 0)),   # J1 swing, arm out
    Move('swing_tuck', (-60, 20, 120, 0, 0, 0), (60, 20, 120, 0, 0, 0)),  # J1 swing, arm tucked
    Move('combo', (-30, 10, 110, 0, 30, 0), (40, 60, 40, 0, 80, 0)),      # several joints
]
PAYLOADS = [0.0, 5.0, 10.0, 15.0, 20.0]
DELTAS = [0.05, 0.10, 0.20]       # gap between window end and s*, in override fraction
WIDTHS = [0.20, 0.40]
LEAK_MAX = 0.01                   # max |regime - base| inside window, fraction of base speed at window end
BREAK_MIN = 0.10                  # extra test RMSE vs base world, fraction of mean base test speed
SOFTNESS = [0.0, 0.05, 0.15]


def window_check(cell, delta, width, softness, grid=np.linspace(0.05, 1.0, 96)):
    s_star = cell.boundary
    hi = s_star - delta
    lo = hi - width
    if lo < 0.05 or s_star >= 1.0:
        return None
    train = grid[(grid >= lo) & (grid <= hi)]
    test = grid[grid > hi]
    if len(train) < 5 or len(test) < 5:
        return None
    base_tr, reg_tr = cell.speed(train, regime=False), cell.speed(train, softness)
    base_te, reg_te = cell.speed(test, regime=False), cell.speed(test, softness)
    leak = np.max(np.abs(reg_tr - base_tr)) / base_tr[-1]
    coef = np.polyfit(train, reg_tr, 1)
    rmse_reg = np.sqrt(np.mean((np.polyval(coef, test) - reg_te) ** 2))
    coef_b = np.polyfit(train, base_tr, 1)
    rmse_base = np.sqrt(np.mean((np.polyval(coef_b, test) - base_te) ** 2))
    breakage = (rmse_reg - rmse_base) / np.mean(base_te)
    return leak, breakage, rmse_reg, rmse_base, len(test)


def main(urdf):
    chain = load_urdf(urdf)
    print('twin torque limits (Nm, assumption):', TWIN_TORQUE_LIMITS)
    print('URDF velocity limits (deg/s):', np.round(np.rad2deg([l.velocity for l in chain]), 1))
    print('payload offset (m):', PAYLOAD_OFFSET, '\n')

    cells = [build_cell(chain, m, p) for m in MOVES for p in PAYLOADS]
    print('1) boundary s* (override where the torque limit binds; >1.00 = never within 0-100 %)')
    print(f"{'move':12s}" + ''.join(f'{p:>8.0f}kg' for p in PAYLOADS) + '   static max |tau|/limit @20kg')
    for m in MOVES:
        row = [c for c in cells if c.move is m]
        loaded = with_payload(chain, 20.0, PAYLOAD_OFFSET)
        stat = max(np.max(np.abs(rnea(loaded, np.deg2rad(q), np.zeros(6), np.zeros(6))) / TWIN_TORQUE_LIMITS)
                   for q in (m.q0_deg, m.q1_deg))
        print(f'{m.name:12s}' + ''.join(f'{c.boundary:10.2f}' for c in row) + f'   {stat:6.2f}')
    inside = [c.boundary for c in cells if 0 < c.boundary < 1]
    print(f'\n   cells with s* inside (0,1): {len(inside)}/{len(cells)}; '
          f'range {min(inside):.2f}-{max(inside):.2f}, median {np.median(inside):.2f}')
    spec_default = 1.0
    errs = [abs(spec_default - b) for b in inside]
    print(f'   spec-sheet default (s* = 1.00, speed limit only): mean |error| {np.mean(errs):.2f}')
    med = float(np.median(inside))
    print(f'   best single constant (median {med:.2f}): mean |error| '
          f'{np.mean([abs(med - b) for b in inside]):.2f}\n')

    print('2) clean & important windows  (leak <= %.0f%%, breakage >= %.0f%%)' % (LEAK_MAX * 100, BREAK_MIN * 100))
    for soft in SOFTNESS:
        n_ok = n_all = 0
        per_delta = {d: [0, 0] for d in DELTAS}
        examples = []
        for c in cells:
            for d in DELTAS:
                for w in WIDTHS:
                    r = window_check(c, d, w, soft)
                    if r is None:
                        continue
                    n_all += 1; per_delta[d][1] += 1
                    leak, brk, rr, rb, nt = r
                    if leak <= LEAK_MAX and brk >= BREAK_MIN:
                        n_ok += 1; per_delta[d][0] += 1
                        if len(examples) < 3:
                            examples.append(f'{c.move.name}/{c.payload_kg:.0f}kg d={d} w={w}: '
                                            f'leak {leak*100:.2f}%, extrap RMSE {rr:.1f} vs base {rb:.1f} deg/s')
        pd = ', '.join(f'd={d}: {a}/{b}' for d, (a, b) in per_delta.items())
        print(f'   softness {soft:.2f}: {n_ok}/{n_all} windows pass  ({pd})')
        for e in examples:
            print('      e.g.', e)

    print('\n3) sensitivity to the assumed torque limits (all limits scaled together)')
    from saturation import torque_min_duration, nominal_duration
    for scale in (0.75, 0.80, 0.85, 0.90, 1.00, 1.20):
        lim = TWIN_TORQUE_LIMITS * scale
        b = []
        n_inf = 0
        for m in MOVES:
            tn = nominal_duration(chain, m)
            for p in PAYLOADS:
                tt = torque_min_duration(with_payload(chain, p, PAYLOAD_OFFSET), m, lim)
                if not np.isfinite(tt):
                    n_inf += 1
                else:
                    b.append(tn / tt)
        b = np.array(b)
        ins = b[b < 1]
        rng = f'{ins.min():.2f}-{ins.max():.2f}' if len(ins) else '-'
        print(f'   x{scale:.2f} (J2 {lim[1]:.0f} Nm): infeasible(static) {n_inf:2d}, '
              f's* inside (0,1) {len(ins):2d}/{len(b)}, range {rng}')

    print('\n4) payload axis at 100 % override: payload p* where the torque limit starts to bind (0-40 kg)')
    for m in MOVES:
        tn = nominal_duration(chain, m)
        lo, hi = 0.0, 40.0
        def binds(p):
            return torque_min_duration(with_payload(chain, p, PAYLOAD_OFFSET), m) > tn
        if not binds(hi):
            print(f'   {m.name:12s} p* > 40 kg')
            continue
        if binds(lo):
            print(f'   {m.name:12s} p* = 0 (binds unloaded)')
            continue
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if binds(mid) else (mid, hi)
        print(f'   {m.name:12s} p* = {hi:5.1f} kg')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
