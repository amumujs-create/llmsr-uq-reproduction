"""Design exploration for the low-speed friction twin (no LLM answers, no rule scoring).

Usage: python3 analysis/robot_twin/explore_friction.py

Noise-free twin values only. Questions:
  1. Where is the transition k (location), and how much does it spread across cells
     (so that one textbook default cannot be right)?
  2. Clean & important windows: training at higher speeds shows no regime trace
     (leak) but a straight-line extrapolation to lower speeds breaks (breakage)?
  3. Direction: does the regime push torque UP (Stribeck) or DOWN (roll-off)?
Thresholds are design choices, to be fixed at registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from friction import FrictionCell, panda_cells  # noqa: E402

V_MIN, V_MAX = 0.002, 2.0          # rad/s; H2017 URDF joint speed limits are 1.40-3.93 rad/s
DELTAS = [0.5, 1.0, 3.0]           # window low end = k * (1 + delta)
WIDTHS = [3.0, 6.0]                # window high end = low end * width
LEAK_MAX = 0.02                    # fraction of fc
BREAK_MIN = 0.20                   # extra test RMSE vs base world, fraction of fc
FV_ASSUMED = 0.2                   # Nm s/rad per 1 Nm of Coulomb level (assumption)


def assumed_stribeck_cells():
    cells = []
    for vs in (0.01, 0.03, 0.1, 0.3):
        for ratio in (1.3, 1.6):
            cells.append(FrictionCell(f'strib_vs{vs}_r{ratio}', 'stribeck', 1.0, FV_ASSUMED,
                                      fs=ratio, vs=vs))
    return cells


def window_check(cell, k, delta, width, grid=np.geomspace(V_MIN, V_MAX, 400)):
    lo = k * (1 + delta)
    hi = lo * width
    if hi > V_MAX or k <= 0:
        return None
    train = grid[(grid >= lo) & (grid <= hi)]
    test = grid[grid < lo]
    if len(train) < 5 or len(test) < 5:
        return None
    leak = np.max(np.abs(cell.torque(train) - cell.base(train))) / cell.fc
    fit_reg = np.polyfit(train, cell.torque(train), 1)
    fit_base = np.polyfit(train, cell.base(train), 1)
    rmse_reg = np.sqrt(np.mean((np.polyval(fit_reg, test) - cell.torque(test)) ** 2))
    rmse_base = np.sqrt(np.mean((np.polyval(fit_base, test) - cell.base(test)) ** 2))
    return leak, (rmse_reg - rmse_base) / cell.fc, rmse_reg / cell.fc, len(test)


def report(title, cells):
    print(title)
    print(f"   {'cell':22s} {'dir':5s} {'fc[Nm]':>7s} {'k[rad/s]':>9s} {'tau(0.005)/fc':>14s}  windows pass")
    ks = []
    n_ok_all = n_all_all = 0
    for c in cells:
        k = c.transition()
        ks.append(k)
        direction = 'UP' if c.torque(0.005) > c.base(0.005) else 'DOWN'
        n_ok = n_all = 0
        for d in DELTAS:
            for w in WIDTHS:
                r = window_check(c, k, d, w)
                if r is None:
                    continue
                n_all += 1
                if r[0] <= LEAK_MAX and r[1] >= BREAK_MIN:
                    n_ok += 1
        n_ok_all += n_ok; n_all_all += n_all
        print(f'   {c.name:22s} {direction:5s} {c.fc:7.3f} {k:9.3f} {c.torque(0.005) / c.fc:14.2f}  {n_ok}/{n_all}')
    ks = np.array(ks)
    logk = np.log10(ks)
    best = np.median(logk)
    print(f'   k range {ks.min():.3f}-{ks.max():.3f} rad/s ({logk.max() - logk.min():.2f} decades); '
          f'best single constant {10**best:.3f}: mean |log10 error| {np.mean(np.abs(logk - best)):.2f}')
    print(f'   clean & important windows: {n_ok_all}/{n_all_all}\n')
    return ks


def main():
    print(f'speed range {V_MIN}-{V_MAX} rad/s; leak <= {LEAK_MAX:.0%} of fc, breakage >= {BREAK_MIN:.0%} of fc')
    print('k = highest speed where |regime - base| >= 5% of fc\n')
    report('A) literature roll-off (DOWN): Franka Panda joints, Gaz et al. 2019 sigmoid', panda_cells())
    report(f'B) assumed Stribeck (UP): fc = 1, fv = {FV_ASSUMED}, Fs/Fc and vs are assumptions',
           assumed_stribeck_cells())
    ctrl = FrictionCell('control_none', 'none', 1.0, FV_ASSUMED)
    r = window_check(ctrl, 0.05, 1.0, 6.0)
    print(f'C) control without regime (window 0.1-0.6 rad/s): extrapolation RMSE {r[2]:.3f} fc (should be 0)')


if __name__ == '__main__':
    main()
