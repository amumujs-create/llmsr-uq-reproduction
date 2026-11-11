"""Design exploration for the tool-hose effective-limit twin (no LLM answers, no rule scoring).

Usage: python3 analysis/robot_twin/explore_hose.py data/robot/h2017.urdf

(i) Angle extrapolation: train on J6 angles where the hose is slack, predict the
    J6 reaction torque at larger angles. Leak / breakage as in the CCPP clean-task check.
(ii) Condition extrapolation: how far does the taut angle k move when the hose is
    pressurized, and how wrong is the 'geometry at zero pressure' default?
Thresholds and all hose parameters are design assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2017_dynamics import load_urdf  # noqa: E402
from hose import EPS_PRESSURE, K_TAUT, R_WRAP, HoseCell  # noqa: E402

POSES = {'j5_30': (0, 30, 90, 0, 30, 0), 'j5_60': (0, 30, 90, 0, 60, 0), 'j5_90': (0, 30, 90, 0, 90, 0)}
SLACKS = [1.05, 1.10, 1.20, 1.35]
PRESSURES = [0.0, 0.5, 1.0]
DELTAS = [10.0, 20.0, 40.0]   # deg between window end and k
WIDTHS = [40.0, 80.0]         # deg
LEAK_MAX, BREAK_MIN = 0.02, 0.20


def angle_windows(chain, cell, k):
    scale = abs(cell.torque(chain, [k + 30.0])[0])
    res = []
    for d in DELTAS:
        for w in WIDTHS:
            hi = k - d
            lo = hi - w
            if lo < 0:
                continue
            train = np.arange(lo, hi + 0.01, 2.0)
            test = np.arange(hi + 2.0, k + 60.0, 2.0)
            reg_tr, base_tr = cell.torque(chain, train), cell.torque(chain, train, regime=False)
            reg_te, base_te = cell.torque(chain, test), cell.torque(chain, test, regime=False)
            leak = np.max(np.abs(reg_tr - base_tr)) / scale
            f_r, f_b = np.polyfit(train, reg_tr, 1), np.polyfit(train, base_tr, 1)
            rm_r = np.sqrt(np.mean((np.polyval(f_r, test) - reg_te) ** 2))
            rm_b = np.sqrt(np.mean((np.polyval(f_b, test) - base_te) ** 2))
            res.append((leak, (rm_r - rm_b) / scale))
    return res, scale


def main(urdf):
    chain = load_urdf(urdf)
    print(f'assumptions: wrap radius {R_WRAP} m, taut stiffness {K_TAUT:.0f} N/m, '
          f'length change at full pressure {EPS_PRESSURE:+.0%}\n')
    print('taut angle k of J6 (deg, positive side) by pose / slack / pressure')
    print(f"   {'pose':6s} {'slack':>5s} " + ''.join(f'  p={p:<4}' for p in PRESSURES) + '   shift p0->p1')
    ks, shifts, rows = [], [], []
    for pname, pose in POSES.items():
        for s in SLACKS:
            kp = [HoseCell(f'{pname}_s{s}', pose, s, p).taut_angle(chain) for p in PRESSURES]
            ks += [k for k in kp if np.isfinite(k)]
            if np.isfinite(kp[0]) and np.isfinite(kp[-1]):
                shifts.append(kp[0] - kp[-1])
            rows.append((pname, s, kp))
            print(f'   {pname:6s} {s:5.2f} ' + ''.join(f'{k:8.1f}' for k in kp) + f'   {kp[0] - kp[-1]:6.1f}')
    ks = np.array(ks)
    print(f'\n   k range {ks.min():.1f}-{ks.max():.1f} deg; robot spec J6 limit +-360 deg')
    print(f'   pressure shift (k at p=0 minus k at p=1): {min(shifts):.1f}-{max(shifts):.1f} deg, '
          f'median {np.median(shifts):.1f}')
    print('   -> "geometry at zero pressure" default is off by that shift at full pressure\n')

    print(f'(i) angle extrapolation windows (leak <= {LEAK_MAX:.0%}, breakage >= {BREAK_MIN:.0%} of torque at k+30)')
    n_ok = n_all = 0
    for pname, s, kp in rows:
        for p, k in zip(PRESSURES, kp):
            if not np.isfinite(k):
                continue
            res, scale = angle_windows(chain, HoseCell('c', POSES[pname], s, p), k)
            ok = sum(1 for lk, br in res if lk <= LEAK_MAX and br >= BREAK_MIN)
            n_ok += ok; n_all += len(res)
            if p == 0.0:
                worst_leak = max((lk for lk, _ in res), default=float('nan'))
                print(f'   {pname} slack {s:.2f} p=0: k {k:5.1f}, torque@k+30 {scale:6.2f} Nm, '
                      f'pass {ok}/{len(res)}, max leak {worst_leak:.3f}')
    print(f'   all cells: {n_ok}/{n_all} windows pass')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
