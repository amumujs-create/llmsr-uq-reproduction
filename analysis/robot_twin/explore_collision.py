"""Design exploration for the collision false-alarm twin (no LLM answers, no rule scoring).

Usage: python3 analysis/robot_twin/explore_collision.py data/robot/h2017.urdf

Question 1 (does speed matter?): how does the peak no-contact residual y(s)
  (normalized by the detection threshold; y > 1 = false alarm) change with the
  speed override s, and which part dominates (payload gravity, payload inertia,
  low-speed friction)?
Question 2 (is there a boundary inside the operating range?): override s_fa
  where y first exceeds 1, per scenario, and its spread.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collision import (FRICTION_FC_FRACTION, STRIBECK_RATIO, STRIBECK_VS, THRESHOLD_FRACTION,
                       Scenario, joint_frictions)  # noqa: E402
from explore_saturation import MOVES  # noqa: E402
from h2017_dynamics import load_urdf  # noqa: E402

S_GRID = [0.1, 0.25, 0.5, 0.75, 1.0]
REGISTERED = 10.0
UNREGISTERED = [1.0, 3.0, 5.0]


def main(urdf):
    chain = load_urdf(urdf)
    fr = joint_frictions('stribeck')
    print(f'assumptions: threshold {THRESHOLD_FRACTION:.0%} of twin torque limit; friction Fc {FRICTION_FC_FRACTION:.0%} '
          f'of limit, Fs/Fc {STRIBECK_RATIO}, vs {STRIBECK_VS} rad/s; registered payload {REGISTERED} kg\n')
    print('1) peak residual / threshold vs speed override s  (y > 1 = false alarm)')
    print(f"   {'move':12s} {'dm':>4s} " + ''.join(f's={s:<6}' for s in S_GRID) + '  max/min over s')
    ratios = []
    rows = []
    for m in MOVES:
        for dm in UNREGISTERED:
            sc = Scenario(m, REGISTERED, dm)
            ys = np.array([sc.residual_peak(chain, s, fr) for s in S_GRID])
            ratios.append(ys.max() / ys.min())
            rows.append((m.name, dm, ys))
            print(f'   {m.name:12s} {dm:4.0f} ' + ''.join(f'{y:<8.2f}' for y in ys) + f'  {ys.max() / ys.min():.2f}')
    print(f'\n   speed sensitivity (max/min of y over s): median {np.median(ratios):.2f}, max {max(ratios):.2f}')

    alarm_all = sum(1 for _, _, ys in rows if np.all(ys > 1))
    alarm_none = sum(1 for _, _, ys in rows if np.all(ys <= 1))
    crossing = len(rows) - alarm_all - alarm_none
    print(f'2) scenarios: alarm at every speed {alarm_all}, never {alarm_none}, '
          f'crossing inside 0.1-1.0 {crossing} (of {len(rows)})\n')

    print('3) decomposition at s = 1.0 (peak residual / threshold of each part alone)')
    print(f"   {'move':12s} {'dm':>4s} {'payload gravity':>16s} {'payload total':>14s} {'friction':>9s}")
    for m in MOVES:
        g, p, f = Scenario(m, REGISTERED, 3.0).components(chain, 1.0, fr)
        print(f'   {m.name:12s} {3:4d} {g:16.2f} {p:14.2f} {f:9.2f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
