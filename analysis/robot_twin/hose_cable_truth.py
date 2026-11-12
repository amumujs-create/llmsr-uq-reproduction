"""MuJoCo-cable 'truth' for the pressure worlds (out-of-family check for hose gates, 2026-11-12).

Usage: python hose_cable_truth.py data/robot/h2017.urdf   (needs mujoco; tested with 3.2.3)

For each pose / slack / pressure world eps, the cable length is slack * chord0 * (1 + eps),
EI = hose.EI_HOSE, no gravity. Each quasi-static J6 sweep is stored as rows
(q6 deg, axial force N, torque the joint must supply against the hose Nm = -applied torque),
so the sign matches hose.HoseCell.torque. Output: runs/hose_cable_truth.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose  # noqa: E402
from h2017_dynamics import load_urdf  # noqa: E402
from hose_statements import EPS_WORLDS, POSES, SLACKS  # noqa: E402
from validate_hose_cable import endpoints, geom_angle, sweep  # noqa: E402


def main(urdf):
    chain = load_urdf(urdf)
    out = {}
    for pname, pose in POSES.items():
        A, P0 = endpoints(chain, pose, 0.0)
        for s in SLACKS:
            if geom_angle(chain, pose, s * np.linalg.norm(P0 - A)) > 200:
                continue
            for eps in EPS_WORLDS:
                se = s * (1 + eps)
                r = sweep(chain, pose, se, hose.EI_HOSE, False, geom_angle(chain, pose, se * np.linalg.norm(P0 - A)))
                r[:, 2] *= -1.0
                out[f'{pname}|{s}|{eps}'] = r
                print(f'{pname} slack {s} eps {eps:+.2f}: {len(r)} angles, last {r[-1, 0]:.1f} deg', flush=True)
    path = Path(__file__).resolve().parent / 'runs' / 'hose_cable_truth.npz'
    np.savez(path, **out)
    print(f'saved {len(out)} sweeps to {path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
