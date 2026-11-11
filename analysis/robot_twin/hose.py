"""Tool hose twin: effective joint limit set by a hydraulic supply hose (design draft, 2026-11-12).

A hose runs from a clamp on the forearm (link_4 frame) to the tool (link_6 frame).
Its required path length is the chord between the two points plus a helical wrap
term when J6 twists away from the mounting angle:

    l(q) = sqrt(chord(q)^2 + (r_wrap * (q6 - q6_mount))^2)

Slack hose: only a small bending reaction (the in-window 'leak').
Taut hose: tension k * softplus(l - L_eff); the joint reaction is tension * dl/dq_j.
The joint angle where the hose becomes taut is the effective limit (the regime location).

External condition: L_eff = L0 * (1 + EPS_PRESSURE * p), p in [0, 1] = fraction of working
pressure. Hydraulic hoses change length under pressure; the coefficient here is a TWIN
ASSUMPTION (a few %, sign = contraction), not a checked SAE J517 value.
All geometry, stiffness and coefficients below are assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from h2017_dynamics import Link, link_frames

ANCHOR_LINK4 = np.array([0.10, 0.0, -0.15])   # clamp on forearm, link_4 frame [m]
TOOL_LINK6 = np.array([0.07, 0.0, 0.12])      # hose port on tool, link_6 frame [m]
R_WRAP = 0.06                                 # helical wrap radius around J6 [m]
EPS_PRESSURE = -0.04                          # length change at full pressure (assumption)
K_TAUT = 2.0e4                                # axial stiffness once taut [N/m]
SOFT_M = 0.002                                # softplus width [m] (2 mm)
K_BEND = 30.0                                 # slack bending reaction [N/m of length change]


def path_length(chain: list[Link], q, q6_mount: float = 0.0) -> float:
    frames = link_frames(chain, q)
    p4, r4 = frames[3]
    p6, r6 = frames[5]
    chord = np.linalg.norm((p6 + r6 @ TOOL_LINK6) - (p4 + r4 @ ANCHOR_LINK4))
    return float(np.hypot(chord, R_WRAP * (q[5] - q6_mount)))


@dataclass(frozen=True)
class HoseCell:
    name: str
    base_pose_deg: tuple     # J1..J6 (J6 is the swept joint)
    slack: float             # L0 = slack * l(base pose)
    pressure: float = 0.0    # 0..1

    def l_eff(self, chain) -> float:
        return self.slack * path_length(chain, np.deg2rad(self.base_pose_deg)) * (1 + EPS_PRESSURE * self.pressure)

    def tension(self, l: float, l_eff: float, regime: bool = True) -> float:
        bend = K_BEND * (l - l_eff * 0.9)   # small, smooth reaction present in slack
        if not regime:
            return bend
        x = (l - l_eff) / SOFT_M
        return bend + K_TAUT * SOFT_M * np.logaddexp(0.0, x)

    def torque(self, chain, q6_deg, regime: bool = True, joint: int = 5, h: float = 1e-5) -> np.ndarray:
        """Hose reaction torque on `joint` (Nm) as J6 sweeps q6_deg."""
        le = self.l_eff(chain)
        out = []
        for a in np.atleast_1d(q6_deg):
            q = np.deg2rad(np.array(self.base_pose_deg, dtype=float))
            q[5] = np.deg2rad(a)
            l = path_length(chain, q)
            dq = np.zeros(6); dq[joint] = h
            dl = (path_length(chain, q + dq) - path_length(chain, q - dq)) / (2 * h)
            out.append(self.tension(l, le, regime) * dl)
        return np.array(out)

    def taut_angle(self, chain, side: float = 1.0, grid=np.arange(0.0, 360.5, 0.5)) -> float:
        """Smallest |J6| (deg) on one side where the required length reaches L_eff."""
        le = self.l_eff(chain)
        for a in grid:
            q = np.deg2rad(np.array(self.base_pose_deg, dtype=float))
            q[5] = np.deg2rad(side * a)
            if path_length(chain, q) >= le:
                return float(a)
        return float('inf')
