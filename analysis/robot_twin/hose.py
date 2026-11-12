"""Tool hose twin: effective joint limit set by a hydraulic supply hose (design draft, 2026-11-12).

A hose runs from a clamp on the forearm (link_4 frame) to the tool (link_6 frame).
Its required path length is the chord between the two points plus a helical wrap
term when J6 twists away from the mounting angle:

    l(q) = sqrt(chord(q)^2 + (r_wrap * (q6 - q6_mount))^2)

Slack hose (revised 2026-11-12 after the MuJoCo cable check, validate_hose_cable.py):
a short, stiff hose bent between two fixed points pushes like a buckled column.
Axial force F = -P_E * tanh(e* / E_C) with P_E = C_EULER * EI / L^2 (Euler load,
clamped-pinned), e = 1 - l/L (relative slack), e* = e - ALPHA * e0 (e0 = e at the base
pose). ALPHA and E_C are fitted to MuJoCo elastic-cable sweeps (leave-one-pose-out rms
0.11-0.15 Euler loads). The old model had only a linear 'bending' term of ~0.5 N; the cable
pushes with 17-726 N.
Taut hose: F += K_TAUT * softplus(-e* L / SOFT_M); taut where e* = 0 (the regime location,
the effective limit). Joint reaction = F * dl/dq_j (force taken along the chord; the cable's
lateral force makes the true J6 torque ~1.2-1.35x larger, not modelled).

External condition: L_eff = L0 * (1 + EPS_PRESSURE * p), p in [0, 1] = fraction of working
pressure. Hydraulic hoses change length under pressure; the coefficient here is a TWIN
ASSUMPTION (contraction end of the +2% .. -4% range cited in trade literature;
SAE J517 text not checked; the sign can differ between hoses).
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
EI_HOSE = 0.3                                 # hose bending stiffness [N m^2] (assumption; 0.1-1 checked)
C_EULER = 2.046 * np.pi ** 2                  # Euler load factor, clamped-pinned column
E_C = 0.025                                   # thrust fade-out width in relative slack (fit to cable)
ALPHA_SHIFT = 0.08                            # taut point at e = ALPHA_SHIFT * e0 (fit to cable)


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
        """Hose length L at this pressure."""
        return self.slack * path_length(chain, np.deg2rad(self.base_pose_deg)) * (1 + EPS_PRESSURE * self.pressure)

    def e0(self, chain) -> float:
        return 1.0 - path_length(chain, np.deg2rad(self.base_pose_deg)) / self.l_eff(chain)

    def l_taut(self, chain) -> float:
        """Required path length at which the hose turns taut (e* = 0)."""
        return self.l_eff(chain) * (1.0 - ALPHA_SHIFT * self.e0(chain))

    def tension(self, l: float, L: float, e0: float, regime: bool = True) -> float:
        """Axial force at the tool port: + pulls (taut), - pushes (bent hose)."""
        e_star = (1.0 - l / L) - ALPHA_SHIFT * e0
        f = -C_EULER * EI_HOSE / L ** 2 * np.tanh(e_star / E_C)   # present in slack and taut
        if not regime:
            return f
        return f + K_TAUT * SOFT_M * np.logaddexp(0.0, -e_star * L / SOFT_M)

    def torque(self, chain, q6_deg, regime: bool = True, joint: int = 5, h: float = 1e-5) -> np.ndarray:
        """Torque the joint must supply against the hose (Nm), F * dl/dq_j, as J6 sweeps q6_deg."""
        L, e0 = self.l_eff(chain), self.e0(chain)
        out = []
        for a in np.atleast_1d(q6_deg):
            q = np.deg2rad(np.array(self.base_pose_deg, dtype=float))
            q[5] = np.deg2rad(a)
            l = path_length(chain, q)
            dq = np.zeros(6); dq[joint] = h
            dl = (path_length(chain, q + dq) - path_length(chain, q - dq)) / (2 * h)
            out.append(self.tension(l, L, e0, regime) * dl)
        return np.array(out)

    def taut_angle(self, chain, side: float = 1.0, grid=np.arange(0.0, 360.5, 0.5)) -> float:
        """Smallest |J6| (deg) on one side where the required length reaches the taut length."""
        lt = self.l_taut(chain)
        for a in grid:
            q = np.deg2rad(np.array(self.base_pose_deg, dtype=float))
            q[5] = np.deg2rad(side * a)
            if path_length(chain, q) >= lt:
                return float(a)
        return float('inf')
