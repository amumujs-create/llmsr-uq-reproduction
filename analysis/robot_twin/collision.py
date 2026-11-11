"""Collision-detection false-alarm twin (design draft, 2026-11-12).

Model-based collision detection compares measured joint torque with the
controller's model; a "collision" is flagged when |residual| exceeds a
per-joint threshold. With no contact, the residual is pure model error:

  residual = [rigid(true payload) + friction_true] - [rigid(registered payload) + friction_model]

Sources of model error kept here:
  * unregistered payload dm (true = registered + dm), same mounting offset;
  * low-speed friction not in the controller model (Stribeck excess, see friction.py).
The controller's friction model is Coulomb + viscous (the base line of friction.py).

Thresholds and friction magnitudes are TWIN ASSUMPTIONS (no Doosan values).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from friction import FrictionCell
from h2017_dynamics import Link, rnea, with_payload
from saturation import PAYLOAD_OFFSET, TWIN_TORQUE_LIMITS, Move, nominal_duration, quintic

# Assumptions (not Doosan specifications).
THRESHOLD_FRACTION = 0.10                       # detection threshold = 10 % of twin torque limit
FRICTION_FC_FRACTION = 0.02                     # Coulomb level = 2 % of twin torque limit
FRICTION_FV_PER_FC = 0.2                        # viscous slope per unit Coulomb [s/rad]
STRIBECK_RATIO, STRIBECK_VS = 1.5, 0.03         # Fs/Fc, transition speed [rad/s]


def joint_frictions(kind: str = 'stribeck') -> list[FrictionCell]:
    cells = []
    for j, lim in enumerate(TWIN_TORQUE_LIMITS, 1):
        fc = FRICTION_FC_FRACTION * lim
        cells.append(FrictionCell(f'j{j}', kind, fc, FRICTION_FV_PER_FC * fc,
                                  fs=STRIBECK_RATIO * fc, vs=STRIBECK_VS))
    return cells


def signed(cell: FrictionCell, v: float, true: bool) -> float:
    a = abs(v)
    if a < 1e-9:
        return 0.0
    val = cell.torque(a) if true else cell.base(a)
    return float(np.sign(v) * val)


@dataclass(frozen=True)
class Scenario:
    move: Move
    registered_kg: float
    unregistered_kg: float

    def residual_peak(self, chain: list[Link], s: float, frictions: list[FrictionCell],
                      thresholds=None, n_samples: int = 101) -> float:
        """max over time and joints of |residual_j| / threshold_j at override s (no contact)."""
        thresholds = THRESHOLD_FRACTION * TWIN_TORQUE_LIMITS if thresholds is None else thresholds
        true_chain = with_payload(chain, self.registered_kg + self.unregistered_kg, PAYLOAD_OFFSET)
        model_chain = with_payload(chain, self.registered_kg, PAYLOAD_OFFSET)
        q0, q1 = np.deg2rad(self.move.q0_deg), np.deg2rad(self.move.q1_deg)
        amp = q1 - q0
        t_total = nominal_duration(chain, self.move) / s
        peak = 0.0
        for tn in np.linspace(0, 1, n_samples):
            p, dp, ddp = quintic(np.array(tn))
            q, qd, qdd = q0 + amp * p, amp * dp / t_total, amp * ddp / t_total**2
            r = rnea(true_chain, q, qd, qdd) - rnea(model_chain, q, qd, qdd)
            for j in range(6):
                r[j] += signed(frictions[j], qd[j], True) - signed(frictions[j], qd[j], False)
            peak = max(peak, float(np.max(np.abs(r) / thresholds)))
        return peak

    def components(self, chain, s, frictions):
        """Peak normalized residual from payload-gravity, payload-inertia and friction parts separately."""
        none = [FrictionCell(f.name, 'none', f.fc, f.fv) for f in frictions]
        only_payload = self.residual_peak(chain, s, none)
        static = Scenario(self.move, self.registered_kg, self.unregistered_kg)
        gravity_only = static.residual_peak(chain, 1e-4, none)   # quasi-static
        only_friction = Scenario(self.move, self.registered_kg, 0.0).residual_peak(chain, s, frictions)
        return gravity_only, only_payload, only_friction
