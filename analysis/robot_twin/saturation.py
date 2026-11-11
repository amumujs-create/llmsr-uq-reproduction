"""Speed-override saturation twin for the H2017 (design draft, 2026-11-12).

Each "cell" (analogue of one power plant) is a fixed point-to-point joint move
with a fixed payload. The input axis is the speed override s (fraction of the
nominal speed, 0.05..1.0). The output is the achieved mean speed of the move,
y = path length / actual duration (deg/s).

Base world: the controller follows the command, so y is proportional to s.
Regime world: the controller time-scales the move so that no joint exceeds its
torque limit; above the boundary s*, y stops rising (a cap, like the CCPP
output cap). s* depends on payload and pose, so no single spec-sheet number
gives it.

Torque limits are TWIN ASSUMPTIONS (see TWIN_TORQUE_LIMITS). The URDF effort
values (372 Nm for J1-J3) are below the static gravity torque of a 20 kg
payload at reach and are therefore not used as real limits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from h2017_dynamics import Link, rnea, with_payload

# Twin assumption, not a Doosan specification (unverified).
TWIN_TORQUE_LIMITS = np.array([900.0, 900.0, 450.0, 160.0, 100.0, 50.0])
PAYLOAD_OFFSET = np.array([0.0, 0.0, 0.10])   # 10 cm beyond the flange
QUINTIC_PEAK = 1.875                          # peak of d/ds (10s^3-15s^4+6s^5)


@dataclass(frozen=True)
class Move:
    name: str
    q0_deg: tuple
    q1_deg: tuple


def quintic(t_norm: np.ndarray):
    s = t_norm
    p = 10 * s**3 - 15 * s**4 + 6 * s**5
    dp = 30 * s**2 - 60 * s**3 + 30 * s**4
    ddp = 60 * s - 180 * s**2 + 120 * s**3
    return p, dp, ddp


def nominal_duration(chain: list[Link], move: Move) -> float:
    """Duration at 100 % override: the fastest move allowed by the URDF joint speed limits."""
    amp = np.abs(np.deg2rad(np.subtract(move.q1_deg, move.q0_deg)))
    vmax = np.array([l.velocity for l in chain])
    return float(np.max(QUINTIC_PEAK * amp / vmax))


def torque_min_duration(chain: list[Link], move: Move, limits=TWIN_TORQUE_LIMITS,
                        n_samples: int = 201) -> float:
    """Shortest duration T with |tau_j(t)| <= limit_j along the quintic path.

    With time scaling, tau(t) = g(q) + b(q, path)/T^2, so each sample gives a
    closed-form bound. Returns inf if gravity alone exceeds a limit.
    """
    q0, q1 = np.deg2rad(move.q0_deg), np.deg2rad(move.q1_deg)
    amp = q1 - q0
    zero = np.zeros(6)
    t2_min = 0.0
    for tn in np.linspace(0, 1, n_samples):
        p, dp, ddp = quintic(np.array(tn))
        q = q0 + amp * p
        g = rnea(chain, q, zero, zero)
        b = rnea(chain, q, amp * dp, amp * ddp) - g   # dynamic part at T = 1
        for j in range(6):
            if abs(g[j]) >= limits[j]:
                return float('inf')
            if b[j] == 0:
                continue
            room = limits[j] - g[j] * np.sign(b[j])   # same sign as b: limit - |g|; opposite: limit + |g|
            t2_min = max(t2_min, abs(b[j]) / room)
    return float(np.sqrt(t2_min))


@dataclass(frozen=True)
class Cell:
    move: Move
    payload_kg: float
    t_nom: float
    t_torque: float

    @property
    def boundary(self) -> float:
        """s* = override where the torque limit starts to bind (>1 means never within range)."""
        return self.t_nom / self.t_torque if np.isfinite(self.t_torque) else 0.0

    def path_deg(self) -> float:
        return float(np.linalg.norm(np.subtract(self.move.q1_deg, self.move.q0_deg)))

    def speed(self, s, softness: float = 0.0, regime: bool = True) -> np.ndarray:
        """Achieved mean speed (deg/s). softness > 0 blends the cap smoothly (log-sum-exp in duration)."""
        s = np.asarray(s, dtype=float)
        t_cmd = self.t_nom / s
        if not regime:
            t = t_cmd
        elif softness <= 0:
            t = np.maximum(t_cmd, self.t_torque)
        else:
            k = softness * self.t_torque
            t = k * np.logaddexp(t_cmd / k, self.t_torque / k)
        return self.path_deg() / t


def build_cell(chain: list[Link], move: Move, payload_kg: float) -> Cell:
    loaded = with_payload(chain, payload_kg, PAYLOAD_OFFSET)
    return Cell(move, payload_kg, nominal_duration(chain, move), torque_min_duration(loaded, move))
