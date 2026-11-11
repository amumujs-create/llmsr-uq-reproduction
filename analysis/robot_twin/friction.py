"""Low-speed joint friction regime twin (design draft, 2026-11-12).

Task: predict a joint's friction torque at constant positive speed v (rad/s).
Training window = higher speeds; target = lower speeds (extrapolation downward).

Base world: Coulomb + viscous, tau = Fc + Fv v (a straight line for v > 0).
Regime worlds add a low-speed effect that only appears below a transition speed:
  - 'stribeck' (UP):   tau rises toward static friction Fs as v -> 0
                       tau = Fc + (Fs - Fc) exp(-(v/vs)^2) + Fv v
  - 'rolloff'  (DOWN): tau falls toward 0 as v -> 0 (smooth sigmoid, Panda-type)
                       tau = Fc * (2 sigmoid(v/vt) - 1)-shaped roll-off + Fv v
Both directions are kept on purpose: the location (where it changes) and the
direction (up or down) can then be scored separately, as in the CCPP twin.

Grounding (see docs/research/robot-twin-friction-design.md):
  * PANDA_SIGMOID: identified Franka Panda joint friction, Gaz et al., IEEE RA-L 2019,
    values from marcocognetti/FrankaEmikaPandaDynModel (GPL-3.0) commit 24d85ce,
    cpp/.../include/franka_model.h. Only the numbers are used; no code is copied.
  * Stribeck values (Fs/Fc, vs) are TWIN ASSUMPTIONS: no numeric source retrieved yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# phi1 [Nm], phi2 [s/rad], phi3 [rad/s] per Panda joint (Gaz et al. 2019).
PANDA_SIGMOID = (
    (0.54615, 5.1181, 0.039533),
    (0.87224, 9.0657, 0.025882),
    (0.64068, 10.136, -0.04607),
    (1.2794, 5.5903, 0.036194),
    (0.83904, 8.3469, 0.026226),
    (0.30301, 17.133, -0.021047),
    (0.56489, 10.336, 0.0035526),
)


def panda_friction(v, phi1: float, phi2: float, phi3: float) -> np.ndarray:
    """Gaz et al. eq. (25): zero at v = 0, saturates at about phi1 / (1 + e^{-phi2 phi3}) offset."""
    v = np.asarray(v, dtype=float)
    return phi1 / (1 + np.exp(-phi2 * (v + phi3))) - phi1 / (1 + np.exp(-phi2 * phi3))


@dataclass(frozen=True)
class FrictionCell:
    name: str
    kind: str          # 'stribeck' (up), 'rolloff' (down), 'none' (control)
    fc: float          # Coulomb level reached at high speed [Nm]
    fv: float          # viscous slope [Nm s/rad]
    fs: float = 0.0    # static friction (stribeck only) [Nm]
    vs: float = 0.0    # transition speed [rad/s]
    panda: tuple | None = None   # (phi1, phi2, phi3) for literature roll-off cells

    def base(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return self.fc + self.fv * v

    def torque(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if self.kind == 'none':
            return self.base(v)
        if self.kind == 'stribeck':
            return self.fc + (self.fs - self.fc) * np.exp(-(v / self.vs) ** 2) + self.fv * v
        if self.kind == 'rolloff':
            if self.panda is not None:
                return panda_friction(v, *self.panda) + self.fv * v
            return self.fc * np.tanh(v / self.vs) + self.fv * v
        raise ValueError(self.kind)

    def transition(self, rel: float = 0.05, grid=None) -> float:
        """Highest speed where |regime - base| >= rel * fc (the 'location' k)."""
        grid = np.geomspace(1e-4, 5.0, 4000) if grid is None else grid
        gap = np.abs(self.torque(grid) - self.base(grid)) >= rel * self.fc
        return float(grid[gap].max()) if gap.any() else 0.0


def panda_cells(fv: float = 0.0) -> list[FrictionCell]:
    """Literature roll-off cells. fc = high-speed level of the sigmoid."""
    cells = []
    for j, (p1, p2, p3) in enumerate(PANDA_SIGMOID, 1):
        level = float(panda_friction(50.0, p1, p2, p3))
        cells.append(FrictionCell(f'panda_j{j}', 'rolloff', level, fv, panda=(p1, p2, p3)))
    return cells
