"""CCPP twin simulator v0: combined-cycle physics on CCPP-style inputs.

Gas turbine (Brayton) + heat recovery + steam turbine, with optional regime
modules (cold-end gas-turbine power cap, hot-end inlet evaporative cooling).
Inputs are AT (degC), AP (mbar), RH (%), V (cm Hg) arrays; the real CCPP
inputs can be used as-is so the covariate geometry matches the data.
See docs/research/ccpp-twin-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

R_DRY = 287.05  # J/(kg K)


@dataclass(frozen=True)
class TwinConfig:
    pressure_ratio: float = 15.0
    gamma: float = 1.4
    eta_compressor: float = 0.88
    eta_turbine: float = 0.90
    turbine_inlet_k: float = 1600.0
    cp_air: float = 1005.0
    cp_gas: float = 1150.0
    lhv: float = 50.0e6
    eta_burner: float = 0.99
    eta_generator: float = 0.985
    intake_flow_m3s: float = 541.0
    stack_k: float = 360.0
    eta_steam_ref: float = 0.355
    vacuum_ref_cmhg: float = 40.0
    steam_eta_per_cmhg: float = 5.0e-4
    aux_mw: float = 5.0
    # Regime modules (hypotheses for the flat ends of the real output curve).
    cold_cap: bool = False
    gt_cap_mw: float = 359.0
    cap_softness_mw: float = 2.0
    evap_cooling: bool = False
    evap_effectiveness: float = 0.85
    evap_on_c: float = 24.0
    evap_ramp_c: float = 3.0
    noise_sd_mw: float = 0.0


def saturation_pressure_pa(t_c):
    """Tetens-type saturation vapour pressure over water (Pa)."""
    t_c = np.asarray(t_c, dtype=np.float64)
    return 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))


def wet_bulb_c(t_c, rh):
    """Stull (2011) wet-bulb approximation; t in degC, rh in percent."""
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    return (t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t_c + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def _softmin(a, b, softness):
    return -softness * np.logaddexp(-a / softness, -b / softness)


def simulate(at_c, ap_mbar, rh_pct, v_cmhg, config: TwinConfig = TwinConfig(), *, seed=None) -> dict:
    """Return net power PE (MW) and its components for each input row."""
    at_c, ap_mbar, rh_pct, v_cmhg = (np.asarray(a, dtype=np.float64) for a in (at_c, ap_mbar, rh_pct, v_cmhg))
    c = config
    t_intake_c = at_c.copy()
    cooling_k = np.zeros_like(at_c)
    if c.evap_cooling:
        ramp = 1.0 / (1.0 + np.exp(-(at_c - c.evap_on_c) / c.evap_ramp_c))
        cooling_k = ramp * c.evap_effectiveness * np.clip(at_c - wet_bulb_c(at_c, rh_pct), 0.0, None)
        t_intake_c = at_c - cooling_k
        rh_pct = np.full_like(rh_pct, 95.0) * ramp + rh_pct * (1.0 - ramp)
    t1 = t_intake_c + 273.15
    p = ap_mbar * 100.0
    pv = np.clip(rh_pct, 0.0, 100.0) / 100.0 * saturation_pressure_pa(t_intake_c)
    w = 0.622 * pv / (p - pv)
    rho = p / (R_DRY * t1) * (1.0 + w) / (1.0 + 1.608 * w)
    m_air = rho * c.intake_flow_m3s

    kappa = (c.gamma - 1.0) / c.gamma
    t2 = t1 * (1.0 + (c.pressure_ratio ** kappa - 1.0) / c.eta_compressor)
    w_comp = m_air * c.cp_air * (t2 - t1)
    t3 = c.turbine_inlet_k
    fuel_air = c.cp_gas * (t3 - t2) / (c.eta_burner * c.lhv - c.cp_gas * t3)
    m_gas = m_air * (1.0 + fuel_air)
    t4 = t3 * (1.0 - c.eta_turbine * (1.0 - c.pressure_ratio ** -kappa))
    w_turb = m_gas * c.cp_gas * (t3 - t4)
    p_gt = c.eta_generator * (w_turb - w_comp) / 1e6
    p_gt_uncapped = p_gt.copy()
    if c.cold_cap:
        p_gt = _softmin(p_gt, np.full_like(p_gt, c.gt_cap_mw), c.cap_softness_mw)
    # Gas flow scales down with the cap (reduced firing/IGV), so recovered heat does too.
    flow_scale = p_gt / p_gt_uncapped
    q_recovered = flow_scale * m_gas * c.cp_gas * np.clip(t4 - c.stack_k, 0.0, None) / 1e6
    eta_st = c.eta_steam_ref - c.steam_eta_per_cmhg * (v_cmhg - c.vacuum_ref_cmhg)
    p_st = c.eta_generator * eta_st * q_recovered
    pe_clean = p_gt + p_st - c.aux_mw
    noise = np.zeros_like(pe_clean)
    if c.noise_sd_mw > 0:
        noise = np.random.default_rng(seed).normal(0.0, c.noise_sd_mw, size=pe_clean.shape)
    return {"pe": pe_clean + noise, "pe_clean": pe_clean, "p_gt": p_gt, "p_st": p_st,
            "air_density": rho, "intake_cooling_k": cooling_k, "noise": noise}


def with_modules(config: TwinConfig = TwinConfig(), *, cold_cap=False, evap_cooling=False, **overrides) -> TwinConfig:
    return replace(config, cold_cap=cold_cap, evap_cooling=evap_cooling, **overrides)


def load_ccpp_inputs(path):
    """Read AT, V, AP, RH from Sheet1 of the UCI CCPP workbook (PE is not returned)."""
    import openpyxl
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]
    idx = {name: header.index(name) for name in ("AT", "V", "AP", "RH")}
    data = np.array([[float(r[idx[k]]) for k in ("AT", "V", "AP", "RH")] for r in rows if r[0] is not None])
    return {"AT": data[:, 0], "V": data[:, 1], "AP": data[:, 2], "RH": data[:, 3]}
