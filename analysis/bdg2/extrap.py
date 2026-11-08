"""Window grid, information statistic I, and extrapolation rules for BDG2 (bdg2-regime-extrapolation.md §3-§4).

Every rule starts from the window linear fit R0. Statement rules are "flat below X with slope factor f" and reuse
regime_rules.compose with the window slope in the physics-slope slot, the window guard (GUARD_MARGIN) and blend 0.5.
R_cp is the in-window change-point fit extrapolated with its cold-side slope; I = BIC(linear) - BIC(change-point).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from regime_rules import GUARD_MARGIN, compose, is_flat  # noqa: E402

QUANTILES = (0.50, 0.65, 0.80, 0.90)
WIDTHS = (2.5, 5.0, 10.0)
DISTANCES = (2.5, 5.0, 10.0)
SHELL = 2.5
MIN_WINDOW, MIN_TEST, MIN_SEGMENT = 20, 15, 5
LAMBDA = 0.5
HARM_REL = 1.10
PLACEBO = tuple(float(x) for x in range(4, 27, 2))
D_CONVENTION = 18.3


def ols1(t, y):
    X = np.column_stack([np.ones(len(t)), t])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = max(len(y) - 2, 1)
    se = np.sqrt((r @ r / dof) * np.linalg.pinv(X.T @ X)[1, 1])
    return b, float(se), float(r @ r)


def changepoint(t, y):
    """Continuous two-slope fit with the knot on a grid inside the window.
    Returns (I, knot, b) with b = (intercept, slope_cold, slope_warm) of y = b0 + b1*(t-k)*(t<k) + b2*(t-k)*(t>=k)."""
    n = len(t)
    b_lin, _, sse_lin = ols1(t, y)
    order = np.sort(t)
    if n < 2 * MIN_SEGMENT + 1:
        return 0.0, None, None
    knots = np.unique(order[MIN_SEGMENT:n - MIN_SEGMENT])
    best = (np.inf, None, None)
    for k in knots:
        X = np.column_stack([np.ones(n), np.where(t < k, t - k, 0.0), np.where(t >= k, t - k, 0.0)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ b) ** 2).sum())
        if sse < best[0]:
            best = (sse, float(k), b)
    sse_cp, knot, b_cp = best
    if knot is None or sse_cp <= 0 or sse_lin <= 0:
        return 0.0, None, None
    bic_lin = n * np.log(sse_lin / n) + 2 * np.log(n)
    bic_cp = n * np.log(sse_cp / n) + 4 * np.log(n)   # intercept, two slopes, knot
    return float(bic_lin - bic_cp), knot, b_cp


def rows_for_world(t, y, tc=None):
    """Window/test cells for one building: list of dicts with the window fit, I and test arrays."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    out = []
    for q in QUANTILES:
        e = float(np.quantile(t, q))
        for w in WIDTHS:
            lo = e - w
            win = (t >= lo) & (t <= e)
            if win.sum() < MIN_WINDOW:
                continue
            tw, yw = t[win], y[win]
            b, se, _ = ols1(tw, yw)
            I, knot, b_cp = changepoint(tw, yw)
            for d in DISTANCES:
                te = (t < lo - d) & (t >= lo - d - SHELL)
                if te.sum() < MIN_TEST:
                    continue
                out.append(dict(q=q, w=w, d=d, e=e, lo=lo, n_win=int(win.sum()), n_test=int(te.sum()), b=b, se=se,
                                I=I, knot=knot, b_cp=b_cp, t_test=t[te], y_test=y[te],
                                delta=None if tc is None else lo - tc))
    return out


def predict_statement(cell, x_c, factor):
    """Blend-0.5 prediction for 'flat below x_c' (factor f beyond it) with the window guard; x_c None -> R0."""
    b, se, lo, e = cell["b"], cell["se"], cell["lo"], cell["e"]
    tt = cell["t_test"]
    p0 = b[0] + b[1] * tt
    if x_c is None:
        return p0
    stmt = [("below", float(x_c))]
    center = 0.5 * (lo + e)
    wflat = bool(is_flat([center], stmt)[0])
    z_flat, z_normal = abs(b[1]) / se if se > 0 else np.inf, 0.0   # physics slope = window slope
    z_stated, z_other = (z_flat, z_normal) if wflat else (z_normal, z_flat)
    if (z_stated - z_other) >= GUARD_MARGIN:
        return p0
    base_at_edge = p0 - b[1] * (tt - lo)
    p2 = base_at_edge + compose(stmt, lo, tt, b[1], b[1], factor, wflat)
    return p0 + LAMBDA * (p2 - p0)


def predict_cp(cell):
    """R_cp: in-window change-point fit extrapolated with its cold-side slope; R0 when I <= 0."""
    b, lo, tt = cell["b"], cell["lo"], cell["t_test"]
    if cell["I"] <= 0 or cell["b_cp"] is None:
        return b[0] + b[1] * tt
    k, bc = cell["knot"], cell["b_cp"]
    return bc[0] + bc[1] * (tt - k)   # all test points are below lo <= k


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def evaluate(cell, statements, tc=None, rho=None):
    """RMSE of every rule for one cell. statements: {name: (x_c or None, factor)}."""
    y = cell["y_test"]
    r0 = rmse(predict_statement(cell, None, 0.0), y)
    rec = {k: cell[k] for k in ("q", "w", "d", "n_win", "n_test", "I", "delta")}
    rec.update(R0=r0, R_cp=rmse(predict_cp(cell), y))
    if tc is not None:
        rec["ORACLE"] = rmse(predict_statement(cell, tc, 0.0), y)
        rec["ORACLE_half"] = rmse(predict_statement(cell, tc, 0.5), y)
    for name, (x_c, f) in statements.items():
        rec[name] = rmse(predict_statement(cell, x_c, f), y)
    return rec


def fixed_statements(site_default=None):
    """LLM-free statements: placebo grid, convention D, optional site default."""
    s = {f"P|{x:g}": (x, 0.0) for x in PLACEBO}
    s["D"] = (D_CONVENTION, 0.0)
    if site_default is not None:
        s["D_site"] = (site_default, 0.0)
    return s


def run(worlds, series_of, llm_statements=None):
    """worlds: DataFrame from worlds.py; series_of(meter) -> (t, y); llm_statements: {cell_name: {meter: (x_c, f)}}.
    Returns one row per (meter, q, w, d) with the RMSE of every rule."""
    llm_statements = llm_statements or {}
    rows = []
    for _, wrow in worlds.iterrows():
        t, y = series_of(wrow.meter)
        others = worlds[(worlds.site == wrow.site) & (worlds.meter != wrow.meter)].tc
        stmts = fixed_statements(float(others.median()) if len(others) else None)
        for name, per_meter in llm_statements.items():
            stmts[name] = per_meter.get(wrow.meter, (None, 0.0))
        for cell in rows_for_world(t, y, wrow.tc):
            rec = evaluate(cell, stmts, wrow.tc, wrow.rho)
            rec.update(meter=wrow.meter, site=wrow.site, tc=wrow.tc, rho=wrow.rho)
            rows.append(rec)
    return pd.DataFrame(rows)
