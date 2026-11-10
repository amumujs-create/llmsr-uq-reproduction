"""Twin information axis (docs/research/ccpp-twin-information-axis.md, pre-registered 2026-11-09, amended §4 2026-11-10).

Worlds and truth are fixed; only the window moves (delta = extrapolation-side window edge - kink) and the window
sample size n and noise sigma change the information I = BIC(linear) - BIC(change-point) the window carries.
Rules R0, R_cp, HA (per LLM cell), SWITCH (I > 0 -> R_cp else HA), ORACLE, M, D_real, P share windows, reps and
seeds. Report-only constraint-class intervals (IV_*) are written beside them and never enter the verdicts.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/info_axis.py <Folds5x2_pp.xlsx> [--reps 20] [--workers 8]
Writes analysis/ccpp_twin/runs/info_axis/{cells.csv, summary.json, summary.txt}. Real PE is never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_statements"))
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import _seed, build_worlds, ols  # noqa: E402
from regime_rules import GUARD_MARGIN, compose, is_flat  # noqa: E402
from assemble import load  # noqa: E402
from controls import build_external  # noqa: E402
from parse_score import TRUTH  # noqa: E402

KINKS = {"evap": (24.0, "hot"), "cap_strong": (9.0, "cold")}
DELTAS = {"evap": (-6, -4, -2, 0, 2, 4, 8), "cap_strong": (-2, 0, 2, 4, 8)}
BASE_AXES = (("base|k24", 24.0, "hot", DELTAS["evap"]), ("base|k9", 9.0, "cold", DELTAS["cap_strong"]))
SIGMAS = (2.15, 4.3, 8.6)
WIDTHS = (5.0, 10.0)
NS = (50, 150, 500)
TEST_GAP, TEST_W = 2.5, 2.5
MIN_TEST, MIN_SEGMENT = 30, 5
LAMBDA = 0.5
HARM_REL = 1.10
REF = {"sigma": 4.3, "n": 150, "w": 5.0}
VALIDATION_TAGS = ("gemma3-27b", "llama33-70b", "gpt-oss-20b")
VALIDATION = tuple(f"{t}|{lv}" for t in VALIDATION_TAGS for lv in ("L1", "L2"))
TOL = 0.05


# ----------------------------------------------------------------------------- windows (§3)
def window_specs():
    """One spec per (axis, delta): the kink k, the extrapolation side and the twin module that supplies the truth."""
    out = []
    for mod, (k, side) in KINKS.items():
        out += [dict(module=mod, axis=mod, k=k, side=side, delta=float(d)) for d in DELTAS[mod]]
    for axis, k, side, ds in BASE_AXES:
        out += [dict(module="base", axis=axis, k=k, side=side, delta=float(d)) for d in ds]
    return out


def window_bounds(k, side, delta, w):
    """delta = (extrapolation-side window edge) - k. Cold side: [k + delta, k + delta + w]; hot: [k - delta - w, k - delta]."""
    if side == "cold":
        return k + delta, k + delta + w
    return k - delta - w, k - delta


def test_band(k, side, lo, hi):
    """m = the one of {window edge, k} nearer the extrapolation side; band of width TEST_W at distance TEST_GAP from m."""
    if side == "cold":
        m = min(lo, k)
        return m - TEST_GAP - TEST_W, m - TEST_GAP          # (a, b]
    m = max(hi, k)
    return m + TEST_GAP, m + TEST_GAP + TEST_W              # [a, b)


def test_mask(at, k, side, lo, hi):
    a, b = test_band(k, side, lo, hi)
    return (at > a) & (at <= b) if side == "cold" else (at >= a) & (at < b)


# ----------------------------------------------------------------------------- information statistic (§3)
def _lstsq(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, float(((y - X @ b) ** 2).sum())


def changepoint(at, Z, y):
    """Continuous two-slope fit in AT (knot on the window's AT values, >= MIN_SEGMENT points each side) with linear
    V, AP, RH, against the linear fit. Returns (I, knot, b) with b = (intercept, slope_cold, slope_warm, *cov)."""
    at, y = np.asarray(at, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    n = len(y)
    _, sse_lin = _lstsq(np.column_stack([np.ones(n), at, Z]), y)
    if n < 2 * MIN_SEGMENT + 1:
        return 0.0, None, None
    order = np.sort(at)
    best = (np.inf, None, None)
    for k in np.unique(order[MIN_SEGMENT:n - MIN_SEGMENT]):
        X = np.column_stack([np.ones(n), np.where(at < k, at - k, 0.0), np.where(at >= k, at - k, 0.0), Z])
        b, sse = _lstsq(X, y)
        if sse < best[0]:
            best = (sse, float(k), b)
    sse_cp, knot, b_cp = best
    if knot is None or sse_cp <= 0 or sse_lin <= 0:
        return 0.0, None, None
    p_lin, p_cp = 2 + Z.shape[1], 4 + Z.shape[1]            # change-point: slope and knot in addition
    bic_lin = n * np.log(sse_lin / n) + p_lin * np.log(n)
    bic_cp = n * np.log(sse_cp / n) + p_cp * np.log(n)
    return float(bic_lin - bic_cp), knot, b_cp


def predict_cp(I, knot, b_cp, side, at_test, Z_test, p0):
    """R_cp: the change-point fit extrapolated with its extrapolation-side slope; R0 when I <= 0."""
    if I <= 0 or b_cp is None:
        return p0
    slope = b_cp[1] if side == "cold" else b_cp[2]
    return b_cp[0] + slope * (at_test - knot) + np.asarray(Z_test).reshape(len(at_test), -1) @ b_cp[3:]


def switch(I, p_cp, p_ha):
    """SWITCH: trust the data when BIC picks the change-point (I* = 0), else the statement rule."""
    return p_cp if I > 0 else p_ha


# ----------------------------------------------------------------------------- statement rule HA (§4, as risk_limits)
def predict_statement(stmt, p0, b, se, slope_phys, edge, center, at_test, rh_test):
    """Window guard (GUARD_MARGIN) + compose + blend LAMBDA, exactly as risk_limits.run. Returns (pred, rejected).
    An empty statement is R0 and is not a rejection."""
    if not stmt:
        return p0, False
    wflat = bool(is_flat([center], stmt)[0])
    z_normal, z_flat = abs(b[1] - slope_phys) / se[1], abs(b[1]) / se[1]
    z_stated, z_other = (z_flat, z_normal) if wflat else (z_normal, z_flat)
    if (z_stated - z_other) >= GUARD_MARGIN:
        return p0, True
    base_at_edge = p0 - b[1] * (at_test - edge)
    p2 = base_at_edge + compose(stmt, edge, at_test, b[1], slope_phys, 0.0, wflat, rh_test)
    return p0 + LAMBDA * (p2 - p0), False


# ----------------------------------------------------------------------------- report-only intervals (§4 amendment)
def interval(stmt, side, edge, at_test, y_edge, L):
    """Constraint class "non-increasing in AT, |slope| <= L": the output at distance D beyond the edge lies in
    hot side [y_e - L D, y_e], cold side [y_e, y_e + L D]. A statement "flat beyond x_c" on the extrapolation side
    caps D at |x_c - edge|^+ (width 0 when x_c is inside the window). No guard, no blend."""
    dist = np.abs(np.asarray(at_test, float) - edge)
    for s, a, *_ in stmt:
        if side == "hot" and s == "above":
            dist = np.minimum(dist, max(0.0, a - edge))
        elif side == "cold" and s == "below":
            dist = np.minimum(dist, max(0.0, edge - a))
    if side == "hot":
        return y_edge - L * dist, y_edge
    return y_edge, y_edge + L * dist


def interval_stats(lo, hi, y, r0):
    mid = 0.5 * (lo + hi)
    return {"cov": float(np.mean((y >= lo - 1e-9) & (y <= hi + 1e-9))), "width": float(np.mean(hi - lo)),
            "mid": float(np.sqrt(np.mean((mid - y) ** 2))) / r0}


# ----------------------------------------------------------------------------- one world (module x sigma)
def run_world(args):
    module, sigma, x, reps, ext, cfg = args
    warnings.filterwarnings("ignore")
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    wname = f"{module}|realV|{sigma}"
    truth = TRUTH[module]
    ha_cells = sorted(k[:-3] for k in ext if k.endswith("|HA"))
    iv_names = {"ORACLE": truth, "D_real": ext["D_real"], **{c: ext[f"{c}|HA"] for c in ha_cells}}
    rows = []
    for spec in [s for s in window_specs() if s["module"] == module]:
        k, side = spec["k"], spec["side"]
        for w in WIDTHS:
            lo, hi = window_bounds(k, side, spec["delta"], w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < MIN_TEST:
                continue
            Xte, yt = feats[tmask], clean[tmask]
            for n in NS:
                if len(pool) < n:
                    continue
                for rep in range(reps):
                    pe = clean + np.random.default_rng(_seed(f"noise|{wname}|{rep}")).normal(0.0, sigma, len(at))
                    idx = np.random.default_rng(_seed(f"sample|info|{spec['axis']}|{spec['delta']:g}|{w:g}|{n}|{rep}")
                                                ).choice(pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    I, knot, b_cp = changepoint(Xtr[:, 0], Xtr[:, 1:], ytr)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))  # noqa: E731
                    r0 = rmse(p0)
                    p_cp = predict_cp(I, knot, b_cp, side, Xte[:, 0], Xte[:, 1:], p0)
                    rec = dict(module=module, axis=spec["axis"], k=k, side=side, delta=spec["delta"], w=w, n=n,
                               sigma=sigma, rep=rep, n_pool=len(pool), n_test=nt, edge=edge, I=I,
                               knot=knot if knot is not None else np.nan, slope_fit=float(b[1]),
                               slope_phys=float(slope_phys), R0=r0, R_cp=rmse(p_cp))
                    p_or, rej = predict_statement(truth, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                    rec.update(ORACLE=rmse(p_or), ORACLE_rej=rej)
                    for name, stmt in ext.items():
                        p, rej = predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                        rec[name] = rmse(p)
                        if name.endswith("|HA"):
                            rec[name + "_rej"] = rej
                            rec[name[:-3] + "|SWITCH"] = rmse(switch(I, p_cp, p))
                    y_edge = p0 - b[1] * (Xte[:, 0] - edge)
                    L = abs(slope_phys)
                    for name, stmt in {"none": [], **iv_names}.items():
                        st = interval_stats(*interval(stmt, side, edge, Xte[:, 0], y_edge, L), yt, r0)
                        rec.update({f"IV|{name}|{kk}": vv for kk, vv in st.items()})
                    rows.append(rec)
    return rows


def run(x, ext_by_world, reps=20, workers=8):
    worlds, _ = build_worlds(x)
    cfg = {w["module"]: w["cfg"] for w in worlds}
    jobs = [(m, s, x, reps, ext_by_world[m], cfg[m]) for m in ("evap", "cap_strong", "base") for s in SIGMAS]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_world, jobs):
            rows += r
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summary and verdicts (§5-§6)
def rule_columns(df):
    skip = {"R0"}
    return [c for c in df.columns if c not in skip and (c in ("R_cp", "ORACLE") or c.endswith(("|HA", "|SWITCH", "|M", "|S0maj"))
                                                      or c.startswith(("P|", "D_real")))]


def relative(df):
    """Relative RMSE (rule / R0) for every rule column; keeps the key columns."""
    rel = df.copy()
    for c in rule_columns(df):
        rel[c] = df[c] / df.R0
    return rel


def ha_cells(df):
    return sorted(c[:-3] for c in df.columns if c.endswith("|HA"))


def ref(df):
    return df[(df.sigma == REF["sigma"]) & (df.n == REF["n"]) & (df.w == REF["w"])]


def verdicts(rel):
    cells = [c for c in VALIDATION if f"{c}|HA" in rel.columns]
    if len(cells) < 6:
        return {"status": f"validation cells missing: {sorted(set(VALIDATION) - set(cells))}"}
    ev, cs, base = rel[rel.axis == "evap"], rel[rel.axis == "cap_strong"], rel[rel.module == "base"]
    out = {"n_validation_cells": len(cells)}
    # T1: at the reference setting, median I > 0 for delta <= -2 and <= 0 for delta >= +2, both worlds.
    t1 = {}
    for name, g in (("evap", ref(ev)), ("cap_strong", ref(cs))):
        med = g.groupby("delta").I.median()
        t1[name] = {"I_median": med.round(2).to_dict(),
                    "ok": bool((med[med.index <= -2] > 0).all() and (med[med.index >= 2] <= 0).all())}
    out["T1"] = all(v["ok"] for v in t1.values())
    out["T1_detail"] = t1
    # T2: evap delta <= -2 (pooled): median R_cp < 1 and R_cp mean < HA mean in >= 4/6 cells.
    g = ev[ev.delta <= -2]
    t2_med = float(g.R_cp.median())
    t2_wins = sum(g.R_cp.mean() < g[f"{c}|HA"].mean() for c in cells)
    out["T2"] = bool(t2_med < 1 and t2_wins >= 4)
    out["T2_detail"] = {"R_cp_median": t2_med, "R_cp_mean": float(g.R_cp.mean()), "wins": int(t2_wins)}
    # T3: delta >= +2 (evap + cap_strong pooled): HA mean < R_cp mean in >= 4/6 cells.
    g = pd.concat([ev[ev.delta >= 2], cs[cs.delta >= 2]])
    t3_wins = sum(g[f"{c}|HA"].mean() < g.R_cp.mean() for c in cells)
    out["T3"] = bool(t3_wins >= 4)
    out["T3_detail"] = {"wins": int(t3_wins), "R_cp_mean": float(g.R_cp.mean()),
                        "HA_mean": {c: float(g[f"{c}|HA"].mean()) for c in cells}}
    # T4: all worlds and deltas: SWITCH mean < HA mean and < R_cp mean in >= 4/6 cells.
    t4_wins = sum((rel[f"{c}|SWITCH"].mean() < rel[f"{c}|HA"].mean()) and (rel[f"{c}|SWITCH"].mean() < rel.R_cp.mean())
                  for c in cells)
    out["T4"] = bool(t4_wins >= 4)
    out["T4_detail"] = {"wins": int(t4_wins), "R_cp_mean": float(rel.R_cp.mean()),
                        "per_cell": {c: {"HA": float(rel[f"{c}|HA"].mean()), "SWITCH": float(rel[f"{c}|SWITCH"].mean())}
                                     for c in cells}}
    # T5: n invariance at sigma = 4.3, w = 5 (amendment): ORACLE and HA for delta in {+4, +8}; R_cp gains with n for
    # evap delta in {-4, -2}.
    fx = rel[(rel.sigma == REF["sigma"]) & (rel.w == REF["w"])]
    far = fx[(fx.axis.isin(["evap", "cap_strong"])) & (fx.delta.isin([4, 8]))]
    near = fx[(fx.axis == "evap") & (fx.delta.isin([-4, -2]))]
    by_n = lambda g, c: g.groupby("n")[c].mean()  # noqa: E731
    o = by_n(far, "ORACLE")
    t5_oracle = bool(abs(o[500] - o[50]) < TOL)
    t5_ha = sum(abs(by_n(far, f"{c}|HA")[500] - by_n(far, f"{c}|HA")[50]) < TOL for c in cells)
    rc = by_n(near, "R_cp")
    t5_cp = bool(rc[500] < rc[50] - TOL)
    out["T5"] = bool(t5_oracle and t5_ha >= 4 and t5_cp)
    out["T5_detail"] = {"ORACLE_by_n": o.round(4).to_dict(), "HA_wins": int(t5_ha), "R_cp_by_n_near": rc.round(4).to_dict()}
    # T6: noise at n = 150, w = 5 (amendment), evap delta <= -2.
    g = ev[(ev.delta <= -2) & (ev.n == REF["n"]) & (ev.w == REF["w"])]
    by_s = lambda c: g.groupby("sigma")[c].mean()  # noqa: E731
    rc, o = by_s("R_cp"), by_s("ORACLE")
    out["T6"] = bool(rc[8.6] > rc[2.15] + TOL and abs(o[8.6] - o[2.15]) < TOL)
    out["T6_detail"] = {"R_cp_by_sigma": rc.round(4).to_dict(), "ORACLE_by_sigma": o.round(4).to_dict()}
    # T7: evap ORACLE rejection rate (pooled over sigma, n, w): mean over {-2, 0} > at +8 and > at -6.
    rr = ev.groupby("delta").ORACLE_rej.mean()
    edge_rate = float(rr[[-2, 0]].mean())
    out["T7"] = bool(edge_rate > rr[8] and edge_rate > rr[-6])
    out["T7_detail"] = {"rejection_by_delta": rr.round(4).to_dict(), "edge_mean": edge_rate}
    # T8: base (pooled): HA harm rate <= 0.10 in >= 4/6 cells.
    harm = {c: float((base[f"{c}|HA"] > HARM_REL).mean()) for c in cells}
    t8_wins = sum(h <= 0.10 for h in harm.values())
    out["T8"] = bool(t8_wins >= 4)
    out["T8_detail"] = {"wins": int(t8_wins), "harm": harm}
    out["base_false_positive_I"] = float((base.I > 0).mean())
    out["base_R_cp_harm"] = float((base.R_cp > HARM_REL).mean())
    return out


def _mm(g, cols):
    """mean and median of relative RMSE per column, as one row."""
    return pd.concat([g[cols].mean().add_suffix("_mean"), g[cols].median().add_suffix("_median")])


def summarize(df):
    rel = relative(df)
    cells = ha_cells(df)
    val = [c for c in VALIDATION if c in cells]
    dev = [c for c in cells if c not in val]
    p_cols = [c for c in df.columns if c.startswith("P|")]
    rel["P_mean"] = rel[p_cols].mean(axis=1)
    core = ["R_cp", "ORACLE", "D_real", "P_mean"]
    ha_val = [f"{c}|HA" for c in val]
    sw_val = [f"{c}|SWITCH" for c in val]
    lines, out = [], {}
    lines.append("twin information axis (ccpp-twin-information-axis.md); relative RMSE = rule / R0; validation cells: "
                 + ", ".join(val) + "; dev cells: " + ", ".join(dev))
    lines.append(f"rows={len(df)}  windows per (axis,delta,w,n): {df.groupby(['axis','delta','w','n']).size().min()}-"
                 f"{df.groupby(['axis','delta','w','n']).size().max()} (reps x sigma)")

    # 1. I along delta
    lines.append("\n=== I: median at the reference setting (sigma 4.3, n 150, w 5) and share I > 0 (pooled), by axis x delta ===")
    t = pd.DataFrame({"I_median_ref": ref(rel).groupby(["axis", "delta"]).I.median(),
                      "I_pos_share_pooled": rel.groupby(["axis", "delta"]).I.apply(lambda s: (s > 0).mean()),
                      "n_rows": rel.groupby(["axis", "delta"]).size()})
    lines.append(t.round(3).to_string())
    out["I_by_axis_delta"] = {f"{a}|{d:g}": v for (a, d), v in t.round(4).to_dict("index").items()}

    # 2. rules along delta (pooled over sigma, n, w)
    lines.append("\n=== relative RMSE by axis x delta, pooled over sigma, n, w: mean / median ===")
    t = rel.groupby(["axis", "delta"]).apply(lambda g: _mm(g, core + ha_val + sw_val))
    lines.append(t.round(3).T.to_string())
    out["rules_by_axis_delta"] = {f"{a}|{d:g}": v for (a, d), v in t.round(4).to_dict("index").items()}
    lines.append("\n=== dev cells: HA and SWITCH mean relative RMSE by axis x delta ===")
    t2 = rel.groupby(["axis", "delta"])[[f"{c}|HA" for c in dev] + [f"{c}|SWITCH" for c in dev]].mean()
    lines.append(t2.round(3).T.to_string())
    out["dev_by_axis_delta"] = {f"{a}|{d:g}": v for (a, d), v in t2.round(4).to_dict("index").items()}

    # 3. harm rates
    lines.append(f"\n=== harm rate (relative RMSE > {HARM_REL}) by axis, pooled ===")
    t = rel.groupby("axis")[core[:3] + ha_val + sw_val].apply(lambda g: (g > HARM_REL).mean())
    lines.append(t.round(3).T.to_string())
    out["harm_by_axis"] = t.round(4).to_dict("index")

    # 4. ORACLE rejection by delta
    lines.append("\n=== ORACLE guard rejection rate by axis x delta (pooled) and HA rejection (validation cells) ===")
    t = rel.groupby(["axis", "delta"])[["ORACLE_rej"] + [f"{c}|HA_rej" for c in val]].mean()
    lines.append(t.round(3).to_string())
    out["rejection_by_axis_delta"] = {f"{a}|{d:g}": v for (a, d), v in t.round(4).to_dict("index").items()}

    # 5. n axis (sigma 4.3, w 5) and sigma axis (n 150, w 5)
    fx = rel[(rel.sigma == REF["sigma"]) & (rel.w == REF["w"])]
    lines.append("\n=== n axis (sigma 4.3, w 5): mean relative RMSE by axis x delta x n ===")
    t = fx.groupby(["axis", "delta", "n"])[core + ha_val].mean()
    lines.append(t.round(3).to_string())
    out["n_axis"] = {f"{a}|{d:g}|{n}": v for (a, d, n), v in t.round(4).to_dict("index").items()}
    fx = rel[(rel.n == REF["n"]) & (rel.w == REF["w"])]
    lines.append("\n=== sigma axis (n 150, w 5): mean relative RMSE and I median by axis x delta x sigma ===")
    t = fx.groupby(["axis", "delta", "sigma"])[core + ha_val].mean()
    t["I_median"] = fx.groupby(["axis", "delta", "sigma"]).I.median()
    lines.append(t.round(3).to_string())
    out["sigma_axis"] = {f"{a}|{d:g}|{s}": v for (a, d, s), v in t.round(4).to_dict("index").items()}

    # 6. I-quantile curve of R_cp - HA (description only; T4 failure reading)
    lines.append("\n=== R_cp - HA (mean relative RMSE) by I decile, kink worlds pooled (description only) ===")
    kw = rel[rel.module != "base"].copy()
    kw["I_decile"] = pd.qcut(kw.I.rank(method="first"), 10, labels=False)
    t = kw.groupby("I_decile").apply(lambda g: pd.Series({"I_lo": g.I.min(), "I_hi": g.I.max(), "R_cp": g.R_cp.mean(),
                                                          **{c: g.R_cp.mean() - g[f"{c}|HA"].mean() for c in val}}))
    lines.append(t.round(3).to_string())
    out["I_decile_curve"] = t.round(4).to_dict("index")

    # 7. report-only intervals
    iv_names = sorted({c.split("|")[1] for c in df.columns if c.startswith("IV|")})
    lines.append("\n=== report-only constraint-class intervals (no guard, no blend): coverage / mean width MW / midpoint rel RMSE, "
                 "by axis x delta (pooled) ===")
    for stat in ("cov", "width", "mid"):
        t = rel.groupby(["axis", "delta"])[[f"IV|{nm}|{stat}" for nm in iv_names]].mean()
        t.columns = iv_names
        lines.append(f"--- {stat} ---")
        lines.append(t.round(3).to_string())
        out[f"IV_{stat}"] = {f"{a}|{d:g}": v for (a, d), v in t.round(4).to_dict("index").items()}

    v = verdicts(rel)
    out["verdicts"] = v
    lines.append("\n=== verdicts T1-T8 (validation cells; see registration §6 and the 2026-11-10 axis fixing) ===")
    lines.append(json.dumps(v, indent=1, default=str))
    return out, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE.parent / "llm_statements" / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs" / "info_axis"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    ext, meta, _ = build_external(s0, s2)
    x = load_ccpp_inputs(a.xlsx)
    df = run(x, ext, reps=a.reps, workers=a.workers)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cells.csv", index=False)
    out, text = summarize(df)
    out["cells"] = meta
    json.dump(out, open(out_dir / "summary.json", "w"), indent=1, default=str)
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
