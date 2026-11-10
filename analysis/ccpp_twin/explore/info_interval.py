"""EXPLORATORY (not pre-registered, 2026-11-10). Never use these numbers for verdicts.

A. Power of the information statistic I and the data rule R_cp when the twin kink is sharper, noise smaller, n larger.
B. Constraint-class extrapolation intervals with the slope bound L taken from the window and an edge-uncertainty pad.
Seeds are prefixed "explore|" so nothing here reuses the registered info-axis draws.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/info_interval.py <Folds5x2_pp.xlsx> [--reps 10] [--workers 8]
Writes analysis/ccpp_twin/runs/explore_info_interval/{cells.csv.gz, summary.txt}.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import info_axis as ia  # noqa: E402
from assemble import load  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate, with_modules  # noqa: E402
from controls import build_external  # noqa: E402
from grid import _seed, build_worlds  # noqa: E402
from parse_score import TRUTH  # noqa: E402

SIGMAS = (0.5, 1.0, 2.15, 4.3)
NS = (150, 500, 2000)
WIDTHS = (5.0, 10.0)
MAX_KNOTS = 400
LLM = {"evap": ("llama33-70b|L2", "gemma3-27b|L1"), "cap_strong": ("gpt-oss-120b|L1",), "base": ()}
SPECS = {"evap": [(24.0, "hot", d) for d in (-6, -4, -2, 0, 2, 4)],
         "cap_strong": [(9.0, "cold", d) for d in (-2, 0, 2, 4)],
         "base": [(24.0, "hot", d) for d in (-6, -2, 0, 2, 4)] + [(9.0, "cold", d) for d in (-2, 0, 2, 4)]}


def ols_cov(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    cov = (r @ r / max(len(y) - X1.shape[1], 1)) * np.linalg.pinv(X1.T @ X1)
    return b, np.sqrt(np.diag(cov)), cov


def changepoint(at, Z, y, min_seg):
    """As info_axis.changepoint, with the minimum segment size as a parameter and at most MAX_KNOTS knots."""
    n = len(y)
    _, sse_lin = ia._lstsq(np.column_stack([np.ones(n), at, Z]), y)
    if n < 2 * min_seg + 1:
        return 0.0, None, None
    knots = np.unique(np.sort(at)[min_seg:n - min_seg])
    if len(knots) > MAX_KNOTS:
        knots = knots[np.linspace(0, len(knots) - 1, MAX_KNOTS).astype(int)]
    best = (np.inf, None, None)
    for k in knots:
        X = np.column_stack([np.ones(n), np.where(at < k, at - k, 0.0), np.where(at >= k, at - k, 0.0), Z])
        b, sse = ia._lstsq(X, y)
        if sse < best[0]:
            best = (sse, float(k), b)
    sse_cp, knot, b_cp = best
    if knot is None:
        return 0.0, None, None
    I = (n * np.log(sse_lin / n) + 5 * np.log(n)) - (n * np.log(sse_cp / n) + 7 * np.log(n))
    return float(I), knot, b_cp


def run_world(args):
    module, variant, cfg, sigma, x, reps, stmts = args
    warnings.filterwarnings("ignore")
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    wname = f"{module}|{variant}|{sigma}"
    rows = []
    for k, side, delta in SPECS[module]:
        for w in WIDTHS:
            lo, hi = ia.window_bounds(k, side, float(delta), w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = ia.test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < ia.MIN_TEST:
                continue
            Xte, yt = feats[tmask], clean[tmask]
            for n in NS:
                if len(pool) < n:
                    continue
                for rep in range(reps):
                    pe = clean + np.random.default_rng(_seed(f"explore|noise|{wname}|{rep}")).normal(0, sigma, len(at))
                    idx = np.random.default_rng(_seed(f"explore|sample|{module}|{k}|{delta}|{w}|{n}|{rep}")).choice(
                        pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se, cov = ols_cov(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))  # noqa: E731
                    r0 = rmse(p0)
                    rec = dict(module=module, variant=variant, k=k, side=side, delta=delta, w=w, n=n, sigma=sigma,
                               rep=rep, R0_abs=r0)
                    preds = {}
                    for name, stmt in stmts.items():
                        preds[name], rec[f"rej|{name}"] = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center,
                                                                              Xte[:, 0], Xte[:, 3])
                        rec[name] = rmse(preds[name]) / r0
                    # A. information statistic and data rule, two minimum segment sizes
                    for tag, ms in (("5", 5), ("seg", max(20, n // 10))):
                        I, knot, b_cp = changepoint(Xtr[:, 0], Xtr[:, 1:], ytr, ms)
                        p_cp = ia.predict_cp(I, knot, b_cp, side, Xte[:, 0], Xte[:, 1:], p0)
                        rec[f"I_{tag}"] = I
                        rec[f"knot_{tag}"] = np.nan if knot is None else knot
                        rec[f"R_cp_{tag}"] = rmse(p_cp) / r0
                        for thr in (0, 10):
                            rec[f"SW_{tag}_{thr}"] = rmse(p_cp if I > thr else preds["ORACLE"]) / r0
                    # B. intervals: slope bound L from physics or from the window, with or without an edge pad
                    xe = np.column_stack([np.ones(nt), np.full(nt, edge), Xte[:, 1:]])
                    y_edge = xe @ b
                    pad = 2.0 * np.sqrt(np.einsum("ij,jk,ik->i", xe, cov, xe))
                    Ls = {"phys": abs(slope_phys), "win": abs(b[1]) + 2 * se[1],
                          "max": max(abs(slope_phys), abs(b[1]) + 2 * se[1])}
                    for lname, L in Ls.items():
                        for name, stmt in {"none": [], **stmts}.items():
                            lo_i, hi_i = ia.interval(stmt, side, edge, Xte[:, 0], y_edge, L)
                            for pname, pd_ in (("0", 0.0), ("pad", pad)):
                                l2, h2 = lo_i - pd_, hi_i + pd_
                                rec[f"cov|{lname}|{pname}|{name}"] = float(np.mean((yt >= l2) & (yt <= h2)))
                                rec[f"wid|{lname}|{pname}|{name}"] = float(np.mean(h2 - l2))
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    x = load_ccpp_inputs(a.xlsx)
    cap = build_worlds(x)[1]["strong"]
    s0, s2 = load(HERE.parent.parent / "llm_statements" / "runs")
    ext, _, _ = build_external(s0, s2)
    worlds = [("evap", "ramp3.0", with_modules(evap_cooling=True)),
              ("evap", "ramp0.5", with_modules(evap_cooling=True, evap_ramp_c=0.5)),
              ("cap_strong", "soft2.0", with_modules(cold_cap=True, gt_cap_mw=cap)),
              ("cap_strong", "soft0.5", with_modules(cold_cap=True, gt_cap_mw=cap, cap_softness_mw=0.5)),
              ("base", "base", with_modules())]
    jobs = []
    for mod, var, cfg in worlds:
        stmts = {"ORACLE": TRUTH[mod], "D_real": ext[mod]["D_real"], **{c: ext[mod][f"{c}|HA"] for c in LLM[mod]}}
        jobs += [(mod, var, cfg, s, x, a.reps, stmts) for s in SIGMAS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_world, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out = HERE.parent / "runs" / "explore_info_interval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cells.csv.gz", index=False)
    pd.set_option("display.width", 250)
    L = ["EXPLORATORY (not pre-registered). relative RMSE = rule / R0. rows=%d" % len(df)]
    kw = df[df.module != "base"]
    L += ["\n=== A1. share I_5 > 0 by variant x delta x sigma (n pooled) ===",
          kw.assign(p=kw.I_5 > 0).pivot_table(index=["variant", "delta"], columns="sigma", values="p").round(2).to_string(),
          "\n=== A1b. share I_5 > 0 by variant x delta x n (sigma pooled) ===",
          kw.assign(p=kw.I_5 > 0).pivot_table(index=["variant", "delta"], columns="n", values="p").round(2).to_string(),
          "\n=== A1c. base: share I > 0 (false positives) by sigma, I_5 / I_seg ===",
          df[df.module == "base"].groupby("sigma").apply(lambda g: pd.Series({"I_5": (g.I_5 > 0).mean(), "I_seg": (g.I_seg > 0).mean(), "I_5>10": (g.I_5 > 10).mean()})).round(3).to_string()]
    cols = ["R_cp_5", "R_cp_seg", "ORACLE", "SW_5_0", "SW_5_10", "SW_seg_0", "SW_seg_10"]
    L += ["\n=== A2. mean relative RMSE by variant x delta (all sigma, n) ===",
          kw.groupby(["variant", "delta"])[cols].mean().round(3).to_string(),
          "\n=== A2b. median ===", kw.groupby(["variant", "delta"])[cols].median().round(3).to_string(),
          "\n=== A3. R_cp when it fires: mean rel RMSE of R_cp vs ORACLE on rows with I > thr, by variant and sign(delta) ==="]
    kw = kw.assign(dsign=np.where(kw.delta < 0, "in(<0)", np.where(kw.delta == 0, "edge(0)", "out(>0)")))
    for tag in ("5", "seg"):
        for thr in (0, 10):
            g = kw[kw[f"I_{tag}"] > thr]
            t = g.groupby(["variant", "dsign"]).agg(n_rows=("ORACLE", "size"), R_cp=(f"R_cp_{tag}", "mean"),
                                                    R_cp_med=(f"R_cp_{tag}", "median"), ORACLE=("ORACLE", "mean"))
            L += [f"--- I_{tag} > {thr} ---", t.round(3).to_string()]
    L += ["\n=== A4. overall mean rel RMSE (kink worlds + base), by rule ===",
          df.groupby("module")[cols].mean().round(3).to_string(),
          "\n=== A5. sharp worlds, sigma <= 1: mean rel RMSE by delta ===",
          kw[(kw.variant.isin(["ramp0.5", "soft0.5"])) & (kw.sigma <= 1)].groupby(["variant", "delta"])[cols].mean().round(3).to_string()]
    # B
    std = df[df.variant.isin(["ramp3.0", "soft2.0", "base"])]
    for lname in ("phys", "win", "max"):
        for pname in ("0", "pad"):
            names = [c.split("|", 3)[3] for c in df.columns if c.startswith(f"cov|{lname}|{pname}|")]
            for mod in ("evap", "cap_strong", "base"):
                g = std[std.module == mod]
                nm = [n for n in names if g[f"cov|{lname}|{pname}|{n}"].notna().any()]
                t = pd.concat({"cov": g.groupby("delta")[[f"cov|{lname}|{pname}|{n}" for n in nm]].mean().set_axis(nm, axis=1),
                               "wid": g.groupby("delta")[[f"wid|{lname}|{pname}|{n}" for n in nm]].mean().set_axis(nm, axis=1)}, axis=1)
                L += [f"\n=== B. L={lname} pad={pname} | {mod} (standard twin; coverage, mean width MW) ===", t.round(2).to_string()]
    L += ["\n=== B2. L=win pad=pad: coverage by sigma and n (standard twin, kink worlds, ORACLE / none) ===",
          std[std.module != "base"].groupby(["module", "sigma", "n"])[["cov|win|pad|ORACLE", "cov|win|pad|none", "wid|win|pad|ORACLE", "wid|win|pad|none"]].mean().round(2).to_string()]
    text = "\n".join(L)
    (out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
