"""EXPLORATION (post-hoc, not a verdict): worst case of the statement rule, and kink-capable data-only baselines.

docs/research/ccpp-twin-explore-robust-bound.md. No LLM answers are read. New seeds ("explore_robust1"). Tasks, windows,
test bands and the grid are those of the outside-regime confirmation (outside_confirm.TASKS).

Part 1, worst case. Every single-boundary statement (side in below/above, position 2..37 degC in 1 degC steps) is
pushed through ia.predict_statement (window guard + blend 0.5). The un-blended prediction p2 is recovered from the
blended one, so other blend weights lam can be scored without touching info_axis:
    pred(lam) = p0 + lam * (p2 - p0)
  ADV_fixed : the single statement with the worst task-mean relative RMSE (an adversary who cannot see the noise)
  ADV_row   : the worst statement chosen separately in every row (an adversary who sees everything)

Part 2, data-only baselines that can bend (fitted on the window only):
  HINGE : OLS + one hinge in AT, knot searched over the window deciles, kept only if BIC beats the linear fit
  QUAD  : OLS + AT^2
  EDGE  : "flat beyond the window edge" with no blend (what a tree model does when extrapolating)
  EDGE_h: the same with blend 0.5 (a hedge that needs no outside knowledge)

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/robust_bound.py <Folds5x2_pp.xlsx>
"""
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
import outside_confirm as oc  # noqa: E402
import outside_regime as orr  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import ols  # noqa: E402

SEED_PREFIX = "explore_robust1"
TASKS = oc.TASKS
LAMS = (0.25, 0.5, 0.75, 1.0)
POSITIONS = tuple(float(p) for p in range(2, 38))
ADV = [(s, p) for s in ("below", "above") for p in POSITIONS]
KNOT_Q = np.linspace(0.1, 0.9, 9)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def hinge_fit(Xtr, ytr, Xte, side):
    """Linear fit, or linear + one hinge in AT when BIC prefers it. Returns (test prediction, hinge kept?)."""
    n = len(ytr)
    sign = 1.0 if side == "hot" else -1.0

    def design(X, c):
        cols = [np.ones(len(X)), X]
        if c is not None:
            cols.append(np.maximum(0.0, sign * (X[:, 0] - c)))
        return np.column_stack(cols)

    def fit(c):
        A = design(Xtr, c)
        b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        sse = float(((ytr - A @ b) ** 2).sum())
        return b, n * np.log(sse / n) + (A.shape[1] + (c is not None)) * np.log(n)   # the knot costs one parameter

    best_c, (best_b, best_bic) = None, fit(None)
    for c in np.quantile(Xtr[:, 0], KNOT_Q):
        b, bic = fit(c)
        if bic < best_bic:
            best_c, best_b, best_bic = c, b, bic
    return design(Xte, best_c) @ best_b, best_c is not None


def quad_fit(Xtr, ytr, Xte):
    m = Xtr[:, 0].mean()
    A = np.column_stack([np.ones(len(Xtr)), Xtr, (Xtr[:, 0] - m) ** 2])
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte, (Xte[:, 0] - m) ** 2]) @ b


def run_task(args):
    task, sigma, x, reps, cfg = args
    warnings.filterwarnings("ignore")
    world, side, k, kind = TASKS[task]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    oracle = oc.TRUTH_ALL[world]
    rows = []
    for delta in orr.DELTAS:
        for w in orr.WIDTHS:
            lo, hi = ia.window_bounds(k, side, delta, w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = ia.test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < ia.MIN_TEST:
                raise RuntimeError(f"{task} delta={delta} w={w}: only {nt} test rows")
            Xte, yt = feats[tmask], clean[tmask]
            for n in orr.NS:
                if len(pool) < n:
                    raise RuntimeError(f"{task} delta={delta} w={w}: pool {len(pool)} < n {n}")
                for rep in range(reps):
                    pe = clean + np.random.default_rng(orr.noise_seed(world, sigma, rep, SEED_PREFIX)).normal(
                        0.0, sigma, len(at))
                    idx = np.random.default_rng(orr.sample_seed(task, delta, w, n, rep, SEED_PREFIX)).choice(
                        pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    r0 = rmse(p0, yt)

                    def unblended(stmt):
                        p, rej = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                        return p0 + (p - p0) / ia.LAMBDA, rej

                    rec = dict(task=task, world=world, kind=kind, delta=delta, w=w, n=n, sigma=sigma, rep=rep,
                               R0_abs=r0, dist=float(np.abs(Xte[:, 0] - edge).mean()), b_at=float(b[1]),
                               slope_phys=float(slope_phys))
                    p2, rej = unblended(oracle)
                    for lam in LAMS:
                        rec[f"ORACLE@{lam}"] = rmse(p0 + lam * (p2 - p0), yt) / r0
                    rec["ORACLE_rej"] = rej
                    pe_edge, _ = unblended([("above" if side == "hot" else "below", float(edge))])
                    rec["EDGE"], rec["EDGE_h"] = rmse(pe_edge, yt) / r0, rmse(p0 + 0.5 * (pe_edge - p0), yt) / r0
                    ph, kept = hinge_fit(Xtr, ytr, Xte, side)
                    rec["HINGE"], rec["HINGE_kept"] = rmse(ph, yt) / r0, kept
                    rec["QUAD"] = rmse(quad_fit(Xtr, ytr, Xte), yt) / r0
                    nrej = 0
                    for s, p in ADV:
                        p2, rej = unblended([(s, p)])
                        nrej += rej
                        for lam in LAMS:
                            rec[f"A|{s}|{p:g}@{lam}"] = rmse(p0 + lam * (p2 - p0), yt) / r0
                    rec["adv_rej"] = nrej / len(ADV)
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_robust_bound"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    x = load_ccpp_inputs(a.xlsx)
    cfg = oc.world_configs(x)
    jobs = [(t, s, x, a.reps, cfg[TASKS[t][0]]) for t in TASKS for s in orr.SIGMAS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cells.csv.gz", index=False)

    pd.set_option("display.width", 250)
    order = list(TASKS)
    g = df.groupby("task")
    L = [f"EXPLORATION (post-hoc). Worst case of the statement rule and bending data-only baselines. "
         f"relative RMSE = rule / R0. rows={len(df)} seeds={SEED_PREFIX}. No LLM answers used.", ""]

    L += ["=== Part 1: worst single statement per task (ADV_fixed = worst task mean; ADV_row = worst per row) ==="]
    tab = {}
    for lam in LAMS:
        cols = [f"A|{s}|{p:g}@{lam}" for s, p in ADV]
        tm = g[cols].mean().reindex(order)
        worst = tm.idxmax(axis=1)
        rowmax = df[cols].max(axis=1)
        excess = (rowmax - 1.0).clip(lower=0) * df["R0_abs"]
        tab[lam] = pd.DataFrame({
            "ADV_fixed": tm.max(axis=1), "stmt": worst.str.replace(f"@{lam}", "", regex=False).str[2:],
            "harm_fixed": [float((df.loc[df.task == t, worst[t]] > orr.HARM_REL).mean()) for t in order],
            "ADV_row_mean": rowmax.groupby(df["task"]).mean().reindex(order),
            "ADV_row_p95": rowmax.groupby(df["task"]).quantile(0.95).reindex(order),
            "ADV_row_max": rowmax.groupby(df["task"]).max().reindex(order),
            "excess_MW_mean": excess.groupby(df["task"]).mean().reindex(order),
            "excess_MW_max": excess.groupby(df["task"]).max().reindex(order)})
        L += ["", f"--- blend lam = {lam} ---", tab[lam].round(3).to_string()]
    L += ["", "share of the 72 adversarial statements rejected by the window guard, task mean:",
          g["adv_rej"].mean().reindex(order).round(3).to_string(),
          "", "R0 absolute RMSE (MW), task mean / mean distance of the test band from the window edge (degC):",
          g[["R0_abs", "dist"]].mean().reindex(order).round(2).to_string()]

    L += ["", "=== Part 1b: gain when right (ORACLE) against worst case when wrong (ADV_fixed), by blend ==="]
    kink, nokink = list(oc.KINK), list(oc.NOKINK)
    rowsb = {}
    for lam in LAMS:
        om = g[f"ORACLE@{lam}"].mean()
        rowsb[lam] = {"ORACLE kink mean": om[kink].mean(), "ADV_fixed kink mean": tab[lam].loc[kink, "ADV_fixed"].mean(),
                      "ADV_fixed nokink mean": tab[lam].loc[nokink, "ADV_fixed"].mean(),
                      "ADV_fixed worst task": tab[lam].loc[kink + nokink, "ADV_fixed"].max(),
                      "excess MW worst task (row mean)": tab[lam].loc[kink + nokink, "excess_MW_mean"].max()}
    L.append(pd.DataFrame(rowsb).round(3).to_string())

    L += ["", "=== Part 2: data-only baselines, task mean ==="]
    base = ["ORACLE@0.5", "ORACLE@1.0", "EDGE", "EDGE_h", "HINGE", "QUAD"]
    L += [g[base].mean().reindex(order).T.round(3).to_string(),
          "", "harm share (> 1.10):", (df[base] > orr.HARM_REL).groupby(df["task"]).mean().reindex(order).T.round(3).to_string(),
          "", "share of rows where the hinge was kept:", g["HINGE_kept"].mean().reindex(order).round(3).to_string(),
          "", "macro means: " + "  ".join(
              f"{c} kink {g[c].mean()[kink].mean():.3f} / primary {g[c].mean()[kink + nokink].mean():.3f}" for c in base),
          "", "=== Part 2b: kink tasks by delta (B/C split: does a bending model recover the regime from the window?) ===",
          df[df.task.isin(kink)].groupby(["task", "delta"])[base].mean().round(3).T.to_string(),
          "", "=== Part 2c: kink tasks by n (does more data help the data-only models?) ===",
          df[df.task.isin(kink)].groupby(["task", "n"])[["HINGE", "QUAD", "HINGE_kept"]].mean().round(3).T.to_string(),
          "", "=== Part 2d: evap|hot by delta and n, HINGE ===",
          df[df.task == "evap|hot"].pivot_table(index="delta", columns="n", values="HINGE").round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
