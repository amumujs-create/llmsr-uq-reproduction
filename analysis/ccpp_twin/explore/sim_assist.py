"""EXPLORATION (post-hoc, not a verdict): "LLM gives structure, a helper model gives numbers".

docs/research/ccpp-twin-explore-sim-assist.md. Uses the development answers already seen (old models), new seeds
("explore_sim1"). Nothing here feeds a registered verdict.

For a cell whose S2 answers mostly say the GT generator limit is reached:
  headroom h = median over answers of 100 * (limit_mw / gt_mw_at_15c - 1)     [%; L2 uses the stated 343 MW]
  SIM@c   : boundary below 15 - h / c, c = helper-model GT sensitivity in %/degC (true twin value is about 0.64)
  DATA_p  : c taken from the window, |partial AT coefficient| / mean output * 100
  DATA_t  : c taken from the window, |simple AT slope| / mean output * 100
A position outside 2..37 degC falls back to "below 8" (as HA_fb_gt). Other sides keep the HA statement.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/sim_assist.py <Folds5x2_pp.xlsx>
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
import outside_fallback as of  # noqa: E402
import outside_regime as orr  # noqa: E402
from assemble import L2_GT_15C_MW, load  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from controls import build_external  # noqa: E402
from elicit import RANGE, _num  # noqa: E402
from grid import build_worlds, ols  # noqa: E402
from parse_score import TRUTH  # noqa: E402

SEED_PREFIX = "explore_sim1"
# cap_mild|cold (k = 6) was tried and dropped: only 15 clean test rows below 3.5 degC (ia.MIN_TEST not met).
TASKS = dict(orr.TASKS)
CAP_TASKS = ("cap_strong|cold",)
KINK = tuple(t for t, v in TASKS.items() if v[3])
NOKINK = tuple(t for t, v in TASKS.items() if not v[3])
C_GRID = (0.40, 0.50, 0.64, 0.80, 1.00)
CELLS = orr.VALIDATION + orr.DEVELOPMENT
FALLBACK_AT = of.DEFAULT_AT["below"]


def headroom(objs, level):
    """Median stated headroom to the generator limit in % of GT output at 15 degC, or None if not majority-claimed."""
    if not objs or of.claim_share(objs)["gt_limit"] <= 0.5:
        return None
    hs = []
    for o in objs:
        g = o.get("gt_generator_limit") or {}
        if g.get("reached_in_range") is not True:
            continue
        p15, lim = _num(g.get("gt_mw_at_15c")), _num(g.get("limit_mw"))
        if level == "L2":
            p15 = L2_GT_15C_MW
        if p15 is not None and lim is not None and p15 > 0:
            hs.append(100.0 * (lim / p15 - 1.0))
    return float(np.median(hs)) if hs else None


def position(h, c):
    """Boundary position from headroom h (%) and sensitivity c (%/degC); fallback when out of range or c unusable."""
    if not np.isfinite(c) or c <= 0:
        return FALLBACK_AT
    at = 15.0 - h / c
    return at if RANGE[0] < at < RANGE[1] else FALLBACK_AT


def with_below(ha_stmt, at):
    return [("below", float(at))] + [(s, a) for s, a in ha_stmt if s != "below"]


def run_task(args):
    task, sigma, x, reps, ha, fb, hr, cfg = args
    warnings.filterwarnings("ignore")
    world, side, k, _ = TASKS[task]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    fixed = {"ORACLE": TRUTH[world], "D_real": orr.D_REAL, "D_aware": orr.D_AWARE[world]}
    for c in CELLS:
        fixed[f"{c}|HA"], fixed[f"{c}|FB"] = ha[c], fb[c]
        for cg in C_GRID:
            fixed[f"{c}|SIM@{cg:.2f}"] = ha[c] if hr[c] is None else with_below(ha[c], position(hr[c], cg))
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
                    r0 = float(np.sqrt(np.mean((p0 - yt) ** 2)))
                    c_p = abs(b[1]) / ytr.mean() * 100.0
                    c_t = abs(np.polyfit(at[idx], ytr, 1)[0]) / ytr.mean() * 100.0

                    def rel(stmt):
                        p, _ = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                        return float(np.sqrt(np.mean((p - yt) ** 2))) / r0

                    rec = dict(task=task, world=world, delta=delta, w=w, n=n, sigma=sigma, rep=rep, R0_abs=r0,
                               c_p=c_p, c_t=c_t)
                    for name, stmt in fixed.items():
                        rec[name] = rel(stmt)
                    for c in CELLS:
                        for tag, cc in (("DATA_p", c_p), ("DATA_t", c_t)):
                            rec[f"{c}|{tag}"] = (rec[f"{c}|HA"] if hr[c] is None
                                                 else rel(with_below(ha[c], position(hr[c], cc))))
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE.parent.parent / "llm_statements" / "runs"))
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_sim_assist"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    ext, _, _ = build_external(s0, s2)
    raw = of.load_s2_raw(a.runs)
    ext_fb = of.with_fallback(ext, raw, ("gt_limit",))
    x = load_ccpp_inputs(a.xlsx)
    worlds, _ = build_worlds(x)
    cfg = {w["module"]: w["cfg"] for w in worlds}
    per_world = {}
    for world in {t[0] for t in TASKS.values()}:
        ha = {c: ext[world][f"{c}|HA"] for c in CELLS}
        fb = {c: ext_fb[world][f"{c}|HA"] for c in CELLS}
        hr = {c: headroom(raw.get((*c.split("|"), world), []), c.split("|")[1]) for c in CELLS}
        per_world[world] = (ha, fb, hr)
    jobs = [(t, s, x, a.reps, *per_world[TASKS[t][0]], cfg[TASKS[t][0]]) for t in TASKS for s in orr.SIGMAS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cells.csv.gz", index=False)

    rules = ["HA", "FB"] + [f"SIM@{c:.2f}" for c in C_GRID] + ["DATA_p", "DATA_t"]
    tm = df.groupby("task").mean(numeric_only=True).reindex(list(TASKS))
    harm = lambda cols, tasks: (df[df["task"].isin(tasks)][cols] > orr.HARM_REL).mean()  # noqa: E731
    pd.set_option("display.width", 250)
    L = ["EXPLORATION (post-hoc). LLM structure + helper-model numbers. relative RMSE = rule / R0. "
         f"rows={len(df)} seeds={SEED_PREFIX}", "",
         "=== reference rules: task mean ===", tm[["ORACLE", "D_real", "D_aware"]].T.round(3).to_string(), "",
         "=== stated headroom h (%) and positions (degC) by cell, cap worlds (truth: cap_strong 9, cap_mild 6) ==="]
    for world in ("cap_strong", "cap_mild"):
        ha, fb, hr = per_world[world]
        for c in CELLS:
            pos = "" if hr[c] is None else " ".join(f"{cg:.2f}:{position(hr[c], cg):5.1f}" for cg in C_GRID)
            L.append(f"{world:11s} {c:28s} h={hr[c] if hr[c] is None else round(hr[c], 2)} HA={ha[c]} FB={fb[c]} "
                     f"SIM {pos}")
    L += ["", "=== window sensitivity proxies (%/degC), cap tasks: mean and 10-90 percentile ==="]
    for t in CAP_TASKS:
        g = df[df["task"] == t]
        for col in ("c_p", "c_t"):
            L.append(f"{t:16s} {col}: mean {g[col].mean():.3f}  p10 {g[col].quantile(.1):.3f}  "
                     f"p90 {g[col].quantile(.9):.3f}")
    for label, tasks in (("cap kink tasks mean", CAP_TASKS), ("all kink tasks mean", KINK), ("all tasks mean", TASKS)):
        tab = pd.DataFrame({r: {c: tm.loc[list(tasks), f"{c}|{r}"].mean() for c in CELLS} for r in rules})
        L += ["", f"=== {label} (rows = cells) ===", tab.round(3).to_string(),
              "reference: " + "  ".join(f"{r} {tm.loc[list(tasks), r].mean():.3f}" for r in ("ORACLE", "D_real",
                                                                                               "D_aware"))]
    for t in CAP_TASKS:
        tab = pd.DataFrame({r: {c: tm.loc[t, f"{c}|{r}"] for c in CELLS} for r in rules})
        L += ["", f"=== {t} task mean ===", tab.round(3).to_string(),
              "reference: " + "  ".join(f"{r} {tm.loc[t, r]:.3f}" for r in ("ORACLE", "D_real", "D_aware"))]
    for label, tasks in (("harm share, all tasks", TASKS), ("harm share, no-kink tasks", NOKINK)):
        tab = pd.DataFrame({r: {c: harm([f"{c}|{r}"], tasks).iloc[0] for c in CELLS} for r in rules})
        L += ["", f"=== {label} (> {orr.HARM_REL}) ===", tab.round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
