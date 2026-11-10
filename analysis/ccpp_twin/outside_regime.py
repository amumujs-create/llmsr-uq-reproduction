"""Outside-regime experiment (docs/research/ccpp-twin-outside-regime.md): kink outside the window (delta > 0) only.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/outside_regime.py <Folds5x2_pp.xlsx>
"""
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
import info_axis as ia  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import _seed, build_worlds, ols  # noqa: E402
from assemble import load  # noqa: E402
from controls import DOMAIN, build_external, placebo_statements  # noqa: E402
from parse_score import TRUTH  # noqa: E402

# task -> (world, side, k, has a kink on the extrapolation side)   (§3)
TASKS = {
    "evap|hot": ("evap", "hot", 24.0, True),
    "evap+cap_mild|hot": ("evap+cap_mild", "hot", 24.0, True),
    "cap_strong|cold": ("cap_strong", "cold", 9.0, True),
    "base|hot": ("base", "hot", 24.0, False),
    "base|cold": ("base", "cold", 9.0, False),
    "cap_strong|hot": ("cap_strong", "hot", 24.0, False),
    "cap_mild|hot": ("cap_mild", "hot", 24.0, False),
}
KINK_TASKS = tuple(t for t, v in TASKS.items() if v[3])
NOKINK_TASKS = tuple(t for t, v in TASKS.items() if not v[3])
DELTAS = (1.0, 2.0, 4.0)
SIGMAS, WIDTHS, NS = (2.15, 4.3, 8.6), (5.0, 10.0), (50, 150, 500)
SEED_PREFIX = "outside1"
HARM_REL = 1.10
VALIDATION = ia.VALIDATION
DEVELOPMENT = tuple(f"{t}|{lv}" for t in ("gpt-oss-120b", "qwen3-30b-a3b-instruct") for lv in ("L1", "L2"))
D_REAL = DOMAIN["D_real"]
# Description-aware default (§4): the D_real number for each device the world description names.
D_AWARE = {"base": [], "cap_mild": [("below", 8.0)], "cap_strong": [("below", 8.0)], "evap": [("above", 27.0)],
           "evap+cap_mild": [("below", 8.0), ("above", 27.0)]}
NEED = {"H1": 5, "H2": 4, "H3": 4, "H4": 4, "H5": 4}
H0_ORACLE, H0_DREAL_HARM, H1_HARM, H2_REL = 0.75, 0.30, 0.10, 0.95


def side_placebos(side):
    """Fake boundaries on the extrapolation side only (P_side)."""
    word = "above" if side == "hot" else "below"
    return {name: st for name, st in placebo_statements().items() if st[0][0] == word}


def noise_seed(world, sigma, rep, prefix=SEED_PREFIX):
    return _seed(f"{prefix}|noise|{world}|{sigma}|{rep}")


def sample_seed(task, delta, w, n, rep, prefix=SEED_PREFIX):
    return _seed(f"{prefix}|sample|{task}|{delta:g}|{w:g}|{n}|{rep}")


def run_task(args):
    task, sigma, x, reps, ext, cfg, prefix = args
    warnings.filterwarnings("ignore")
    world, side, k, _ = TASKS[task]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    cells = [c for c in VALIDATION + DEVELOPMENT if f"{c}|HA" in ext]
    stmts = {"ORACLE": TRUTH[world], "D_real": D_REAL, "D_aware": D_AWARE[world],
             **{f"{c}|HA": ext[f"{c}|HA"] for c in cells}}
    fakes = side_placebos(side)
    rows = []
    for delta in DELTAS:
        for w in WIDTHS:
            lo, hi = ia.window_bounds(k, side, delta, w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = ia.test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < ia.MIN_TEST:
                raise RuntimeError(f"{task} delta={delta} w={w}: only {nt} test rows")
            Xte, yt = feats[tmask], clean[tmask]
            for n in NS:
                if len(pool) < n:
                    raise RuntimeError(f"{task} delta={delta} w={w}: pool {len(pool)} < n {n}")
                for rep in range(reps):
                    pe = clean + np.random.default_rng(noise_seed(world, sigma, rep, prefix)).normal(0.0, sigma, len(at))
                    idx = np.random.default_rng(sample_seed(task, delta, w, n, rep, prefix)).choice(pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    r0 = float(np.sqrt(np.mean((p0 - yt) ** 2)))

                    def rel(stmt):
                        p, rej = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                        return float(np.sqrt(np.mean((p - yt) ** 2))) / r0, rej

                    rec = dict(task=task, world=world, side=side, k=k, delta=delta, w=w, n=n, sigma=sigma, rep=rep,
                               n_test=nt, R0_abs=r0)
                    for name, stmt in stmts.items():
                        rec[name], rec[f"rej|{name}"] = rel(stmt)
                    rec["P_side"] = float(np.mean([rel(st)[0] for st in fakes.values()]))
                    rows.append(rec)
    return rows


def run(x, ext_by_world, reps=20, workers=8, prefix=SEED_PREFIX):
    worlds, _ = build_worlds(x)
    cfg = {w["module"]: w["cfg"] for w in worlds}
    jobs = [(t, s, x, reps, ext_by_world[TASKS[t][0]], cfg[TASKS[t][0]], prefix) for t in TASKS for s in SIGMAS]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- aggregation and verdicts (§5-§6)
def rule_names(df):
    return [c for c in df.columns if c in ("ORACLE", "D_real", "D_aware", "P_side") or c.endswith("|HA") and
            not c.startswith("rej|")]


def task_means(df):
    """Task mean of each rule's relative RMSE (rows = tasks)."""
    return df.groupby("task")[rule_names(df)].mean().reindex(list(TASKS))


def macro(tm, tasks):
    """Equal-weight mean of the task means over the given tasks."""
    return tm.loc[list(tasks)].mean()


def harm_share(df, tasks=None):
    sub = df if tasks is None else df[df["task"].isin(tasks)]
    return (sub[rule_names(df)] > HARM_REL).mean()


def verdicts(df):
    tm = task_means(df)
    kink, allm = macro(tm, KINK_TASKS), macro(tm, TASKS)
    harm_all, harm_nokink = harm_share(df), harm_share(df, NOKINK_TASKS)
    cells = [c for c in VALIDATION if f"{c}|HA" in df.columns]
    if len(cells) != len(VALIDATION):
        raise RuntimeError(f"validation cells missing: {sorted(set(VALIDATION) - set(cells))}")
    per = {}
    for c in cells:
        h = f"{c}|HA"
        per[c] = {"kink": float(kink[h]), "all": float(allm[h]), "harm": float(harm_all[h]),
                  "H1": bool(harm_all[h] <= H1_HARM), "H2": bool(kink[h] <= H2_REL),
                  "H3": bool(allm[h] < allm["D_real"]), "H4": bool(kink[h] < kink["P_side"]),
                  "H5": bool(kink[h] < kink["D_aware"])}
    out = {"H0": {"oracle_kink": float(kink["ORACLE"]), "dreal_nokink_harm": float(harm_nokink["D_real"]),
                  "pass": bool(kink["ORACLE"] < H0_ORACLE and harm_nokink["D_real"] >= H0_DREAL_HARM)}}
    for h, need in NEED.items():
        n_ok = sum(per[c][h] for c in cells)
        out[h] = {"cells": n_ok, "need": need, "pass": bool(n_ok >= need)}
    out["stage1"] = bool(all(out[h]["pass"] for h in ("H0", "H1", "H2", "H3", "H4")))
    out["strong"] = bool(out["stage1"] and out["H5"]["pass"])
    out["per_cell"] = per
    out["reference"] = {r: {"kink": float(kink[r]), "all": float(allm[r]), "harm_all": float(harm_all[r]),
                            "harm_nokink": float(harm_nokink[r])} for r in ("ORACLE", "D_real", "D_aware", "P_side")}
    return out


def summarize(df, ext_by_world):
    v = verdicts(df)
    names = rule_names(df)
    tm = task_means(df)
    pd.set_option("display.width", 250)
    L = [f"Outside-regime run (delta > 0 only). relative RMSE = rule / R0. rows={len(df)}", "",
         "=== verdicts ==="]
    L.append(f"H0 premise: ORACLE kink mean {v['H0']['oracle_kink']:.3f} (< {H0_ORACLE}), D_real no-kink harm "
             f"{v['H0']['dreal_nokink_harm']:.3f} (>= {H0_DREAL_HARM}) -> {'PASS' if v['H0']['pass'] else 'FAIL'}")
    for h in NEED:
        L.append(f"{h}: {v[h]['cells']}/6 cells (need {v[h]['need']}) -> {'PASS' if v[h]['pass'] else 'FAIL'}")
    L.append(f"stage 1 (H0-H4): {'HOLDS' if v['stage1'] else 'DOES NOT HOLD'}; strong (+H5): "
             f"{'HOLDS' if v['strong'] else 'DOES NOT HOLD'}")
    L += ["", "=== per validation cell ===", pd.DataFrame(v["per_cell"]).T.round(3).to_string(),
          "", "=== reference rules ===", pd.DataFrame(v["reference"]).T.round(3).to_string(),
          "", "=== task mean (rows = rules) ===", tm.T.round(3).to_string(),
          "", "=== task median ===", df.groupby("task")[names].median().reindex(list(TASKS)).T.round(3).to_string(),
          "", "=== harm share (> 1.10) by task ===",
          (df[names] > HARM_REL).groupby(df["task"]).mean().reindex(list(TASKS)).T.round(3).to_string(),
          "", "=== macro means: kink tasks / all tasks ===",
          pd.DataFrame({"kink": macro(tm, KINK_TASKS), "all": macro(tm, TASKS)}).round(3).to_string(),
          "", "=== gap closed (1 - rule) / (1 - ORACLE), kink tasks, from task means ===",
          ((1 - tm.loc[list(KINK_TASKS)]).div(1 - tm.loc[list(KINK_TASKS), "ORACLE"], axis=0)).T.round(2).to_string(),
          "", "=== task mean by delta ===",
          df.groupby(["task", "delta"])[names].mean().round(3).T.to_string(),
          "", "=== absolute RMSE (MW), task mean ==="]
    ab = df[names].mul(df["R0_abs"], axis=0)
    ab["R0"] = df["R0_abs"]
    L.append(ab.groupby(df["task"]).mean().reindex(list(TASKS)).T.round(2).to_string())
    L += ["", "=== guard rejection share by task ===",
          df.groupby("task")[[c for c in df.columns if c.startswith("rej|")]].mean().reindex(list(TASKS)).T.round(2)
          .to_string(), "", "=== assembled statements (HA) by world ==="]
    for world in sorted({t[0] for t in TASKS.values()}):
        for c in VALIDATION + DEVELOPMENT:
            L.append(f"{world:15s} {c:28s} {ext_by_world[world].get(f'{c}|HA')}")
    return v, "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE.parent / "llm_statements" / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs" / "outside_regime"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    ext, meta, _ = build_external(s0, s2)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run(load_ccpp_inputs(a.xlsx), ext, reps=a.reps, workers=a.workers)
    df.to_csv(out_dir / "cells.csv.gz", index=False)
    v, text = summarize(df, ext)
    json.dump({"verdicts": v, "cells": meta}, open(out_dir / "summary.json", "w"), indent=1, default=str)
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
