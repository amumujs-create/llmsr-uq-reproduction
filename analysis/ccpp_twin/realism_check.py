"""Realism check of the twin's device-free world against the real CCPP output (docs/research/ccpp-twin-realism-check.md).

Registered 2026-11-12, run once. No LLM answers are read.

Question: in the two "no-kink" task geometries of clean_diag.py (hot side beyond 28 degC, cold side beyond 9 degC), does
the real plant behave like the twin's device-free base world (a "flat beyond the window edge" statement hurts) or like
a world with a device on that side (the statement helps)?

Worlds, all on the real CCPP inputs:
  real      observed PE, no noise added
  base      twin without devices                          + N(0, SIGMA) noise
  kink      twin with a sharp device on that side         + N(0, SIGMA) noise
            (hot: evaporative cooler on at 28.5 degC; cold: the clean-world cap357 generator limit, about 9.5 degC)
Every world is scored against its own observed (noisy) output, so all share a noise floor.

Rules per row: R0 (window OLS), EDGE@0.5 and EDGE@1.0 (flat from the window edge, blended 0.5 / taken in full; the
window guard applies), POS_p@1.0 (flat beyond a fixed position p, taken in full).

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/realism_check.py <Folds5x2_pp.xlsx>
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_statements"))
import clean_worlds as cw  # noqa: E402
import info_axis as ia  # noqa: E402
import outside_regime as orr  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate, with_modules  # noqa: E402
from grid import build_worlds, ols  # noqa: E402

SEED = "realism1"
SIGMA = 4.3
SIDES = {"hot": 28.0, "cold": 9.0}                # nominal k of the no-kink tasks in clean_diag.py
WORD = {"hot": "above", "cold": "below"}
DELTAS, WIDTHS, NS = orr.DELTAS, orr.WIDTHS, orr.NS
REPS = 20
POSITIONS = {"hot": (24.0, 26.0, 28.0, 30.0, 32.0), "cold": (12.0, 10.0, 8.0, 6.0)}
SEGMENTS = (8.0, 27.0)                             # report-only partial slopes: below 8, 8..27, above 27
N_CELLS = len(DELTAS) * len(WIDTHS)
CELLS_NEEDED = 5                                   # of 6 (delta, w) cells


def load_pe(path):
    import openpyxl
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]
    i = header.index("PE")
    return np.array([float(r[i]) for r in rows if r[0] is not None])


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def twin_configs(x):
    _, cap = build_worlds(x)
    return {"base": with_modules(),
            "kink|hot": with_modules(**cw.SHARP_EVAP, evap_on_c=28.5),
            "kink|cold": cw.configs(cap["strong"])["cap357"]}


def outputs(x, pe_real):
    """{(world, side): (observed output, is_twin)}; twin noise is drawn once per world and side."""
    cfg = twin_configs(x)
    sim = lambda c: simulate(x["AT"], x["AP"], x["RH"], x["V"], c)["pe_clean"]  # noqa: E731
    out = {}
    for side in SIDES:
        out[("real", side)] = pe_real
        for name, c in (("base", cfg["base"]), ("kink", cfg[f"kink|{side}"])):
            noise = np.random.default_rng(orr.noise_seed(f"{name}|{side}", SIGMA, 0, SEED)).normal(0.0, SIGMA,
                                                                                                  len(pe_real))
            out[(name, side)] = sim(c) + noise
    return out


def score(x, y, world, side, reps=REPS):
    """Rows for one world and side: relative RMSE of every rule against the observed output y."""
    warnings.filterwarnings("ignore")
    k = SIDES[side]
    at = x["AT"]
    feats = np.column_stack([at, x["V"], x["AP"], x["RH"]])
    rows = []
    for delta in DELTAS:
        for w in WIDTHS:
            lo, hi = ia.window_bounds(k, side, delta, w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = ia.test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < ia.MIN_TEST:
                raise RuntimeError(f"{side} delta={delta} w={w}: only {nt} test rows")
            Xte, yt = feats[tmask], y[tmask]
            for n in NS:
                if len(pool) < n:
                    raise RuntimeError(f"{side} delta={delta} w={w}: pool {len(pool)} < n {n}")
                for rep in range(reps):
                    idx = np.random.default_rng(orr.sample_seed(f"{world}|{side}", delta, w, n, rep, SEED)).choice(
                        pool, n, replace=False)
                    Xtr, ytr = feats[idx], y[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    r0 = rmse(p0, yt)

                    def full(stmt):
                        half, rej = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0],
                                                         Xte[:, 3])
                        return half, p0 + (half - p0) / ia.LAMBDA, rej

                    half, whole, rej = full([(WORD[side], float(edge))])
                    rec = {"world": world, "side": side, "delta": delta, "w": w, "n": n, "rep": rep, "R0_abs": r0,
                           "slope_window": float(b[1]), "EDGE@0.5": rmse(half, yt) / r0,
                           "EDGE@1.0": rmse(whole, yt) / r0, "EDGE_rejected": bool(rej)}
                    for p in POSITIONS[side]:
                        rec[f"POS_{p:g}@1.0"] = rmse(full([(WORD[side], p)])[1], yt) / r0
                    rows.append(rec)
    return rows


def classify(cell_means):
    """'kink-like' / 'no-kink-like' / 'mixed' from the six (delta, w) cell means of EDGE@0.5 relative RMSE."""
    below, above = int((cell_means < 1.0).sum()), int((cell_means > 1.0).sum())
    if below >= CELLS_NEEDED:
        return "kink-like"
    if above >= CELLS_NEEDED:
        return "no-kink-like"
    return "mixed"


def partial_slopes(x, y):
    """Report only: AT slope in three AT segments from one regression with V, AP, RH as linear controls."""
    at = x["AT"]
    lo, hi = SEGMENTS
    seg = [np.minimum(at, lo), np.clip(at, lo, hi) - lo, np.maximum(at, hi) - hi]
    X = np.column_stack([np.ones(len(at)), *seg, x["V"], x["AP"], x["RH"]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"below": float(b[1]), "middle": float(b[2]), "above": float(b[3])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=str(HERE / "runs" / "realism_check"))
    a = ap.parse_args()
    x = load_ccpp_inputs(a.xlsx)
    ys = outputs(x, load_pe(a.xlsx))
    rows = []
    for (world, side), y in ys.items():
        rows += score(x, y, world, side)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cells.csv.gz", index=False)

    verdict, L = {}, ["Twin realism check. relative RMSE = rule / R0, against each world's observed output. "
                      f"rows={len(df)} seeds={SEED}", ""]
    for side in SIDES:
        cells = df[df["side"] == side].groupby(["world", "delta", "w"])["EDGE@0.5"].mean().unstack("world")
        labels = {w: classify(cells[w]) for w in ("real", "base", "kink")}
        ok = labels["base"] == "no-kink-like" and labels["kink"] == "kink-like"
        verdict[side] = {"real": labels["real"], "base": labels["base"], "kink": labels["kink"], "references_ok": ok}
        L += [f"=== {side} side: EDGE@0.5 by (delta, w) cell ===", cells.round(3).to_string(),
              f"classification: real {labels['real']}, base {labels['base']}, kink {labels['kink']}; "
              f"references behave as designed: {ok}", ""]
    pd.set_option("display.width", 220)
    cols = [c for c in df.columns if "@" in c]
    L += ["=== mean relative RMSE by world and side (NaN = rule not defined on that side) ===",
          df.groupby(["side", "world"])[cols].mean().round(3).to_string(), "",
          "=== R0 absolute RMSE (MW), window AT coefficient, guard rejection of EDGE ===",
          df.groupby(["side", "world"])[["R0_abs", "slope_window", "EDGE_rejected"]].mean().round(3).to_string(), "",
          "=== report only: AT slope by segment with V, AP, RH controls (MW/degC) ==="]
    slopes = {}
    for (world, side), y in ys.items():
        name = world if world == "real" else f"{world}|{side}"
        if name not in slopes and not (world == "base" and side == "cold"):
            slopes[name] = partial_slopes(x, y)
    L += [pd.DataFrame(slopes).T.round(3).to_string(), "", "VERDICT: " + json.dumps(verdict)]
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    (out / "summary.json").write_text(json.dumps({"verdict": verdict, "slopes": slopes}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
