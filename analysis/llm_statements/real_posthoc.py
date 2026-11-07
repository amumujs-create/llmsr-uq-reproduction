"""Post-hoc (NOT pre-registered) placebo check for the real-CCPP validation (docs/research/ccpp-real-data.md §10).

Fixed single-boundary statements placed by hand at many locations, scored with the same rule as §9
(window gate + blend 0.5). Answers: does any "flat above X" give the same gain, i.e. how much is the LLM's?
Usage: python real_posthoc.py <Folds5x2_pp.xlsx> [--reps 20] [--boot 4000]
Writes runs/real/posthoc.{txt,json}. Does not change the R1-R4 verdicts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from real_validation import COL, HARM_MW, load_pe  # noqa: E402

ABOVE = range(18, 37, 2)
BELOW = range(4, 17, 2)


def placebo_statements():
    st = {"POOL(LLM, above 29)": [("above", 29.0)], "120b(LLM, below 8)": [("below", 8.0)]}
    st.update({f"above {a}": [("above", float(a))] for a in ABOVE})
    st.update({f"below {b}": [("below", float(b))] for b in BELOW})
    return st


def summarize(cells, boot, seed=0):
    """Per statement: mean RMSE, mean regret vs R0 with a cell-level bootstrap 95% interval, share of cells better."""
    cells = cells.assign(diff=cells[COL] - cells.R0_data)
    per_cell = cells.groupby(["statement", "setting", "w", "d"])["diff"].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = {}
    for s, g in per_cell.groupby("statement"):
        d = g["diff"].to_numpy()
        bs = rng.choice(d, (boot, len(d))).mean(axis=1)
        by = cells[cells.statement == s].groupby("setting")[COL].mean()
        rows[s] = {"rmse": cells.loc[cells.statement == s, COL].mean(), "diff": d.mean(),
                   "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "win": (d < 0).mean(),
                   "harm_cells": (d > HARM_MW).mean(), "cells": len(d), **by.to_dict()}
    return pd.DataFrame(rows).T, per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run
    x = load_ccpp_inputs(a.xlsx)
    pe = load_pe(a.xlsx)
    cells = run(x, a.reps, external={"real": placebo_statements()}, observed=pe)
    t, per_cell = summarize(cells, a.boot)
    above = t.loc[[f"above {v}" for v in ABOVE], "diff"]
    pool = per_cell[per_cell.statement.str.startswith("POOL")]
    by_set = pool.groupby("setting")["diff"].agg(["mean", "min", "max", lambda s: (s < 0).mean()])
    by_set.columns = ["mean", "min", "max", "win"]
    extra = {"R0": float(cells.R0_data.mean()),
             "uniform_above_18_36_mean_diff": float(above.mean()),
             "POOL_diff_without_hot_cold": float(pool[pool.setting != "hot->cold"]["diff"].mean())}
    lines = ["POST-HOC, not pre-registered. diff = blend_0.5 - R0 (MW); interval = bootstrap over (setting,w,d) cells,",
             "which overlap in data, so it is optimistic.", f"{json.dumps(extra)}", "",
             t.astype(float).round(3).to_string(), "", "POOL diff by setting", by_set.round(3).to_string()]
    dest = HERE / "runs" / "real"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "posthoc.txt").write_text("\n".join(lines) + "\n")
    json.dump({"summary": t.astype(float).to_dict("index"), "pool_by_setting": by_set.to_dict("index"), **extra},
              open(dest / "posthoc.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
