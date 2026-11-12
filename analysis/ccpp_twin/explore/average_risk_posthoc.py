"""POST-HOC (not a verdict): the clean-window confirmation re-read in absolute MW.

docs/research/ccpp-twin-average-risk-draft.md. Reads runs/clean_diag/cells.csv.gz (confirmation, already scored once)
and runs/clean_diag_dev/cells.csv.gz (development). Computes nothing new from LLM answers; it only re-aggregates.

For every rule: mean absolute RMSE (MW) over rows of each task, then the plain mean over the six kink tasks and over the
six no-kink tasks, and the mean of those two ("balanced": a kink on the extrapolation side half of the time).
  READ_EDGE : no LLM. The description names the device side (keyword reading); flat from the window edge on that side.
              Equals EDGE_h on kink tasks and R0 on no-kink tasks in this task set.
  break-even: share of kink cases above which the rule beats R0 on average = loss / (gain + loss).

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/average_risk_posthoc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
JUDGES = ("FROZEN", "S0", "S2", "AND", "OR", "CONF")


def table(path):
    df = pd.read_csv(path)
    models = sorted({c.split("|")[0] for c in df.columns if c.endswith("|AND")})
    rel = {"R0": pd.Series(1.0, index=df.index), "ORACLE": df["ORACLE"], "READ": df["READ"], "EDGE_h": df["EDGE_h"],
           "READ_EDGE": df["EDGE_h"].where(df["kind"] == "kink", 1.0)}
    for m in models:
        for j in JUDGES:
            rel[f"{m}|{j}"] = df[f"{m}|{j}"]
    rows = {}
    for name, r in rel.items():
        mw = (r * df["R0_abs"]).groupby([df["kind"], df["task"]]).mean()
        k, nk = mw["kink"].mean(), mw["nokink"].mean()
        rows[name] = dict(kink_MW=k, nokink_MW=nk, balanced_MW=0.5 * (k + nk),
                          worst_task_ratio=((r * df["R0_abs"]).groupby(df["task"]).mean()
                                            / df["R0_abs"].groupby(df["task"]).mean()).max())
    t = pd.DataFrame(rows).T
    r0 = t.loc["R0"]
    t["gain_kink"] = r0["kink_MW"] - t["kink_MW"]
    t["loss_nokink"] = t["nokink_MW"] - r0["nokink_MW"]
    t["vs_R0"] = t["balanced_MW"] / r0["balanced_MW"] - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t["break_even"] = np.where(t["gain_kink"] > 0, t["loss_nokink"].clip(lower=0)
                                   / (t["gain_kink"] + t["loss_nokink"].clip(lower=0)), np.nan)
    return t


def main():
    pd.set_option("display.width", 220)
    L = ["POST-HOC (not a verdict). Absolute RMSE in MW; balanced = mean of kink and no-kink task means.", ""]
    for label, sub in (("confirmation (four new models)", "clean_diag"), ("development (five old models)",
                                                                           "clean_diag_dev")):
        L += [f"=== {label} ===", table(RUNS / sub / "cells.csv.gz").round(3).to_string(), ""]
    out = RUNS / "explore_average_risk"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L)
    (out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
