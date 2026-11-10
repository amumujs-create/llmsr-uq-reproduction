"""POST-HOC slice of the registered information-axis run (not a verdict).

Relative RMSE (rule / R0) on the rows where the kink is outside the window (delta > 0), by world.
Reads analysis/ccpp_twin/runs/info_axis/cells.csv.gz; no recomputation.
"""
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "runs" / "info_axis"
VALIDATION = [f"{m}|{lv}|HA" for m in ("gemma3-27b", "llama33-70b", "gpt-oss-20b") for lv in ("L1", "L2")]
DEVELOPMENT = [f"{m}|{lv}|HA" for m in ("gpt-oss-120b", "qwen3-30b-a3b-instruct") for lv in ("L1", "L2")]


def main():
    d = pd.read_csv(RUN / "cells.csv.gz")
    fake = [c for c in d.columns if c.startswith("P|")]
    rel = d[["ORACLE", "D_real"] + VALIDATION + DEVELOPMENT + fake].div(d["R0"], axis=0)
    rel["P_mean"] = rel[fake].mean(axis=1)
    rel = rel.drop(columns=fake)
    out = ["POST-HOC slice of the registered info-axis run. Not a verdict. relative RMSE = rule / R0."]
    for label, mask in (("delta > 0", d["delta"] > 0), ("delta >= 0", d["delta"] >= 0)):
        sub, mod = rel[mask], d.loc[mask, "module"]
        out.append(f"\n=== {label}: rows per world ===\n{mod.value_counts().sort_index().to_string()}")
        out.append(f"\n=== {label}: mean ===\n{sub.groupby(mod).mean().round(3).T.to_string()}")
        out.append(f"\n=== {label}: median ===\n{sub.groupby(mod).median().round(3).T.to_string()}")
        out.append(f"\n=== {label}: harm share (> 1.10) ===\n{(sub > 1.10).groupby(mod).mean().round(3).T.to_string()}")
    text = "\n".join(out) + "\n"
    (RUN / "posthoc_outside.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
