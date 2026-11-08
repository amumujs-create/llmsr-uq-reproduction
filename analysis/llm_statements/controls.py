"""World-blind controls for the hint assembly (docs/research/ccpp-world-blind-controls.md, pre-registered 2026-11-08).

The same statement is applied to all five twin worlds, scored with the HA rule (window gate + blend 0.5) on the
same windows, reps and seeds:
  P  placebo grid: single boundaries "above X" (X = 18..36) and "below X" (X = 4..16)
  D  domain default: real-plant regression breakpoints (ccpp-domain-check.md §3, §5), fixed 2026-11-04
  M  model default: the model's S0 majority on the base world (no device), used in every world
HA and S0maj are recomputed in the same run (must match runs/assembly within 0.01 MW).

Usage: python controls.py <Folds5x2_pp.xlsx> [--runs runs] [--out runs/controls/controls] [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from assemble import LAMBDA, assemble, load, split_of  # noqa: E402
from parse_score import TRUTH, majority, score  # noqa: E402

PLACEBO_ABOVE = tuple(range(18, 37, 2))
PLACEBO_BELOW = tuple(range(4, 17, 2))
DOMAIN = {"D_real": [("below", 8.0), ("above", 27.0)],
          "D_real_cold": [("below", 8.0)], "D_real_hot": [("above", 27.0)]}
VALIDATION_NEED = {"K1": 4, "K2": 4, "K3": 4, "K4": 5}


def placebo_statements():
    st = {f"P|above {a}": [("above", float(a))] for a in PLACEBO_ABOVE}
    st.update({f"P|below {b}": [("below", float(b))] for b in PLACEBO_BELOW})
    return st


def world_blind_structural_ceiling(statements):
    """Share of the 5 worlds in which a world-blind statement is all-correct, per statement (§4)."""
    return {name: float(np.mean([score(w, st)["all_correct"] for w in TRUTH])) for name, st in statements.items()}


def model_defaults(s0):
    """M: per cell, the S0 majority on the base world (empty statement when the majority names no side)."""
    out = {}
    for (tag, level, world), stmts in s0.items():
        if world == "base":
            out[f"{tag}|{level}"] = majority(stmts)
    return out


def build_external(s0, s2):
    """{world: {name: statement}} for every control and for HA / S0maj of every cell."""
    blind = {**placebo_statements(), **DOMAIN}
    cells = sorted({(tag, level) for tag, level, _ in set(s0) | set(s2)})
    meta = {f"{tag}|{level}": {"tag": tag, "level": level,
                               "split": split_of(tag, any(s2.get((tag, level, w)) for w in TRUTH))}
            for tag, level in cells}
    m = model_defaults(s0)
    ext = defaultdict(dict)
    for world in TRUTH:
        ext[world].update(blind)
        for tag, level in cells:
            cell = f"{tag}|{level}"
            a, b = s0.get((tag, level, world), []), s2.get((tag, level, world), [])
            ext[world][f"{cell}|HA"] = assemble(a, b)
            if a:
                ext[world][f"{cell}|S0maj"] = majority(a)
            if cell in m:
                ext[world][f"{cell}|M"] = m[cell]
    return ext, meta, m


def cell_table(cells, meta):
    """Per (cell, rule): mean blend RMSE over the five worlds, harm rate, and per-world means.

    World-blind statements (P, D) have no cell; they are copied to every cell so comparisons line up."""
    col = f"blend_{LAMBDA}"
    rows = []
    per_stmt = {}
    for name, g in cells.groupby("statement"):
        per_stmt[name] = {"rmse": float(g[col].mean()), "harm": float((g[col] > g.R0_data + 1.0).mean()),
                          "by_world": g.groupby("module")[col].mean().round(3).to_dict()}
    r0 = {"rmse": float(cells.R0_data.mean()), "harm": 0.0}
    p_names = [n for n in per_stmt if n.startswith("P|")]
    p_mean = float(np.mean([per_stmt[n]["rmse"] for n in p_names]))
    p_harm = float(np.mean([per_stmt[n]["harm"] for n in p_names]))
    p_best_name = min(p_names, key=lambda n: per_stmt[n]["rmse"])
    for cell, info in meta.items():
        ha = per_stmt.get(f"{cell}|HA")
        if ha is None:
            continue
        row = {"cell": cell, "split": info["split"], "R0": r0["rmse"], "HA": ha["rmse"], "HA_harm": ha["harm"],
               "S0maj": per_stmt.get(f"{cell}|S0maj", {}).get("rmse"),
               "M": per_stmt.get(f"{cell}|M", r0)["rmse"], "M_harm": per_stmt.get(f"{cell}|M", r0)["harm"],
               "D_real": per_stmt["D_real"]["rmse"], "D_real_cold": per_stmt["D_real_cold"]["rmse"],
               "D_real_hot": per_stmt["D_real_hot"]["rmse"],
               "P_mean": p_mean, "P_harm_mean": p_harm, "P_best": per_stmt[p_best_name]["rmse"],
               "P_best_name": p_best_name,
               "n_P_beaten": int(sum(ha["rmse"] < per_stmt[n]["rmse"] for n in p_names))}
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell"), per_stmt


def verdicts(table):
    v = table[table.split == "validation"]
    if v.empty:
        return {"status": "no validation cells"}
    wins = {"K1": int((v.HA < v.D_real).sum()), "K2": int((v.HA < v.M).sum()),
            "K3": int((v.HA < v.P_mean).sum()), "K4": int((v.HA_harm <= v.P_harm_mean).sum())}
    out = {"n_cells": int(len(v)), "wins": wins,
           **{k: bool(wins[k] >= need) for k, need in VALIDATION_NEED.items()}}
    out["report"] = {"HA_minus_P_best": (v.HA - v.P_best).round(3).to_dict(),
                     "n_P_beaten_of_17": v.n_P_beaten.to_dict(),
                     "HA_lt_D_real_cold": int((v.HA < v.D_real_cold).sum()),
                     "HA_lt_D_real_hot": int((v.HA < v.D_real_hot).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs/controls/controls"))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run

    s0, s2 = load(Path(a.runs))
    ext, meta, m = build_external(s0, s2)
    blind = {**placebo_statements(), **DOMAIN}
    ceiling = world_blind_structural_ceiling(blind)
    cells = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False)
    table, per_stmt = cell_table(cells, meta)
    v = verdicts(table)

    # seed check against the registered HA run
    prev = HERE / "runs/assembly/assembly.json"
    check = {}
    if prev.exists():
        old = json.load(open(prev))["rmse"]
        for key, vals in old.items():
            cell = key.split("|", 1)[1]
            if cell in table.index:
                check[cell] = {"HA_prev": vals.get("HA"), "HA_now": round(float(table.loc[cell, "HA"]), 2),
                               "R0_prev": vals.get("R0"), "R0_now": round(float(table.loc[cell, "R0"]), 2)}
    mism = [c for c, d in check.items() if d["HA_prev"] is not None and abs(d["HA_prev"] - d["HA_now"]) > 0.01]

    show = ["split", "R0", "HA", "S0maj", "M", "D_real", "D_real_cold", "D_real_hot", "P_mean", "P_best", "P_best_name",
            "n_P_beaten", "HA_harm", "M_harm", "P_harm_mean"]
    lines = ["world-blind controls (ccpp-world-blind-controls.md §2-§5); mean clean RMSE over 5 worlds, blend 0.5", "",
             table[show].round(3).sort_values("split").to_string(), "",
             "=== world-blind statements: structural ceiling (share of 5 worlds all-correct) ===",
             json.dumps(ceiling), "",
             "=== per-statement RMSE by world (world-blind statements) ===",
             json.dumps({n: per_stmt[n]["by_world"] for n in blind}, indent=1), "",
             "=== model defaults M (base-world S0 majority) ===", json.dumps(m), "",
             f"=== seed check vs runs/assembly (|HA diff| > 0.01 MW): {mism or 'none'} ===", "",
             "=== verdicts (validation cells) ===", json.dumps(v, indent=1, default=str)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump({"table": table.round(4).to_dict("index"), "ceiling": ceiling, "model_defaults": m,
               "per_statement": per_stmt, "seed_check": check, "verdicts": v}, open(a.out + ".json", "w"),
              indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
