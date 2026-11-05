"""Tail-risk metrics and cross-model consensus (docs/research/ccpp-robustness-tail-consensus.md, pre-registered).

Usage:
  python robustness.py <Folds5x2_pp.xlsx>                       # dev cells only (code check; no verdicts)
  python robustness.py <Folds5x2_pp.xlsx> --split validation --final
                                                                # once, after all five peer models are in
Writes runs/robustness/<split>.{txt,json}. Validation outcomes are never computed without --final, and --final
refuses to run until every peer model has S0 answers for every (level, world).
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
from assemble import CORROBORATE_C, LAMBDA, build, load  # noqa: E402
from parse_score import TRUTH, majority  # noqa: E402

PEERS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")  # models.tsv
CRASH_MW = 4.3      # one noise standard deviation (§3)
HARM_MW = 1.0
RULES = ("HA", "S0maj", "XM", "POOL")
COL = f"blend_{LAMBDA}"


def peer_majorities(s0, level, world, exclude):
    return {t: majority(s0[(t, level, world)]) for t in PEERS if t != exclude and s0.get((t, level, world))}


def confirmed(boundary, peer_stmts):
    side, at = boundary[0], boundary[1]
    return sum(any(s == side and abs(a - at) <= CORROBORATE_C for s, a in st) for st in peer_stmts)


def xm(ha_stmt, peer_stmts):
    """§4: keep an HA boundary only when more than half of the peers' S0 majorities name the same side within
    CORROBORATE_C. Removes boundaries, never adds them. With four peers this is 3 of 4."""
    peers = list(peer_stmts)
    if not peers:
        return []
    return [b for b in ha_stmt if confirmed(b, peers) * 2 > len(peers)]


def pool(s0, level, world):
    """§4: majority over the five models' S0 majority statements (location = median)."""
    stmts = list(peer_majorities(s0, level, world, exclude=None).values())
    return majority(stmts) if stmts else []


def missing_peers(s0):
    return sorted({f"{t}|{lv}|{w}" for t in PEERS for lv in ("L1", "L2") for w in TRUTH if not s0.get((t, lv, w))})


def statements_for(split, s0, s2):
    ext, _, meta = build(s0, s2)
    cells = sorted(c for c, m in meta.items() if m["split"] == split)
    out = {w: {} for w in TRUTH}
    for cell in cells:
        tag, level = cell.split("|")
        for world in TRUTH:
            ha = ext[world].get(f"{cell}|HA")
            if ha is None:
                continue
            out[world][f"{cell}|HA"] = ha
            if f"{cell}|S0maj" in ext[world]:
                out[world][f"{cell}|S0maj"] = ext[world][f"{cell}|S0maj"]
            out[world][f"{cell}|XM"] = xm(ha, peer_majorities(s0, level, world, exclude=tag).values())
    pools = {f"{lv}|{w}": pool(s0, lv, w) for lv in ("L1", "L2") for w in TRUTH}
    # POOL is one statement per level, shared by every cell, so its outcome on dev cells would already be its
    # validation outcome. Score it only in the final validation run (§6).
    if split == "validation":
        for level in sorted({c.split("|")[1] for c in cells}):
            for world in TRUTH:
                out[world][f"POOL@{level}|POOL"] = pools[f"{level}|{world}"]
    return out, cells, meta, pools


def tail_table(df, cells):
    """Row-level regret vs R0 (no averaging over reps first), per cell and rule (§3)."""
    recs = []
    for name, g in df.groupby("statement"):
        head, rule = name.split("|")[:-1], name.split("|")[-1]
        targets = [c for c in cells if c.endswith("|" + head[0].split("@")[1])] if rule == "POOL" else ["|".join(head)]
        reg = (g[COL] - g.R0_data).to_numpy()
        q = np.quantile(reg, 0.95)
        for cell in targets:
            recs.append({"cell": cell, "rule": rule, "rmse": float(g[COL].mean()), "R0": float(g.R0_data.mean()),
                         "CVaR95": float(reg[reg >= q].mean()), "crash": float((reg > CRASH_MW).mean()),
                         "harm": float((reg > HARM_MW).mean()), "max": float(reg.max()), "rows": int(len(reg))})
    return pd.DataFrame(recs)


def verdicts(t):
    p = {m: t.pivot_table(index="cell", columns="rule", values=m) for m in ("rmse", "CVaR95", "crash")}
    n = len(p["rmse"])
    return {"n_cells": n,
            "T1": bool((p["crash"]["HA"] <= 0.05).all()),
            "T2": int((p["CVaR95"]["HA"] < p["CVaR95"]["S0maj"]).sum()) >= 4,
            "X1": int((p["CVaR95"]["XM"] <= p["CVaR95"]["HA"]).sum()) >= 4,
            "X2": bool((p["rmse"]["XM"] <= p["rmse"]["HA"] + 0.10).all()),
            "X3": int((p["rmse"]["POOL"] < p["rmse"]["S0maj"]).sum()) >= 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--split", choices=("dev", "validation"), default="dev")
    ap.add_argument("--final", action="store_true", help="required for --split validation (one-time verdict)")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    if a.split == "validation":
        if not a.final:
            sys.exit("validation outcomes are computed once, with --final (ccpp-robustness-tail-consensus.md §6)")
        miss = missing_peers(s0)
        if miss:
            sys.exit(f"--final refused: peer S0 answers missing for {len(miss)} (model|level|world), e.g. {miss[:3]}")
    ext, cells, meta, pools = statements_for(a.split, s0, s2)
    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run
    df = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False)
    t = tail_table(df, cells)
    lines = [f"split={a.split} cells={len(cells)} peers_missing={len(missing_peers(s0))}",
             f"rows per (cell, rule): {int(t.rows.min())}", ""]
    for m in ("rmse", "CVaR95", "crash", "harm", "max"):
        lines += [f"=== {m} ===", t.pivot_table(index="cell", columns="rule", values=m).round(3).to_string(), ""]
    xm_changes = {w: {k: v for k, v in d.items() if k.endswith("|XM")} for w, d in ext.items()}
    out = {"split": a.split, "cells": cells, "table": t.to_dict("records"), "statements": ext,
           "xm_statements": xm_changes, "pool_statements": pools}
    if a.split == "validation":
        out["verdicts"] = verdicts(t)
        lines += ["=== pre-registered verdicts (T1, T2, X1-X3) ===", json.dumps(out["verdicts"], indent=1)]
    dest = HERE / "runs" / "robustness"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{a.split}.txt").write_text("\n".join(lines) + "\n")
    json.dump(out, open(dest / f"{a.split}.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
