"""Shape-library assembly HA_shape (docs/research/ccpp-shape-assembly.md, pre-registered 2026-11-08).

HA boundaries stay as they are; the slope beyond each boundary is set by the LLM's elicited shape (shape stage 1,
runs/shape/<tag>.jsonl) instead of the fixed factor 0.0:
  near_zero 0.0, weaker 0.5, same_or_stronger 1.0, opposite -1.0; no matching shape item -> 0.0 (HA default).
Comparators in the same run: HA (factor 0.0), HA_half (factor 0.5 everywhere, data-free), SHAPE_ORACLE (truth shape
at the HA boundaries: cap near_zero 0.0, evap weaker 0.5).

Usage: python shape_assembly.py <Folds5x2_pp.xlsx> [--runs runs] [--out runs/shape_assembly/shape_assembly] [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from assemble import LAMBDA, assemble, load, split_of  # noqa: E402
from parse_score import TOL_C, TRUTH, parse_shape  # noqa: E402

FACTOR = {"near_zero": 0.0, "weaker": 0.5, "same_or_stronger": 1.0, "opposite": -1.0}
DEFAULT_FACTOR = 0.0
HALF = 0.5
TRUTH_SHAPE = {"below": "near_zero", "above": "weaker"}   # cold cap / evaporative cooling (§2)
VALIDATION_NEED = {"J1": 6, "J2": 2, "J3": 3, "J4": 3}


def load_shapes(runs):
    """{(tag, level, world): [item, ...]} from shape stage 1 (all samples, parsed)."""
    out = defaultdict(list)
    for path in sorted((runs / "shape").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["error"] is not None or r["world"] not in TRUTH:
                continue
            p = parse_shape(r["response"])
            if p["ok"]:
                out[(r["tag"], r["level"], r["world"])].extend(
                    {**it, "sample": r["sample"]} for it in p["items"])
    return out


def shape_at(items, side, at):
    """Majority response_beyond among items on the same side within TOL_C of the HA boundary; None if no item or tie."""
    hits = [it["response_beyond"] for it in items if it["side"] == side and abs(it["at_c"] - at) <= TOL_C
            and it["response_beyond"] in FACTOR]
    if not hits:
        return None, 0
    top = Counter(hits).most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, len(hits)
    return top[0][0], len(hits)


def factors_for(stmt, items):
    """Per-boundary factors for HA_shape, plus the shape labels used (None -> default)."""
    labels, facs = [], []
    for side, at in stmt:
        lab, _ = shape_at(items, side, at)
        labels.append(lab)
        facs.append(FACTOR[lab] if lab is not None else DEFAULT_FACTOR)
    return facs, labels


def build(s0, s2, shapes):
    ext, fac, meta, labels = defaultdict(dict), defaultdict(dict), {}, {}
    cells = sorted({(tag, level) for tag, level, _ in set(s0) | set(s2)})
    for tag, level in cells:
        cell = f"{tag}|{level}"
        meta[cell] = {"tag": tag, "level": level,
                      "split": split_of(tag, any(s2.get((tag, level, w)) for w in TRUTH))}
        for world in TRUTH:
            stmt = assemble(s0.get((tag, level, world), []), s2.get((tag, level, world), []))
            items = shapes.get((tag, level, world), [])
            f_shape, labs = factors_for(stmt, items)
            for rule, f in (("HA", DEFAULT_FACTOR), ("HA_half", HALF), ("HA_shape", f_shape),
                            ("SHAPE_ORACLE", [FACTOR[TRUTH_SHAPE[s]] for s, _ in stmt])):
                ext[world][f"{cell}|{rule}"] = stmt
                fac[world][f"{cell}|{rule}"] = f
            labels[f"{cell}|{world}"] = {"stmt": stmt, "labels": labs, "n_shape_items": len(items)}
    return ext, fac, meta, labels


def cell_table(cells, meta):
    col = f"blend_{LAMBDA}"
    per = {}
    for name, g in cells.groupby("statement"):
        per[name] = {"rmse": float(g[col].mean()), "harm": float((g[col] > g.R0_data + 1.0).mean()),
                     "by_world": g.groupby("module")[col].mean().round(3).to_dict()}
    rows = []
    for cell, info in meta.items():
        if f"{cell}|HA" not in per:
            continue
        row = {"cell": cell, "split": info["split"], "R0": float(cells.R0_data.mean())}
        for rule in ("HA", "HA_half", "HA_shape", "SHAPE_ORACLE"):
            row[rule] = per[f"{cell}|{rule}"]["rmse"]
            row[f"{rule}_harm"] = per[f"{cell}|{rule}"]["harm"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell"), per


def verdicts(table):
    v = table[table.split == "validation"]
    if v.empty:
        return {"status": "no validation cells"}
    llama = v[v.index.str.startswith("llama33-70b")]
    others = v[~v.index.str.startswith("llama33-70b")]
    wins = {"J1": int((v.HA_shape_harm <= 0.05).sum()),
            "J2": int((llama.HA_shape > llama.HA).sum()),
            "J3": int((others.HA_shape <= others.HA).sum()),
            "J4": int((v.HA_shape < v.HA_half).sum())}
    out = {"n_cells": int(len(v)), "wins": wins,
           "J1": bool(wins["J1"] >= VALIDATION_NEED["J1"]), "J2": bool(wins["J2"] >= VALIDATION_NEED["J2"]),
           "J3": bool(wins["J3"] >= VALIDATION_NEED["J3"]), "J4": bool(wins["J4"] <= VALIDATION_NEED["J4"])}
    out["report"] = {"HA_shape_minus_HA": (v.HA_shape - v.HA).round(3).to_dict(),
                     "HA_shape_minus_HA_half": (v.HA_shape - v.HA_half).round(3).to_dict(),
                     "SHAPE_ORACLE_minus_HA": (v.SHAPE_ORACLE - v.HA).round(3).to_dict()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs/shape_assembly/shape_assembly"))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run

    runs = Path(a.runs)
    s0, s2 = load(runs)
    shapes = load_shapes(runs)
    ext, fac, meta, labels = build(s0, s2, shapes)
    cells = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False, external_factor=fac)
    table, per = cell_table(cells, meta)
    v = verdicts(table)

    prev = HERE / "runs/assembly/assembly.json"
    mism = []
    if prev.exists():
        old = json.load(open(prev))["rmse"]
        for key, vals in old.items():
            cell = key.split("|", 1)[1]
            if cell in table.index and vals.get("HA") is not None and abs(vals["HA"] - table.loc[cell, "HA"]) > 0.01:
                mism.append(cell)
    fallback = {c: sum(lab is None for lab in d["labels"]) for c, d in labels.items() if d["stmt"]}
    show = ["split", "R0", "HA", "HA_half", "HA_shape", "SHAPE_ORACLE", "HA_harm", "HA_half_harm", "HA_shape_harm"]
    lines = ["shape-library assembly (ccpp-shape-assembly.md §2-§4); mean clean RMSE over 5 worlds, blend 0.5", "",
             table[show].round(3).sort_values("split").to_string(), "",
             "=== shape labels at HA boundaries (cell|world: statement, labels; None -> factor 0.0) ===",
             json.dumps({c: d for c, d in labels.items() if d["stmt"]}, default=str), "",
             f"=== boundaries without a matching shape item (fallback): {sum(fallback.values())} of "
             f"{sum(len(d['stmt']) for d in labels.values())} ===", "",
             f"=== seed check vs runs/assembly (|HA diff| > 0.01 MW): {mism or 'none'} ===", "",
             "=== verdicts (validation cells) ===", json.dumps(v, indent=1, default=str)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump({"table": table.round(4).to_dict("index"), "labels": labels, "per_statement": per,
               "fallback": fallback, "seed_mismatch": mism, "verdicts": v},
              open(a.out + ".json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
