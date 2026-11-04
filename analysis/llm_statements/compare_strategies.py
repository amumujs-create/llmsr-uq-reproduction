"""Compare prompting strategies S0 / S1 / S2 (docs/research/llm-statement-collection.md §12).

Usage: python compare_strategies.py <xlsx> --s0 runs/pilot/*.jsonl --s12 runs/strategy/*.jsonl --out runs/strategy/compare [--reps 20]
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
from elicit import assemble_s2  # noqa: E402
from parse_score import majority, parse, score, to_statement  # noqa: E402

RULES = ["R0_data", "R2g", "blend_0.25", "blend_0.5", "budget_2", "budget_5"]


def rows_from(path, strategy=None):
    for line in open(path):
        r = json.loads(line)
        if r["world"] in ("any", "real") or r.get("data", "D0") != "D0" or r["level"] not in ("L1", "L2"):
            continue
        st = strategy or r["strategy"]
        if r["error"]:
            ok, stmt, parts = False, [], {}
        elif st == "S2":
            p = assemble_s2(r["response"])
            ok, stmt, parts = p["ok"], p["stmt"], p["parts"]
        else:
            p = parse(r["response"])
            ok, stmt, parts = p["ok"], to_statement(p["boundaries"]), {}
        s = score(r["world"], stmt) if ok else {"cold": "parse_fail", "hot": "parse_fail", "all_correct": False}
        yield {"tag": r.get("tag", r["model"]), "strategy": st, "world": r["world"], "level": r["level"],
               "temperature": r["temperature"], "ok": ok, "stmt": stmt, "parts": parts, **s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--s0", nargs="+", required=True)
    ap.add_argument("--s12", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    recs = [r for p in a.s0 for r in rows_from(p, "S0")] + [r for p in a.s12 for r in rows_from(p)]
    df = pd.DataFrame(recs)
    samp = df[df.temperature > 0]
    lines, out = [], {}

    t = samp.groupby(["tag", "strategy", "level"]).agg(p_struct=("all_correct", "mean"), parse_ok=("ok", "mean"),
                                                         n=("ok", "size")).round(3)
    out["p_struct"] = {"|".join(k): v for k, v in t.to_dict("index").items()}
    lines.append("=== structural reliability by strategy (samples at T=0.7) ===")
    lines.append(t.to_string())

    by_world = samp.pivot_table(index=["tag", "level", "world"], columns="strategy", values="all_correct",
                                aggfunc="mean").round(2)
    lines.append("\n=== p_struct by world ===")
    lines.append(by_world.to_string())

    side = samp.melt(id_vars=["tag", "strategy", "level"], value_vars=["cold", "hot"], var_name="side",
                     value_name="verdict")
    sv = side.groupby(["tag", "strategy", "level", "side"]).verdict.value_counts(normalize=True).round(2).unstack(fill_value=0)
    out["per_side"] = {"|".join(k): v for k, v in sv.to_dict("index").items()}
    lines.append("\n=== per-side verdicts ===")
    lines.append(sv.to_string())

    s2 = samp[(samp.strategy == "S2") & samp.ok]
    if len(s2):
        parts = pd.DataFrame([{**p, "tag": t_, "level": lv, "world": w} for p, t_, lv, w in
                              zip(s2.parts, s2.tag, s2.level, s2.world)])
        summ = {}
        for (tag, lv), g in parts.groupby(["tag", "level"]):
            summ[f"{tag}|{lv}"] = {c: {"rate": round(float(g[c].notna().mean()), 2),
                                       "median": None if g[c].isna().all() else round(float(g[c].median()), 1)}
                                   for c in g.columns if c not in ("tag", "level", "world")}
        out["s2_mechanisms"] = summ
        lines.append("\n=== S2 mechanism answers (rate claimed active & in range, median degC) ===")
        lines.append(json.dumps(summ, indent=1))

    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run
    ext = {}
    for (tag, st, lv, w), g in samp[samp.ok].groupby(["tag", "strategy", "level", "world"]):
        ext.setdefault(w, {})[f"{tag}|{st}|{lv}"] = majority(list(g.stmt))
    cells = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False)
    res = cells.groupby("statement")[RULES].mean().round(2)
    res["harm_R2g"] = cells.assign(h=cells.R2g > cells.R0_data + 1).groupby("statement").h.mean().round(3)
    out["outcome_world_mean"] = res.to_dict("index")
    lines.append("\n=== outcome of majority statements, mean over worlds (clean RMSE MW) ===")
    lines.append(res.to_string())

    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
