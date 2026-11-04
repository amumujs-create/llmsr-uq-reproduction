"""Analyze collected regime statements (docs/research/llm-statement-collection.md §7–§8).

Usage: python analyze.py <Folds5x2_pp.xlsx> <runs.jsonl> [<runs.jsonl> ...] --out <prefix> [--reps 20] [--no-outcome]
Writes <prefix>.json and <prefix>.txt.
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
from parse_score import TRUTH, agreement, majority, parse, score, to_statement  # noqa: E402
from prompts import WORLDS  # noqa: E402

RULES = ["R0_data", "R2g", "blend_0.25", "blend_0.5", "blend_0.75", "budget_2", "budget_5", "budget_10"]


def auroc(score_, y):
    score_, y = np.asarray(score_, float), np.asarray(y, int)
    pos, neg = score_[y == 1], score_[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def load(paths):
    recs = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            pr = parse(r["response"]) if r["error"] is None else {"ok": False, "boundaries": [], "duplicates": 0,
                                                                   "error": "call_error"}
            r.update(ok=pr["ok"], parse_error=pr["error"], stmt=to_statement(pr["boundaries"]),
                     conf=[c for _, _, c in pr["boundaries"] if c is not None], tag=r.get("tag", r["model"]))
            recs.append(r)
    return recs


def expand(recs):
    """Score world-independent L0+D0 answers against every twin world; drop real-CCPP rows (no truth)."""
    out = []
    for r in recs:
        if r["world"] == "real":
            continue
        for w in (WORLDS if r["world"] == "any" else [r["world"]]):
            out.append({**r, "world": w})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-outcome", action="store_true")
    a = ap.parse_args()
    recs = load(a.runs)
    lines, out = [], {}

    fails = pd.DataFrame([{"tag": r["tag"], "ok": r["ok"], "err": r["parse_error"]} for r in recs])
    out["parse"] = {t: {"n": int(len(g)), "fail_rate": round(1 - g.ok.mean(), 3),
                        "errors": g.err.value_counts().to_dict()} for t, g in fails.groupby("tag")}
    lines.append(f"=== calls and parse failures === {json.dumps(out['parse'])}")

    rows = []
    for r in expand(recs):
        s = score(r["world"], r["stmt"]) if r["ok"] else {"cold": "parse_fail", "hot": "parse_fail", "all_correct": False}
        rows.append({"tag": r["tag"], "world": r["world"], "level": r["level"], "data": r["data"],
                     "setting": r["setting"] or "none", "sample": r["sample"], "temperature": r["temperature"], "ok": r["ok"],
                     "stmt": r["stmt"], "conf": float(np.mean(r["conf"])) if r["conf"] else np.nan, **s})
    df = pd.DataFrame(rows)
    samp = df[df.temperature > 0]
    keys = ["tag", "level", "data", "world", "setting"]

    # Majority statement and per-sample agreement with it.
    maj = {}
    for k, g in samp[samp.ok].groupby(keys, dropna=False):
        maj[k] = majority(list(g.stmt))
    samp = samp.assign(agree=[agreement(st, maj.get(tuple(r[c] for c in keys), [])) if ok else False
                              for st, ok, (_, r) in zip(samp.stmt, samp.ok, samp.iterrows())])
    agree_rate = samp.groupby(keys, dropna=False).agree.mean()
    samp = samp.assign(agree_rate=[agree_rate[tuple(r[c] for c in keys)] for _, r in samp.iterrows()])

    p_struct = samp.groupby(["tag", "level", "data"]).all_correct.mean().round(3)
    maj_rows = [{"tag": k[0], "level": k[1], "data": k[2], "world": k[3], "setting": k[4],
                 **score(k[3], v)} for k, v in maj.items()]
    majdf = pd.DataFrame(maj_rows)
    p_maj = majdf.groupby(["tag", "level", "data"]).all_correct.mean().round(3)
    greedy = df[df.temperature == 0]
    p_greedy = greedy.groupby(["tag", "level", "data"]).all_correct.mean().round(3)
    tab = pd.DataFrame({"p_struct_sample": p_struct, "p_struct_majority": p_maj, "p_struct_greedy": p_greedy})
    out["p_struct"] = {"|".join(map(str, k)): v for k, v in tab.to_dict("index").items()}
    lines.append("\n=== structural reliability p (all sides correct) ===")
    lines.append(tab.to_string())

    side = samp.melt(id_vars=["tag", "level", "world"], value_vars=["cold", "hot"], var_name="side", value_name="verdict")
    side_tab = side.groupby(["tag", "level", "world", "side"]).verdict.value_counts(normalize=True).round(2).unstack(fill_value=0)
    lines.append("\n=== per-side verdicts (share of samples) ===")
    lines.append(side_tab.to_string())
    out["per_side"] = {"|".join(map(str, k)): v for k, v in side_tab.to_dict("index").items()}

    # Label-free signals (Q-c).
    other = {}
    for k, v in maj.items():
        other.setdefault(k[1:], []).append((k[0], v))
    samp = samp.assign(cross=[np.mean([agreement(st, v) for t, v in other.get(tuple(r[c] for c in keys[1:]), [])
                                       if t != r["tag"]]) if len(other.get(tuple(r[c] for c in keys[1:]), [])) > 1
                              else np.nan for st, (_, r) in zip(samp.stmt, samp.iterrows())])
    ok = samp[samp.ok]
    sig = {"agree_rate": auroc(ok.agree_rate, ok.all_correct),
           "confidence": auroc(ok.conf.fillna(ok.conf.mean() if ok.conf.notna().any() else 0.5), ok.all_correct)}
    if ok.cross.notna().any():
        c = ok[ok.cross.notna()]
        sig["cross_model"] = auroc(c.cross, c.all_correct)
    out["signal_auroc"] = sig
    lines.append(f"\n=== label-free signal AUROC for all_correct (per sample) === {sig}")

    if not a.no_outcome:
        from ccpp_twin import load_ccpp_inputs
        from risk_limits import run
        ext = {}
        for k, v in maj.items():
            tag, level, data, world, setting = k
            if data != "D0":
                continue  # D1 statements are window-specific; outcome for them is a later step
            ext.setdefault(world, {})[f"{tag}|{level}|majority"] = v
        for _, r in greedy[(greedy.data == "D0") & greedy.ok].iterrows():
            ext.setdefault(r["world"], {})[f'{r["tag"]}|{r["level"]}|greedy'] = r["stmt"]
        cells = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False)
        res = cells.groupby(["statement", "module"])[RULES + ["rejected"]].mean().round(2)
        harm = cells.assign(h=cells.R2g > cells.R0_data + 1.0).groupby(["statement", "module"]).h.mean().round(3)
        res["harm_R2g"] = harm
        res["truth"] = [str(TRUTH[m]) for _, m in res.index]
        res["stmt"] = [str(ext[m][s]) for s, m in res.index]
        out["outcome"] = {"|".join(k): v for k, v in res.to_dict("index").items()}
        lines.append("\n=== outcome on the twin (D0 statements; mean clean RMSE MW) ===")
        lines.append(res.to_string())
        avg = cells.groupby("statement")[RULES].mean().round(2)
        out["outcome_world_mean"] = avg.to_dict("index")
        lines.append("\n=== outcome, mean over worlds ===")
        lines.append(avg.to_string())

    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
