"""Hint assembly rule HA and its comparators (docs/research/ccpp-hint-assembly.md, pre-registered 2026-11-04).

Usage: python assemble.py <Folds5x2_pp.xlsx> --out runs/assembly/assembly [--reps 20] [--no-outcome]
Reads S0 answers (runs/*.jsonl, runs/pilot/*.jsonl) and S2 answers (runs/strategy/*.jsonl), D0, T>0 samples.
Cells are labelled "dev" or "validation" (§6); H1-H5 verdicts are printed for validation cells only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from elicit import RANGE, _num  # noqa: E402
from parse_score import TRUTH, agreement, majority, parse, score, to_statement  # noqa: E402

SENS_GATE_PCT = (0.4, 1.2)   # GT output change, % per degC, literature (ccpp-domain-check.md §2)
L2_GT_15C_MW = 343.0         # given in the L2 prompt
LAMBDA = 0.5
# §6: models not seen when the rule was fixed; gpt-oss-20b S0 was seen, its S2 answers were not.
VALIDATION_TAGS = {"gemma3-27b", "llama33-70b"}
VALIDATION_S2_TAGS = {"gpt-oss-20b"}
MECHS = (("gt_limit", "below"), ("inlet_anti_icing_heating", "below"),
         ("inlet_evaporative_cooling", "above"), ("exhaust_or_steam_side_limit", "above"))


def _json(text):
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def s2_parts(obj: dict, level: str) -> dict:
    """Step 1 for one S2 answer: {mechanism: (at_c, gate_ok)}; only mechanisms claimed active and in range."""
    parts = {}
    g = obj.get("gt_generator_limit") or {}
    if g.get("reached_in_range") is True:
        p15, lim, sens = _num(g.get("gt_mw_at_15c")), _num(g.get("limit_mw")), _num(g.get("gt_mw_per_degc"))
        if level == "L2":
            p15 = L2_GT_15C_MW
        if None not in (p15, lim, sens) and sens != 0 and p15 > 0:
            at = 15.0 - (lim - p15) / abs(sens)
            pct = abs(sens) / p15 * 100
            if RANGE[0] < at < RANGE[1]:
                parts["gt_limit"] = (at, SENS_GATE_PCT[0] <= pct <= SENS_GATE_PCT[1])
    for key, side in MECHS[1:]:
        d = obj.get(key) or {}
        field = "below_c" if side == "below" else "above_c"
        at = _num(d.get(field))
        if d.get("active") is True and at is not None and RANGE[0] < at < RANGE[1]:
            parts[key] = (at, True)  # no checkable sub-quantity for these (§9)
    return parts


def innermost(cands):
    out = []
    for side in ("below", "above"):
        v = [a for s, a in cands if s == side]
        if v:
            out.append((side, max(v) if side == "below" else min(v)))
    return out


CORROBORATE_C = 3.0  # same tolerance as structural scoring


def assemble_v1(s0_stmts, s2_part_list, use_gate=True):
    """HA as first registered (§3 v1). Kept for reporting: S0 fills any side S2 left empty, which turned an S2
    "no boundary" answer into "no information" (amendment 2026-11-04, §3.1)."""
    kept = []
    k = len(s2_part_list)
    for mech, side in MECHS:
        ats = [p[mech][0] for p in s2_part_list if mech in p and (p[mech][1] or not use_gate)]
        if k and len(ats) * 2 > k:
            kept.append((side, float(np.median(ats))))
    sides = {s for s, _ in kept}
    if s0_stmts:
        kept += [(s, a) for s, a in majority(s0_stmts) if s not in sides]
    return innermost(kept)


def assemble(s0_stmts, s2_part_list, use_gate=True):
    """HA v2, steps 2(a)-3 (§3.1). Step 2(b) (window check) and step 4 (blend) happen in risk_limits.run.

    - A part with a checkable sub-quantity (cap: GT sensitivity) stands alone once it passes the gate.
    - A part with nothing to check (anti-icing, evaporative cooling, exhaust limit) is kept only when the
      independently elicited S0 majority names the same side within CORROBORATE_C.
    - Without S2 answers the S0 majority is used as is.
    """
    if not s2_part_list:
        return majority(s0_stmts) if s0_stmts else []
    s0m = dict(majority(s0_stmts)) if s0_stmts else {}
    kept = []
    k = len(s2_part_list)
    for mech, side in MECHS:
        ats = [p[mech][0] for p in s2_part_list if mech in p and (p[mech][1] or not use_gate)]
        if len(ats) * 2 <= k:
            continue
        at = float(np.median(ats))
        if mech == "gt_limit" or (side in s0m and abs(s0m[side] - at) <= CORROBORATE_C):
            kept.append((side, at))
    return innermost(kept)


def load(root):
    s0, s2 = defaultdict(list), defaultdict(list)
    for path in sorted(glob.glob(str(root / "*.jsonl")) + glob.glob(str(root / "pilot/*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if (r["world"] not in TRUTH or r.get("data", "D0") != "D0" or r["level"] not in ("L1", "L2")
                    or r["temperature"] == 0 or r["error"]):
                continue
            p = parse(r["response"])
            if p["ok"]:
                s0[(r.get("tag", r["model"]), r["level"], r["world"])].append(to_statement(p["boundaries"]))
    for path in sorted(glob.glob(str(root / "strategy/*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if (r["strategy"] != "S2" or r["world"] not in TRUTH or r["level"] not in ("L1", "L2")
                    or r["temperature"] == 0 or r["error"]):
                continue
            obj = _json(r["response"])
            if obj is not None:
                s2[(r.get("tag", r["model"]), r["level"], r["world"])].append(s2_parts(obj, r["level"]))
    return s0, s2


def split_of(tag, has_s2):
    if tag in VALIDATION_TAGS or (tag in VALIDATION_S2_TAGS and has_s2):
        return "validation"
    return "dev"


def build(s0, s2):
    """Statements per (world, name) and per-statement lambdas for AGR."""
    ext, lam, meta = defaultdict(dict), defaultdict(dict), {}
    for key in sorted(set(s0) | set(s2)):
        tag, level, world = key
        a, b = s0.get(key, []), s2.get(key, [])
        cell = f"{tag}|{level}"
        meta[cell] = {"tag": tag, "level": level, "split": split_of(tag, any(s2.get((tag, level, w)) for w in TRUTH))}
        rules = {"HA": assemble(a, b), "HA-gate": assemble(a, b, use_gate=False), "HA_v1": assemble_v1(a, b)}
        if a:
            m = majority(a)
            rules["S0maj"] = rules["AGR"] = m
            lam[world][f"{cell}|AGR"] = float(np.mean([agreement(s, m) for s in a]))
        if b:
            rules["S2raw"] = majority([innermost([(side, p[mech][0]) for mech, side in MECHS if mech in p]) for p in b])
        for name, st in rules.items():
            ext[world][f"{cell}|{name}"] = st
    for world, truth in TRUTH.items():
        ext[world]["ORACLE"] = truth
    return ext, lam, meta


def gate_table(s2, meta):
    rows = []
    for (tag, level, world), plist in s2.items():
        truth_cold = [a for s, a in TRUTH[world] if s == "below"]
        for p in plist:
            if "gt_limit" not in p:
                continue
            at, ok = p["gt_limit"]
            correct = bool(truth_cold) and abs(at - truth_cold[0]) <= 3.0
            rows.append({"split": meta[f"{tag}|{level}"]["split"], "tag": tag, "gate": ok, "cap_part_correct": correct})
    return pd.DataFrame(rows)


def verdicts(res, gt):
    v = res[res.split == "validation"]
    if v.empty:
        return {"status": "no validation cells yet"}
    piv = v.pivot_table(index="cell", columns="rule", values="rmse")
    out = {"n_cells": int(len(piv))}
    out["H1"] = bool((piv["HA"] <= piv["R0"] + 0.10).all())
    if "S0maj" in piv:
        out["H2"] = int((piv["HA"] < piv["S0maj"]).sum()) >= 4
        out["H5"] = int((piv["AGR"] < piv["S0maj"]).sum()) <= 3
    out["H4"] = int((piv["HA"] <= piv["HA-gate"]).sum()) >= 4
    g = gt[gt.split == "validation"] if len(gt) else gt
    if len(g) and g.gate.nunique() == 2:
        diff = g[g.gate].cap_part_correct.mean() - g[~g.gate].cap_part_correct.mean()
        out["H3_diff"], out["H3"] = round(float(diff), 3), bool(diff >= 0.15)
    out["cells_HA_vs"] = piv.round(2).to_dict("index")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs/assembly/assembly"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-outcome", action="store_true")
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    ext, lam, meta = build(s0, s2)
    lines, out = [], {"meta": meta}

    struct = []
    for world, d in ext.items():
        for name, st in d.items():
            if name == "ORACLE":
                continue
            tag, level, rule = name.split("|")
            struct.append({"cell": f"{tag}|{level}", "rule": rule, "world": world, "ok": score(world, st)["all_correct"]})
    sd = pd.DataFrame(struct)
    stab = sd.pivot_table(index="cell", columns="rule", values="ok", aggfunc="mean").round(2)
    stab.insert(0, "split", [meta[c]["split"] for c in stab.index])
    out["structure"] = stab.to_dict("index")
    lines += ["=== structural p of assembled statements (share of 5 worlds all-correct) ===", stab.to_string()]

    gt = gate_table(s2, meta)
    if len(gt):
        gtab = gt.groupby(["split", "tag", "gate"]).cap_part_correct.agg(["mean", "size"]).round(2)
        out["gate"] = {"|".join(map(str, k)): v for k, v in gtab.to_dict("index").items()}
        lines += ["\n=== cap part: correct (within 3 degC of truth) by sensitivity gate ===", gtab.to_string()]

    if not a.no_outcome:
        from ccpp_twin import load_ccpp_inputs
        from risk_limits import run
        cells = run(load_ccpp_inputs(a.xlsx), a.reps, external=ext, builtin=False, external_lambda=lam)
        rows = []
        for name, g in cells.groupby("statement"):
            if name == "ORACLE":
                rows.append({"cell": "ORACLE", "rule": "ORACLE", "rmse": g.R2g.mean(), "harm": np.nan})
                continue
            tag, level, rule = name.split("|")
            col = "blend_custom" if rule == "AGR" else f"blend_{LAMBDA}"
            rows.append({"cell": f"{tag}|{level}", "rule": rule, "rmse": g[col].mean(),
                         "harm": float((g[col] > g.R0_data + 1.0).mean())})
            if rule == "HA":
                rows.append({"cell": f"{tag}|{level}", "rule": "R0", "rmse": g.R0_data.mean(), "harm": 0.0})
                rows.append({"cell": f"{tag}|{level}", "rule": "HA_budget5", "rmse": g.budget_5.mean(),
                             "harm": float((g.budget_5 > g.R0_data + 1.0).mean())})
        res = pd.DataFrame(rows)
        res["split"] = [meta.get(c, {}).get("split", "-") for c in res.cell]
        rt = res.pivot_table(index=["split", "cell"], columns="rule", values="rmse").round(2)
        ht = res.pivot_table(index=["split", "cell"], columns="rule", values="harm").round(3)
        out["rmse"] = {"|".join(k): v for k, v in rt.to_dict("index").items()}
        out["harm"] = {"|".join(k): v for k, v in ht.to_dict("index").items()}
        out["oracle_rmse"] = round(float(res[res.rule == "ORACLE"].rmse.iloc[0]), 2)
        lines += ["\n=== mean clean RMSE over 5 worlds (MW); blend 0.5 unless noted, AGR uses its own lambda ===",
                  rt.to_string(), f"ORACLE (truth, R2g): {out['oracle_rmse']}",
                  "\n=== harm rate (> 1 MW worse than R0) ===", ht.to_string()]
        out["verdicts"] = verdicts(res[res.cell != "ORACLE"], gt)
        lines += ["\n=== pre-registered verdicts (validation cells only) ===",
                  json.dumps({k: v for k, v in out["verdicts"].items() if k != "cells_HA_vs"}, indent=1)]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
