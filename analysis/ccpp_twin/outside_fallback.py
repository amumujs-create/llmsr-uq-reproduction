"""HA_fb on the outside-regime tasks (docs/research/ccpp-twin-outside-fallback.md).

HA_fb = HA, plus: a device most S2 answers claim present, on a side where HA has no boundary, gets the default
position (below 8 / above 27 degC). Development-set verdict: the rule was designed on these same answers.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/outside_fallback.py <Folds5x2_pp.xlsx>
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_statements"))
import outside_regime as orr  # noqa: E402
from ccpp_twin import load_ccpp_inputs  # noqa: E402
from assemble import MECHS, _json, load  # noqa: E402
from controls import build_external  # noqa: E402
from parse_score import TRUTH  # noqa: E402

SEED_PREFIX = "outside2"
DEFAULT_AT = {"below": 8.0, "above": 27.0}          # the D_real numbers
CLAIM_KEY = {"gt_limit": ("gt_generator_limit", "reached_in_range"),
             **{m: (m, "active") for m, _ in MECHS[1:]}}
SIDE = dict(MECHS)


def claims(obj):
    """Devices one S2 answer says are present / reached."""
    return {m for m, (key, field) in CLAIM_KEY.items() if (obj.get(key) or {}).get(field) is True}


def claim_share(objs):
    return {m: (sum(m in claims(o) for o in objs) / len(objs) if objs else 0.0) for m in CLAIM_KEY}


def fallback(ha_stmt, objs, mechs=tuple(CLAIM_KEY)):
    """HA statement plus the default position on every side a majority-claimed device points to and HA left empty."""
    if not objs:
        return list(ha_stmt)
    share = claim_share(objs)
    out = list(ha_stmt)
    sides = {s for s, _ in out}
    for m in mechs:
        side = SIDE[m]
        if share[m] > 0.5 and side not in sides:
            out.append((side, DEFAULT_AT[side]))
            sides.add(side)
    return sorted(out, key=lambda t: t[0], reverse=True)   # "below" first, as assemble.innermost


def load_s2_raw(root):
    """Raw S2 answer objects per (tag, level, world), with the same filters as assemble.load."""
    raw = defaultdict(list)
    for path in sorted(glob.glob(str(Path(root) / "strategy/*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if (r["strategy"] != "S2" or r["world"] not in TRUTH or r["level"] not in ("L1", "L2")
                    or r["temperature"] == 0 or r["error"]):
                continue
            obj = _json(r["response"])
            if obj is not None:
                raw[(r.get("tag", r["model"]), r["level"], r["world"])].append(obj)
    return raw


def with_fallback(ext, raw, mechs=tuple(CLAIM_KEY)):
    """Copy of ext with every |HA statement replaced by its fallback version."""
    out = {}
    for world, st in ext.items():
        out[world] = dict(st)
        for name, stmt in st.items():
            if name.endswith("|HA"):
                tag, level = name[:-3].split("|")
                out[world][name] = fallback(stmt, raw.get((tag, level, world), []), mechs)
    return out


def cell_summary(df):
    tm = orr.task_means(df)
    cols = [f"{c}|HA" for c in orr.VALIDATION + orr.DEVELOPMENT if f"{c}|HA" in df.columns]
    return pd.DataFrame({"kink": orr.macro(tm, orr.KINK_TASKS)[cols], "all": orr.macro(tm, orr.TASKS)[cols],
                         "harm": orr.harm_share(df)[cols], "harm_nokink": orr.harm_share(df, orr.NOKINK_TASKS)[cols]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE.parent / "llm_statements" / "runs"))
    ap.add_argument("--out", default=str(HERE / "runs" / "outside_fallback"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s0, s2 = load(Path(a.runs))
    ext, meta, _ = build_external(s0, s2)
    raw = load_s2_raw(a.runs)
    x = load_ccpp_inputs(a.xlsx)
    variants = {"HA_fb": with_fallback(ext, raw), "HA": ext, "HA_fb_gt": with_fallback(ext, raw, ("gt_limit",))}
    frames = {name: orr.run(x, e, reps=a.reps, workers=a.workers, prefix=SEED_PREFIX) for name, e in variants.items()}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames["HA_fb"].to_csv(out_dir / "cells.csv.gz", index=False)
    v, text = orr.summarize(frames["HA_fb"], variants["HA_fb"])
    L = [text.replace("Outside-regime run", "HA_fb run (development-set verdict, not a confirmation). Outside-regime"),
         "=== report only: same seeds, three assemblies (kink mean / all mean / harm share / no-kink harm) ==="]
    for name, df in frames.items():
        L += [f"--- {name} ---", cell_summary(df).round(3).to_string()]
    L.append("\n=== share of S2 answers claiming each device, by cell and world ===")
    worlds = sorted({t[0] for t in orr.TASKS.values()})
    for c in orr.VALIDATION + orr.DEVELOPMENT:
        tag, level = c.split("|")
        for w in worlds:
            sh = claim_share(raw.get((tag, level, w), []))
            L.append(f"{c:28s} {w:15s} n={len(raw.get((tag, level, w), [])):2d} "
                     + " ".join(f"{m}={sh[m]:.2f}" for m in CLAIM_KEY))
    text = "\n".join(L) + "\n"
    json.dump({"verdicts": v, "cells": meta,
               "report": {n: cell_summary(df).round(4).to_dict() for n, df in frames.items()}},
              open(out_dir / "summary.json", "w"), indent=1, default=str)
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
