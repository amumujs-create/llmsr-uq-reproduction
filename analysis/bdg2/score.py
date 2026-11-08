"""Score the BDG2 pre-registration (bdg2-regime-extrapolation.md §6-§7). Run once after collection.

Usage:
  PYTHONSAFEPATH=1 python3 analysis/bdg2/score.py data/bdg2 --bins-only   # before collection: record information-bin counts
  PYTHONSAFEPATH=1 python3 analysis/bdg2/score.py data/bdg2                   # after collection: full scoring
Reads analysis/bdg2/worlds.csv, descriptions.json, runs/bdg2/<tag>.jsonl; writes runs/bdg2/score.{txt,json}.
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
sys.path.insert(0, str(HERE.parent / "llm_statements"))
from extrap import HARM_REL, run  # noqa: E402
from parse_score import majority, parse, to_statement  # noqa: E402
from worlds import daily_load, daily_temperature, series  # noqa: E402

TAGS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")
VALIDATION_TAGS = ("gemma3-27b", "llama33-70b", "gpt-oss-20b")
LEVELS = ("L1", "L2")
TOL_C = 3.0
NEED = {"B1": 6, "B3": 4, "B4": 4, "B5": 6, "B6a": 4, "B7": 2}
RULE_COLS = ("R0", "R_cp", "ORACLE", "ORACLE_half", "D", "D_site")


def load_answers(runs, tag):
    """{(level, desc_id): [statement, ...]} from one model's jsonl (parse failures -> empty statement)."""
    out = defaultdict(list)
    path = runs / f"{tag}.jsonl"
    if not path.exists():
        return {}
    for line in open(path):
        r = json.loads(line)
        p = parse(r["response"] if r["error"] is None else "")
        out[(r["level"], r["desc_id"])].append(to_statement(p["boundaries"]) if p["ok"] else [])
    return dict(out)


def below_of(stmt):
    """The cold-side boundary of a majority statement, or None."""
    for side, at in stmt:
        if side == "below":
            return float(at)
    return None


def llm_statements(descriptions, answers):
    """{cell_name: {meter: (x_c, factor)}} for S0maj, HA (same statement, scored with/without guard+blend is not
    separable here: S0maj is the raw majority with factor 0, HA the same with guard and blend) and HA_half; plus M."""
    cells = {}
    for tag, ans in answers.items():
        m_stmt = majority(ans.get(("M", "M"), [[]]))
        x_m = below_of(m_stmt)
        cells[f"{tag}|M"] = {}
        for level in LEVELS:
            per = {}
            for did, d in descriptions[level].items():
                x = below_of(majority(ans.get((level, did), [[]])))
                for meter in d["meters"]:
                    per[meter] = (x, 0.0)
                    cells[f"{tag}|M"][meter] = (x_m, 0.0)
            cells[f"{tag}|{level}|HA"] = per
            cells[f"{tag}|{level}|HA_half"] = {k: (v[0], 0.5) for k, v in per.items()}
    return cells


BINS = ("evidence", "outside", "inside_hidden")


def add_bins(df):
    """Information bins (§3, amended 2026-11-08 before collection): evidence = I > 0 (the window shows the kink);
    outside = I <= 0 and the reference kink is below the window (delta > 0); inside_hidden = I <= 0 but the kink is
    inside the window (delta <= 0)."""
    df = df.copy()
    df["I_bin"] = np.where(df.I > 0, "evidence", np.where(df.delta > 0, "outside", "inside_hidden"))
    return df


def bin_counts(df):
    return {b: int((df.I_bin == b).sum()) for b in BINS}


def summarize(df, rule_cols):
    """Relative RMSE (rule / R0), harm rate, filled-gap ratio; overall and by I tercile."""
    out = {}
    gap = (df.R0 - df.ORACLE)
    groups = {b: g for b, g in df.groupby("I_bin")}
    for col in rule_cols:
        if col not in df:
            continue
        rel = df[col] / df.R0
        filled = ((df.R0 - df[col]) / gap)[gap > 0]
        rec = {"rel_mean": float(rel.mean()), "rel_gmean": float(np.exp(np.log(rel).mean())),
               "harm": float((rel > HARM_REL).mean()), "filled": float(filled.mean()) if len(filled) else np.nan}
        for b in BINS:
            g = groups.get(b)
            rec[f"rel_{b}"] = float((g[col] / g.R0).mean()) if g is not None and len(g) else np.nan
        out[col] = rec
    return pd.DataFrame(out).T


def structure(descriptions, answers, worlds):
    """|S0maj boundary - tc| per (tag, level, meter) and the share of empty statements."""
    tc = worlds.set_index("meter").tc
    rows = []
    for tag, ans in answers.items():
        for level in LEVELS:
            for did, d in descriptions[level].items():
                x = below_of(majority(ans.get((level, did), [[]])))
                for meter in d["meters"]:
                    rows.append(dict(tag=tag, level=level, meter=meter, x=x,
                                     abs_err=np.nan if x is None else abs(x - tc[meter])))
    return pd.DataFrame(rows)


def verdicts(df, summary_by_cell, struct):
    """B1-B7 on the validation cells (3 models x L1/L2)."""
    cells = [f"{t}|{lv}" for t in VALIDATION_TAGS for lv in LEVELS]
    have = [c for c in cells if f"{c}|HA" in summary_by_cell]
    if len(have) < len(cells):
        return {"status": f"validation cells missing: {sorted(set(cells) - set(have))}"}
    s = summary_by_cell
    fixed = s["fixed"]
    p_mean = float(np.mean([fixed.loc[c, "rel_mean"] for c in fixed.index if c.startswith("P|")]))
    wins = {"B1": sum(s[f"{c}|HA"].loc["HA", "harm"] <= 0.05 for c in cells),
            "B3": sum(s[f"{c}|HA"].loc["HA", "rel_mean"] < fixed.loc["D", "rel_mean"] for c in cells),
            "B4_M": sum(s[f"{c}|HA"].loc["HA", "rel_mean"] < s[f"{c.split('|')[0]}|M"].loc["M", "rel_mean"] for c in cells),
            "B4_P": sum(s[f"{c}|HA"].loc["HA", "rel_mean"] < p_mean for c in cells),
            "B5": sum(s[f"{c}|HA"].loc["HA", "rel_outside"] < s[f"{c}|HA"].loc["HA", "rel_evidence"] for c in cells),
            "B6a": sum(fixed.loc["R_cp", "rel_evidence"] <= s[f"{c}|HA"].loc["HA", "rel_evidence"] for c in cells)}
    b2 = bool(fixed.loc["D", "filled"] < 0.5)
    b6b = bool(fixed.loc["R_cp", "rel_evidence"] < 1.0)   # data evidence yields a gain where it exists
    b7 = 0
    for tag in VALIDATION_TAGS:
        g = struct[struct.tag == tag]
        m1, m2 = g[g.level == "L1"].abs_err.median(), g[g.level == "L2"].abs_err.median()
        b7 += int(np.isfinite(m1) and np.isfinite(m2) and m2 < m1)
    out = {"n_cells": len(cells), "wins": wins, "P_mean_rel": p_mean,
           "B1": wins["B1"] >= NEED["B1"], "B2": b2, "B3": wins["B3"] >= NEED["B3"],
           "B4": wins["B4_M"] >= NEED["B4"] and wins["B4_P"] >= NEED["B4"], "B5": wins["B5"] >= NEED["B5"],
           "B6": wins["B6a"] >= NEED["B6a"] and b6b, "B6b_Rcp_gain_with_evidence": b6b, "B7": b7 >= NEED["B7"], "B7_wins": b7}
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--worlds", default=str(HERE / "worlds.csv"))
    ap.add_argument("--runs", default=str(HERE / "runs/bdg2"))
    ap.add_argument("--bins-only", action="store_true", help="before collection: record the information-bin counts")
    a = ap.parse_args()
    worlds = pd.read_csv(a.worlds)
    temps, daily = daily_temperature(a.data_dir), daily_load(a.data_dir)
    series_of = lambda m: (lambda df: (df.t.values, df.y.values))(series(daily, temps, m))  # noqa: E731
    runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=True)
    if a.bins_only:
        df = add_bins(run(worlds, series_of))
        info = {"bins": bin_counts(df), "n_rows": int(len(df)), "n_worlds": int(worlds.meter.nunique()),
                "share_I_positive": float((df.I > 0).mean()), "date": "2026-11-08"}
        json.dump(info, open(HERE / "i_bins.json", "w"), indent=1)
        print(info)
        return
    descriptions = json.load(open(HERE / "descriptions.json"))
    answers = {tag: ans for tag in TAGS if (ans := load_answers(runs, tag))}
    cells = llm_statements(descriptions, answers)
    df = add_bins(run(worlds, series_of, cells))
    fixed_cols = list(RULE_COLS) + [c for c in df if c.startswith("P|")]
    by_cell = {"fixed": summarize(df, fixed_cols)}
    for tag in answers:
        by_cell[f"{tag}|M"] = summarize(df.rename(columns={f"{tag}|M": "M"}), ["M"])
        for lv in LEVELS:
            by_cell[f"{tag}|{lv}|HA"] = summarize(df.rename(columns={f"{tag}|{lv}|HA": "HA", f"{tag}|{lv}|HA_half": "HA_half"}),
                                                  ["HA", "HA_half"])
    struct = structure(descriptions, answers, worlds)
    v = verdicts(df, by_cell, struct)
    lines = ["BDG2 chilled-water extrapolation (bdg2-regime-extrapolation.md); relative RMSE = rule / R0", ""]
    for k, tab in by_cell.items():
        lines += [f"=== {k} ===", tab.round(4).to_string(), ""]
    if len(struct):
        lines += ["=== structure: median |S0maj - tc| and empty share by tag/level ===",
                  struct.groupby(["tag", "level"]).agg(abs_err_median=("abs_err", "median"),
                                                       empty=("x", lambda s: float(s.isna().mean()))).round(3).to_string(), ""]
    lines += ["=== verdicts ===", json.dumps(v, indent=1, default=str)]
    text = "\n".join(lines)
    (runs / "score.txt").write_text(text)
    df.drop(columns=[]).to_csv(runs / "cells.csv", index=False)
    json.dump({"by_cell": {k: t.round(5).to_dict("index") for k, t in by_cell.items()}, "verdicts": v},
              open(runs / "score.json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
