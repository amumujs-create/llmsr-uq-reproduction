"""Real-CCPP validation (A7) with the existing L0 answers (docs/research/ccpp-real-data.md, pre-registered 2cf65af).

Usage: python real_validation.py <Folds5x2_pp.xlsx> [--reps 20]
Writes runs/real/real.{txt,json}. Verdicts R1-R4 are printed once at the end.
"""
from __future__ import annotations

import argparse
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
from parse_score import majority, parse, to_statement  # noqa: E402

TAGS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")
COL = "blend_0.5"
CRASH_MW, HARM_MW = 4.3, 1.0
# §7: word-level search; "UCI" alone would match inside "reducing".
RECOGNITION = re.compile(r"\b(UCI|T[üu]fekci|Kaya|Folds5x2|Machine Learning Repository)\b", re.I)


def load_pe(path):
    import openpyxl
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]
    i = header.index("PE")
    return np.array([float(r[i]) for r in rows if r[0] is not None])


def real_answers(runs):
    """{tag: {(data, setting): [statement, ...]}} plus recognition hits per tag."""
    out, hits = {}, {}
    for tag in TAGS:
        d, h = defaultdict(list), []
        for line in open(runs / f"{tag}.jsonl"):
            r = json.loads(line)
            if r["world"] != "real":
                continue
            text = r["response"] if r["error"] is None else ""
            for m in RECOGNITION.finditer(f'{text} {r.get("reasoning") or ""}'):
                h.append(m.group(0))
            p = parse(text)
            d[(r["data"], r["setting"])].append(to_statement(p["boundaries"]) if p["ok"] else [])
        out[tag], hits[tag] = dict(d), h
    return out, hits


def build_statements(ans):
    stmts = {}
    for tag, d in ans.items():
        stmts[f"{tag}|S0maj"] = majority(d[("D0", None)])
        for (data, setting), v in d.items():
            if data == "D1":
                stmts[f"{tag}|D1maj|{setting}"] = majority(v)
    stmts["POOL|POOL"] = majority([stmts[f"{t}|S0maj"] for t in TAGS])
    return stmts


def rows_for_rules(cells):
    """One frame per (rule label, row) with the rule's RMSE; D1maj rows only on their own window setting."""
    keep = []
    for name, g in cells.groupby("statement"):
        parts = name.split("|")
        if parts[1] == "D1maj":
            g = g[g.setting == parts[2]]
            label = f"{parts[0]}|D1maj"
        else:
            label = name
        keep.append(g.assign(rule=label))
    return pd.concat(keep, ignore_index=True)


def metrics(df):
    reg = (df[COL] - df.R0_data).to_numpy()
    q = np.quantile(reg, 0.95)
    return {"rmse": float(df[COL].mean()), "R0": float(df.R0_data.mean()), "harm": float((reg > HARM_MW).mean()),
            "CVaR95": float(reg[reg >= q].mean()), "crash": float((reg > CRASH_MW).mean()),
            "rejected": float(df.rejected.mean()), "rows": int(len(df))}


def verdicts(t, by_setting):
    s0 = {tg: t.loc[f"{tg}|S0maj"] for tg in TAGS}
    r0 = float(t["R0"].iloc[0])
    cold = by_setting[by_setting.setting.isin(["hot->cold", "mid->cold"])]
    big = cold[cold.rule == "gpt-oss-120b|S0maj"]
    return {"R1": bool(all(s0[tg]["rmse"] <= r0 + 0.10 for tg in TAGS)),
            "R2": bool(t.loc["POOL|POOL", "rmse"] < r0),
            "R3": bool(all(s0[tg]["crash"] <= 0.05 for tg in TAGS)),
            "R4": bool(big[COL].mean() >= big.R0_data.mean() - 0.10),
            "R4_values": {"S0maj_120b": round(float(big[COL].mean()), 3), "R0": round(float(big.R0_data.mean()), 3)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from ccpp_twin import load_ccpp_inputs
    from risk_limits import run
    ans, hits = real_answers(Path(a.runs))
    stmts = build_statements(ans)
    x = load_ccpp_inputs(a.xlsx)
    pe = load_pe(a.xlsx)
    assert len(pe) == len(x["AT"])
    cells = run(x, a.reps, external={"real": stmts}, observed=pe)
    df = rows_for_rules(cells)
    t = pd.DataFrame({r: metrics(g) for r, g in df.groupby("rule")}).T
    order = [f"{tg}|S0maj" for tg in TAGS] + ["POOL|POOL"] + [f"{tg}|D1maj" for tg in TAGS]
    t = t.loc[[o for o in order if o in t.index]]
    by_setting = df.groupby(["setting", "rule"])[[COL, "R0_data"]].mean().reset_index()
    any_rule = df[df.rule == "POOL|POOL"]
    base = any_rule.groupby("setting")[["R0_data", "coef_prior", "pb_mid"]].mean()
    check = any_rule[(any_rule.setting == "hot->cold") & (any_rule.w == 2.5) & (any_rule.d == 10)].R0_data.mean()
    v = verdicts(t, df)
    lines = [f"rows per rule (S0maj/POOL): {int(t.loc['POOL|POOL', 'rows'])}",
             f"sanity: R0 hot->cold w2.5 d10 = {check:.2f} MW (grid doc: 15.5)", "",
             "statements:", *[f"  {k}: {s}" for k, s in stmts.items()], "",
             f"dataset recognition hits (word-level): { {k: len(h) for k, h in hits.items()} }", "",
             "=== overall (mean RMSE vs observed PE, MW; regret vs R0 for harm/CVaR/crash) ===", t.round(3).to_string(),
             "", "=== by window setting: blend_0.5 (R0 alongside) ===",
             by_setting.pivot(index="rule", columns="setting", values=COL).round(2).to_string(),
             "", "R0 / data-only comparators by setting", base.round(2).to_string(),
             "", "=== pre-registered verdicts (R1-R4) ===", json.dumps(v)]
    dest = HERE / "runs" / "real"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "real.txt").write_text("\n".join(lines) + "\n")
    json.dump({"statements": stmts, "recognition": hits, "overall": t.to_dict("index"),
               "by_setting": by_setting.to_dict("records"), "comparators": base.to_dict("index"),
               "sanity_R0_hot_cold_w2.5_d10": check, "verdicts": v}, open(dest / "real.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
