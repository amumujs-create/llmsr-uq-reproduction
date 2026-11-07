"""Anti-icing stage scoring: Q1-Q5 (docs/research/ccpp-anti-icing-world.md §7, details fixed in §16 before scoring).

Usage: python score_ice.py <Folds5x2_pp.xlsx> [--reps 20] [--no-outcome]
Reads runs/ice/<tag>.jsonl (and runs/gpt-oss-120b.jsonl for Q4). Writes runs/ice_score/score.{txt,json}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from parse_score import TOL_C, parse, parse_mech, to_statement  # noqa: E402
from prompts import ICE_WORLDS  # noqa: E402

TAGS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")
BIG = "gpt-oss-120b"
ICE_AT, CAP_AT = 7.0, 6.0          # truth (anti-icing §3, §4)
Q4_BAND_C = 8.0                    # "anti-icing band" cold false alarm (§7 Q4)


def null_location(text):
    """True when the answer is valid JSON whose boundaries include one with at_c null (§16: counted separately)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group(0))["boundaries"] if m else None
    except (ValueError, KeyError, TypeError):
        return False
    return isinstance(raw, list) and any(isinstance(b, dict) and b.get("at_c") is None for b in raw)


def ai_ok(items):
    return any(i["mechanism"] == "anti_icing" and i["side"] == "below" and abs(i["at_c"] - ICE_AT) <= TOL_C
               for i in items)


def cap_ok(items):
    return any(i["mechanism"] == "power_cap" and i["side"] == "below" and abs(i["at_c"] - CAP_AT) <= TOL_C
               for i in items)


def load_ice(runs):
    rows = []
    for tag in TAGS:
        for line in open(runs / "ice" / f"{tag}.jsonl"):
            r = json.loads(line)
            text = r["response"] if r["error"] is None else ""
            ice = r["world"] in ICE_WORLDS
            p = parse_mech(text) if ice else parse(text)
            items = p["items"] if ice and p["ok"] else []
            stmt = to_statement(p["boundaries"]) if (not ice and p["ok"]) else []
            rows.append({"tag": tag, "world": r["world"], "level": r["level"], "data": r["data"],
                         "setting": r["setting"], "temperature": r["temperature"], "ok": p["ok"],
                         "null_loc": (not p["ok"]) and null_location(text), "items": items, "stmt": stmt})
    return pd.DataFrame(rows)


def cold_band_alarm(stmt):
    return any(s == "below" and a <= Q4_BAND_C for s, a in stmt)


def q4_old(runs):
    out = []
    for line in open(runs / f"{BIG}.jsonl"):
        r = json.loads(line)
        if r["world"] in ("base", "evap") and r["level"] in ("L1", "L2") and r["data"] == "D0":
            p = parse(r["response"] if r["error"] is None else "")
            out.append(cold_band_alarm(to_statement(p["boundaries"])) if p["ok"] else False)
    return out


def mech_majority(item_lists):
    """§16 Q5: keep a (side, mechanism) named by more than half of the samples; location = median;
    rh_above = median when more than half of those items give one."""
    n = len(item_lists)
    groups = {}
    for items in item_lists:
        for it in items:
            groups.setdefault((it["side"], it["mechanism"]), []).append(it)
    out = []
    for (side, mech), its in sorted(groups.items()):
        if len(its) * 2 <= n:
            continue
        rh = [i["rh_above"] for i in its if i["rh_above"] is not None]
        out.append({"side": side, "mechanism": mech, "at_c": float(np.median([i["at_c"] for i in its])),
                    "rh_above": float(np.median(rh)) if len(rh) * 2 > len(its) else None})
    return out


def structure(df, old):
    res, lines = {}, []
    ice = df[df.world.isin(ICE_WORLDS)]
    # Q1
    g = ice[(ice.world == "ice") & (ice.level == "L1")]
    q1 = g.groupby("tag").apply(lambda d: float(np.mean([any(i["mechanism"] == "anti_icing" for i in it)
                                                         for it in d["items"]]))).to_dict()
    small = float(np.mean([v for t, v in q1.items() if t != BIG]))
    res["Q1"] = {"mention": q1, "small_mean": small, "n": int(len(g) / len(TAGS)),
                 "pass": bool(q1[BIG] >= 0.5 and small < 0.3)}
    # Q2
    g = ice[ice.level == "L2"].copy()
    g["ai"] = g["items"].apply(ai_ok)
    g["falls"] = g["items"].apply(lambda it: any(i["mechanism"] == "anti_icing" and i["side"] == "below"
                                                 and abs(i["at_c"] - ICE_AT) <= TOL_C and i["output_trend"] == "falls"
                                                 for i in it))
    q2 = {t: {"n_ai": int(d.ai.sum()), "falls_given_ai": (float(d.falls.sum() / d.ai.sum()) if d.ai.sum() else None)}
          for t, d in g.groupby("tag")}
    judged = [v["falls_given_ai"] for v in q2.values() if v["falls_given_ai"] is not None]
    res["Q2"] = {"by_model": q2, "pass": (bool(all(v < 0.5 for v in judged)) if judged else None),
                 "excluded": [t for t, v in q2.items() if v["falls_given_ai"] is None]}
    # Q3
    g = ice[(ice.world == "ice+cap_mild") & (ice.level == "L1")]
    both = g["items"].apply(lambda it: ai_ok(it) and cap_ok(it))
    strict = g["items"].apply(lambda it: ai_ok(it) and cap_ok(it) and not any(i["side"] == "above" for i in it))
    q3 = both.groupby(g.tag).mean().to_dict()
    res["Q3"] = {"p_both": q3, "p_strict": strict.groupby(g.tag).mean().to_dict(),
                 "pass": bool(all(v < 0.3 for v in q3.values()))}
    # Q4
    g = df[(df.tag == BIG) & df.world.isin(["base", "evap"]) & df.level.isin(["L1", "L2"]) & (df.data == "D0")]
    new = g.apply(lambda r: r.ok and cold_band_alarm(r.stmt), axis=1)
    old_rate, new_rate = float(np.mean(old)), float(new.mean())
    res["Q4"] = {"old": old_rate, "new": new_rate, "n_old": len(old), "n_new": int(len(new)),
                 "pass": (bool(new_rate <= 0.5 * old_rate) if old_rate > 0 else None)}
    # Extra: null-location answers, anti-icing correctness and humidity condition
    res["null_location"] = (ice[ice.null_loc].groupby(["tag", "world", "level", "data"]).size()
                            .rename("n").reset_index().to_dict("records"))
    res["parse_fail"] = df.groupby("tag").ok.apply(lambda s: int((~s).sum())).to_dict()
    ai_rate = ice.assign(ai=ice["items"].apply(ai_ok)).groupby(["world", "level", "tag"]).ai.mean().unstack("tag")
    rh = ice["items"].apply(lambda it: next((i["rh_above"] for i in it if i["mechanism"] == "anti_icing"
                                             and i["side"] == "below"), np.nan))
    res["ai_correct_rate"] = {f"{w}|{lv}": v for (w, lv), v in ai_rate.round(3).to_dict("index").items()}
    lines += [f"Q1 ice L1 anti_icing mention: {json.dumps({k: round(v, 3) for k, v in q1.items()})}; "
              f"small mean {small:.3f} -> {res['Q1']['pass']}",
              f"Q2 L2 falls | correct anti-icing: {json.dumps(q2)} -> {res['Q2']['pass']}",
              f"Q3 ice+cap_mild L1 both causes: {json.dumps({k: round(v, 3) for k, v in q3.items()})}; strict "
              f"{json.dumps({k: round(v, 3) for k, v in res['Q3']['p_strict'].items()})} -> {res['Q3']['pass']}",
              f"Q4 120b cold alarm <= {Q4_BAND_C:g}C, base+evap L1+L2 D0: old {old_rate:.3f} (n={len(old)}) "
              f"new {new_rate:.3f} (n={len(new)}) -> {res['Q4']['pass']}",
              "", "anti-icing boundary correct (share of samples)", ai_rate.round(2).to_string(),
              "", "rh_above given with a below anti-icing item (median by tag)",
              rh.groupby(ice.tag).median().to_string(),
              "", f"parse failures by tag: {res['parse_fail']}", "null-location answers (counted as failures):",
              pd.DataFrame(res["null_location"]).to_string(index=False) if res["null_location"] else "  none"]
    return res, lines


def q5_statements(df):
    from regime_rules import from_mechanisms
    out, info = {}, []
    g = df[(df.world == "ice") & (df.data == "D0")]
    for (tag, level), d in g.groupby(["tag", "level"]):
        maj = mech_majority(list(d["items"]))
        use = ai_ok(maj)
        info.append({"tag": tag, "level": level, "majority": maj, "used": use})
        if use:
            out[f"{tag}|{level}"] = from_mechanisms(maj)
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-outcome", action="store_true")
    a = ap.parse_args()
    runs = Path(a.runs)
    df = load_ice(runs)
    res, lines = structure(df, q4_old(runs))
    stmts, info = q5_statements(df)
    res["Q5_statements"] = info
    lines += ["", "Q5 ice D0 mechanism-majority statements:"] + [
        f"  {i['tag']}|{i['level']}: {i['majority']} used={i['used']}" for i in info]
    if not a.no_outcome:
        if not stmts:
            res["Q5"] = {"pass": None, "note": "no majority statement with a correct anti-icing boundary"}
        else:
            from ccpp_twin import load_ccpp_inputs
            from risk_limits import run
            cells = run(load_ccpp_inputs(a.xlsx), a.reps, external={"ice": stmts}, builtin=False, include_ice=True)
            cells = cells[cells.module == "ice"]
            tab = cells.groupby(["setting", "statement"])[["R0_data", "blend_0.5", "R2g"]].mean()
            win = tab.groupby(level="setting").mean()
            mc = win.loc["mid->cold"]
            res["Q5"] = {"by_window": win.round(3).to_dict("index"), "by_statement": {
                f"{s}|{st}": v for (s, st), v in tab.round(3).to_dict("index").items()},
                "pass": bool(mc["blend_0.5"] < mc["R0_data"])}
            lines += ["", "Q5 outcome on the ice world (mean clean RMSE MW over used statements)", win.round(3).to_string(),
                      "", "by statement", tab.round(3).to_string(),
                      f"Q5 mid->cold blend_0.5 {mc['blend_0.5']:.3f} < R0 {mc['R0_data']:.3f} -> {res['Q5']['pass']}"]
    lines += ["", "=== verdicts ===", json.dumps({q: res[q]["pass"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5") if q in res})]
    dest = runs / "ice_score"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "score.txt").write_text("\n".join(lines) + "\n")
    json.dump(res, open(dest / "score.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
