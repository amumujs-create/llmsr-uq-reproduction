"""Shape stage scoring: P1-P5 (docs/research/ccpp-shape-elicitation.md §4-§6, fixed before data).

Usage: python score_shape.py [--runs runs]
Reads runs/shape/<tag>.jsonl. Writes runs/shape_score/score.{txt,json}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_score import TOL_C, parse_shape  # noqa: E402

TAGS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")
MIN_ITEMS = 10
# §4: mechanism groups -> (worlds, mechanism, side, truth location, correct response)
GROUPS = {
    "cap": (("cap_mild", "cap_strong", "evap+cap_mild"), "power_cap", "below",
            {"cap_mild": 6.0, "cap_strong": 9.0, "evap+cap_mild": 6.0}, "near_zero"),
    "evap": (("evap", "evap+cap_mild"), "evap_cooling", "above", {"evap": 24.0, "evap+cap_mild": 24.0}, "weaker"),
    "ice": (("ice",), "anti_icing", "below", {"ice": 7.0}, "opposite"),
}
INFERRED = ("evap", "ice")


def null_location(text):
    """Valid JSON whose boundaries include one with at_c null (§4: parse failure, reported separately)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group(0))["boundaries"] if m else None
    except (ValueError, KeyError, TypeError):
        return False
    return isinstance(raw, list) and any(isinstance(b, dict) and b.get("at_c") is None for b in raw)


def load(runs):
    rows = []
    for tag in TAGS:
        f = runs / "shape" / f"{tag}.jsonl"
        for line in open(f):
            r = json.loads(line)
            text = r["response"] if r["error"] is None else ""
            p = parse_shape(text)
            rows.append({"tag": tag, "world": r["world"], "level": r["level"], "ok": p["ok"],
                         "error": r["error"] is not None, "null_loc": (not p["ok"]) and null_location(text),
                         "items": p["items"] if p["ok"] else []})
    return rows


def loc_item(items, world, group):
    """The location-correct item for this group in this sample, or None (§4)."""
    worlds, mech, side, at, _ = GROUPS[group]
    if world not in worlds:
        return None
    for it in items:
        if it["mechanism"] == mech and it["side"] == side and abs(it["at_c"] - at[world]) <= TOL_C:
            return it
    return None


def rates(rows, group, levels=("L1", "L2")):
    """(n_samples, n_loc, loc_rate, n_shape_correct, shape_rate, response counts among location-correct items)."""
    worlds, *_, correct = GROUPS[group]
    sel = [r for r in rows if r["world"] in worlds and r["level"] in levels]
    hits = [loc_item(r["items"], r["world"], group) for r in sel]
    hits = [h for h in hits if h is not None]
    good = sum(h["response_beyond"] == correct for h in hits)
    return {"n": len(sel), "n_loc": len(hits), "loc_rate": len(hits) / len(sel) if sel else None,
            "n_shape": good, "shape_rate": good / len(hits) if hits else None,
            "responses": dict(Counter(str(h["response_beyond"]) for h in hits))}


def pooled(rows, groups, levels=("L1", "L2")):
    rs = [rates(rows, g, levels) for g in groups]
    n, n_loc, n_shape = (sum(r[k] for r in rs) for k in ("n", "n_loc", "n_shape"))
    return {"n": n, "n_loc": n_loc, "loc_rate": n_loc / n if n else None,
            "n_shape": n_shape, "shape_rate": n_shape / n_loc if n_loc else None}


def verdicts(rows):
    by = {t: [r for r in rows if r["tag"] == t] for t in TAGS}
    res = {}
    # P1
    cap = {t: rates(by[t], "cap") for t in TAGS}
    judged = {t: v["shape_rate"] for t, v in cap.items() if v["n_loc"] >= MIN_ITEMS}
    need = 4 if len(judged) >= 4 else len(judged)
    res["P1"] = {"by_model": cap, "judged": judged, "excluded": [t for t in TAGS if t not in judged],
                 "pass": (sum(v >= 0.6 for v in judged.values()) >= need) if judged else None}
    # P2 (L2 only)
    ice2 = {t: rates(by[t], "ice", ("L2",)) for t in TAGS}
    judged = {t: v["shape_rate"] for t, v in ice2.items() if v["n_loc"] >= MIN_ITEMS}
    res["P2"] = {"by_model": ice2, "judged": judged, "excluded": [t for t in TAGS if t not in judged],
                 "L1_report": {t: rates(by[t], "ice", ("L1",)) for t in TAGS},
                 "pass": all(v < 0.5 for v in judged.values()) if judged else None}
    # P3
    ev = {t: rates(by[t], "evap") for t in TAGS}
    judged = {t: v["shape_rate"] for t, v in ev.items() if v["n_loc"] >= MIN_ITEMS}
    res["P3"] = {"by_model": ev, "judged": judged, "excluded": [t for t in TAGS if t not in judged],
                 "pass": (sum(v < 0.5 for v in judged.values()) >= 3) if judged else None}
    # P4: inferred groups, shape_rate < loc_rate
    inf = {t: pooled(by[t], INFERRED) for t in TAGS}
    cmp_ = {t: v["shape_rate"] < v["loc_rate"] for t, v in inf.items() if v["shape_rate"] is not None}
    res["P4"] = {"by_model": inf, "shape_lt_loc": cmp_, "pass": sum(cmp_.values()) >= 4}
    # P5: all models pooled
    s_cap, s_inf = pooled(rows, ("cap",))["shape_rate"], pooled(rows, INFERRED)["shape_rate"]
    res["P5"] = {"cap": s_cap, "inferred": s_inf,
                 "pass": (s_cap - s_inf >= 0.3) if None not in (s_cap, s_inf) else None}
    return res


def reports(rows):
    out = {}
    for t in TAGS:
        rs = [r for r in rows if r["tag"] == t]
        base = [r for r in rs if r["world"] == "base"]
        out[t] = {
            "calls": len(rs), "call_errors": sum(r["error"] for r in rs),
            "parse_failures": sum((not r["ok"]) and not r["error"] for r in rs),
            "null_location": sum(r["null_loc"] for r in rs),
            "base_false_alarm_rate": float(np.mean([bool(r["items"]) for r in base])) if base else None,
            "base_responses": dict(Counter(str(i["response_beyond"]) for r in base for i in r["items"])),
            "by_level": {g: {lv: rates(rs, g, (lv,)) for lv in ("L1", "L2")} for g in GROUPS},
            "responses_any_location": {g: dict(Counter(str(i["response_beyond"]) for r in rs
                                                       if r["world"] in GROUPS[g][0] for i in r["items"]
                                                       if i["mechanism"] == GROUPS[g][1]))
                                       for g in GROUPS},
            "evap_near_zero_share": (lambda v: v["responses"].get("near_zero", 0) / v["n_loc"] if v["n_loc"] else None)(
                rates(rs, "evap")),
            "ice_rh_above_median": (lambda v: float(np.median(v)) if v else None)(
                [i["rh_above"] for r in rs if r["world"] == "ice" for i in r["items"]
                 if i["mechanism"] == "anti_icing" and i["rh_above"] is not None]),
            "mechanism_errors": sum(1 for r in rs for g, (ws, mech, side, at, _) in GROUPS.items() if r["world"] in ws
                                    for i in r["items"] if i["mechanism"] != mech and i["side"] == side
                                    and abs(i["at_c"] - at[r["world"]]) <= TOL_C),
        }
    return out


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    a = ap.parse_args()
    runs = Path(a.runs)
    rows = load(runs)
    v = verdicts(rows)
    rep = reports(rows)
    lines = ["shape stage scoring (ccpp-shape-elicitation.md §4-§6)", "",
             "model | cap loc/shape (n_loc) | evap loc/shape (n_loc) | ice L2 loc/shape (n_loc) | inferred loc/shape"]
    for t in TAGS:
        c, e, i, f = v["P1"]["by_model"][t], v["P3"]["by_model"][t], v["P2"]["by_model"][t], v["P4"]["by_model"][t]
        lines.append(f"{t} | {fmt(c['loc_rate'])}/{fmt(c['shape_rate'])} ({c['n_loc']}) | "
                     f"{fmt(e['loc_rate'])}/{fmt(e['shape_rate'])} ({e['n_loc']}) | "
                     f"{fmt(i['loc_rate'])}/{fmt(i['shape_rate'])} ({i['n_loc']}) | "
                     f"{fmt(f['loc_rate'])}/{fmt(f['shape_rate'])}")
    lines += ["", "responses among location-correct items:"]
    for g, key in (("cap", "P1"), ("evap", "P3"), ("ice L2", "P2")):
        lines.append(f"  {g}: " + json.dumps({t: v[key]['by_model'][t]['responses'] for t in TAGS}))
    lines += ["", f"P5 pooled shape_rate: cap {fmt(v['P5']['cap'])}, inferred {fmt(v['P5']['inferred'])}",
              "", "reports:", json.dumps(rep, indent=1, default=str), "",
              "=== verdicts ===", json.dumps({k: v[k]["pass"] for k in ("P1", "P2", "P3", "P4", "P5")})]
    dest = runs / "shape_score"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "score.txt").write_text("\n".join(lines) + "\n")
    json.dump({"verdicts": v, "reports": rep}, open(dest / "score.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
