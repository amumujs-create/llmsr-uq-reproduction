"""Shape stage 2 scoring: Q1-Q5 (docs/research/ccpp-shape-elicitation-2.md §4-§6, fixed before data).

Usage: python score_shape2.py [--runs runs]
Reads runs/shape2/<tag>.jsonl. Writes runs/shape2_score/score.{txt,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_score import MECHANISMS2, TOL_C, parse_shape  # noqa: E402
from score_shape import null_location  # noqa: E402

TAGS = ("gpt-oss-120b", "qwen3-30b-a3b-instruct", "gpt-oss-20b", "gemma3-27b", "llama33-70b")
MIN_ITEMS = 10
# §4: group -> (world, mechanism, side, truth location, correct response, levels used for the verdict)
GROUPS = {
    "chill": ("chill", "inlet_chilling", "above", 20.0, "near_zero", ("L1", "L2")),
    "gen_hot": ("gen_hot", "power_cap", "above", 28.0, "same_or_stronger", ("L2",)),
    "evap": ("evap", "evap_cooling", "above", 24.0, "weaker", ("L1", "L2")),
    "cap": ("cap_mild", "power_cap", "below", 6.0, "near_zero", ("L1", "L2")),
}
NEW = ("chill", "gen_hot")


def load(runs):
    rows = []
    for tag in TAGS:
        for line in open(runs / "shape2" / f"{tag}.jsonl"):
            r = json.loads(line)
            text = r["response"] if r["error"] is None else ""
            p = parse_shape(text, MECHANISMS2)
            rows.append({"tag": tag, "world": r["world"], "level": r["level"], "ok": p["ok"],
                         "error": r["error"] is not None, "null_loc": (not p["ok"]) and null_location(text),
                         "items": p["items"] if p["ok"] else []})
    return rows


def loc_item(items, group, any_mechanism=False):
    """The location-correct item for this group in one sample, or None (§4)."""
    _, mech, side, at, *_ = GROUPS[group]
    for it in items:
        if (any_mechanism or it["mechanism"] == mech) and it["side"] == side and abs(it["at_c"] - at) <= TOL_C:
            return it
    return None


def rates(rows, group, levels=None, any_mechanism=False):
    world, *_, correct, default_levels = GROUPS[group]
    levels = default_levels if levels is None else levels
    sel = [r for r in rows if r["world"] == world and r["level"] in levels]
    hits = [h for h in (loc_item(r["items"], group, any_mechanism) for r in sel) if h is not None]
    good = sum(h["response_beyond"] == correct for h in hits)
    return {"n": len(sel), "n_loc": len(hits), "loc_rate": len(hits) / len(sel) if sel else None,
            "n_shape": good, "shape_rate": good / len(hits) if hits else None,
            "responses": dict(Counter(str(h["response_beyond"]) for h in hits))}


def _below_half(by_model, need):
    judged = {t: v["shape_rate"] for t, v in by_model.items() if v["n_loc"] >= MIN_ITEMS}
    k = min(need, len(judged))
    return judged, ((sum(s < 0.5 for s in judged.values()) >= k) if judged else None)


def verdicts(rows):
    by = {t: [r for r in rows if r["tag"] == t] for t in TAGS}
    res = {}
    for q, g in (("Q1", "chill"), ("Q2", "gen_hot")):
        bm = {t: rates(by[t], g) for t in TAGS}
        judged, ok = _below_half(bm, 3)
        res[q] = {"by_model": bm, "judged": judged, "excluded": [t for t in TAGS if t not in judged], "pass": ok}
    # Q3: among location-correct items in the new worlds (chill L1+L2, gen_hot L2), "weaker" is the strict mode
    q3 = {}
    for t in TAGS:
        c = Counter()
        for g in NEW:
            c.update(rates(by[t], g)["responses"])
        n = sum(c.values())
        top = c.most_common(2)
        q3[t] = {"n": n, "responses": dict(c),
                 "weaker_mode": (n >= MIN_ITEMS and top[0][0] == "weaker"
                                 and (len(top) == 1 or top[0][1] > top[1][1])) if n >= MIN_ITEMS else None}
    judged = {t: v["weaker_mode"] for t, v in q3.items() if v["weaker_mode"] is not None}
    res["Q3"] = {"by_model": q3, "pass": (sum(judged.values()) >= min(3, len(judged))) if judged else None}
    # Q4: control (evap) replicates stage 1 under the new question
    ev = {t: rates(by[t], "evap") for t in TAGS}
    judged = {t: v["shape_rate"] for t, v in ev.items() if v["n_loc"] >= MIN_ITEMS}
    res["Q4"] = {"by_model": ev, "judged": judged, "excluded": [t for t in TAGS if t not in judged],
                 "pass": (sum(s >= 0.6 for s in judged.values()) >= min(3, len(judged))) if judged else None}
    # Q5: pooled over models, evap control minus new worlds
    ev_all = rates(rows, "evap")
    new = [rates(rows, g) for g in NEW]
    n_loc, n_shape = sum(r["n_loc"] for r in new), sum(r["n_shape"] for r in new)
    s_new = n_shape / n_loc if n_loc else None
    res["Q5"] = {"evap": ev_all["shape_rate"], "new": s_new, "new_n_loc": n_loc,
                 "pass": (ev_all["shape_rate"] - s_new >= 0.3) if None not in (ev_all["shape_rate"], s_new) else None}
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
            "base_false_alarm_rate": (sum(bool(r["items"]) for r in base) / len(base)) if base else None,
            "base_responses": dict(Counter(str(i["response_beyond"]) for r in base for i in r["items"])),
            "by_level": {g: {lv: rates(rs, g, (lv,)) for lv in ("L1", "L2")} for g in GROUPS},
            "any_mechanism": {g: rates(rs, g, any_mechanism=True) for g in GROUPS},
            "responses_any_location": {g: dict(Counter(str(i["response_beyond"]) for r in rs
                                                       if r["world"] == GROUPS[g][0] for i in r["items"]
                                                       if i["side"] == GROUPS[g][2]))
                                       for g in GROUPS},
            "mechanisms_at_location": {g: dict(Counter(i["mechanism"] for r in rs if r["world"] == GROUPS[g][0]
                                                       for i in r["items"] if i["side"] == GROUPS[g][2]
                                                       and abs(i["at_c"] - GROUPS[g][3]) <= TOL_C))
                                       for g in GROUPS},
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
    lines = ["shape stage 2 scoring (ccpp-shape-elicitation-2.md §4-§6)", "",
             "model | chill loc/shape (n_loc) | gen_hot L2 loc/shape (n_loc) | evap loc/shape (n_loc) | new-world responses"]
    for t in TAGS:
        c, g, e = v["Q1"]["by_model"][t], v["Q2"]["by_model"][t], v["Q4"]["by_model"][t]
        lines.append(f"{t} | {fmt(c['loc_rate'])}/{fmt(c['shape_rate'])} ({c['n_loc']}) | "
                     f"{fmt(g['loc_rate'])}/{fmt(g['shape_rate'])} ({g['n_loc']}) | "
                     f"{fmt(e['loc_rate'])}/{fmt(e['shape_rate'])} ({e['n_loc']}) | "
                     f"{json.dumps(v['Q3']['by_model'][t]['responses'])}")
    lines += ["", f"Q5 pooled shape_rate: evap {fmt(v['Q5']['evap'])}, new worlds {fmt(v['Q5']['new'])}",
              "", "reports:", json.dumps(rep, indent=1, default=str), "",
              "=== verdicts ===", json.dumps({k: v[k]["pass"] for k in ("Q1", "Q2", "Q3", "Q4", "Q5")})]
    dest = runs / "shape2_score"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "score.txt").write_text("\n".join(lines) + "\n")
    json.dump({"verdicts": v, "reports": rep}, open(dest / "score.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
