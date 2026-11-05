"""Parse LLM regime statements and score them against twin truth (docs/research/llm-statement-collection.md §5, §7)."""
from __future__ import annotations

import json
import re
from collections import Counter

TRUTH = {
    "base": [],
    "cap_mild": [("below", 6.0)],
    "cap_strong": [("below", 9.0)],
    "evap": [("above", 24.0)],
    "evap+cap_mild": [("below", 6.0), ("above", 24.0)],
}
TOL_C = 3.0


def parse(text: str) -> dict:
    """Return {"ok": bool, "boundaries": [(side, at_c, confidence)], "duplicates": int, "error": str|None}."""
    # Reasoning models may put a <think>...</think> block before the answer; parse the final JSON only.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"ok": False, "boundaries": [], "duplicates": 0, "error": "no_json"}
    try:
        obj = json.loads(m.group(0))
        raw = obj["boundaries"]
        if not isinstance(raw, list):
            raise ValueError("boundaries not a list")
        out = []
        for b in raw:
            side = str(b["side"]).strip().lower()
            if side not in ("below", "above"):
                raise ValueError(f"bad side {side}")
            conf = b.get("confidence")
            out.append((side, float(b["at_c"]), None if conf is None else float(conf)))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return {"ok": False, "boundaries": [], "duplicates": 0, "error": f"bad_format: {e}"}
    # Keep the innermost boundary per side (closest to the middle of the range).
    dup = 0
    keep = {}
    for side, at, conf in out:
        if side in keep:
            dup += 1
            inner = at > keep[side][1] if side == "below" else at < keep[side][1]
            if not inner:
                continue
        keep[side] = (side, at, conf)
    return {"ok": True, "boundaries": list(keep.values()), "duplicates": dup, "error": None}


MECHANISMS = ("power_cap", "anti_icing", "evap_cooling", "other")
TRENDS = ("rises", "falls", "about the same")


def _opt_float(x):
    if x is None or (isinstance(x, str) and x.strip().lower() in ("", "null", "none")):
        return None
    return float(x)


def parse_mech(text: str) -> dict:
    """Parse the mechanism-tagged answer (prompts.QUESTION_MECH, anti-icing §6).

    Returns {"ok", "items": [{"mechanism", "side", "at_c", "rh_above", "output_trend", "confidence"}],
    "duplicates", "error"}. Unknown mechanisms become "other" and unknown trends None (kept, not errors).
    Two cold-side causes may coexist, so the innermost boundary is kept per (side, mechanism), not per side.
    Answers without the new fields parse as mechanism "other" with no humidity condition.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"ok": False, "items": [], "duplicates": 0, "error": "no_json"}
    try:
        raw = json.loads(m.group(0))["boundaries"]
        if not isinstance(raw, list):
            raise ValueError("boundaries not a list")
        items = []
        for b in raw:
            side = str(b["side"]).strip().lower()
            if side not in ("below", "above"):
                raise ValueError(f"bad side {side}")
            mech = str(b.get("mechanism") or "other").strip().lower().replace("-", "_").replace(" ", "_")
            trend = b.get("output_trend")
            trend = str(trend).strip().lower() if trend is not None else None
            items.append({"mechanism": mech if mech in MECHANISMS else "other", "side": side,
                          "at_c": float(b["at_c"]), "rh_above": _opt_float(b.get("rh_above")),
                          "output_trend": trend if trend in TRENDS else None,
                          "confidence": _opt_float(b.get("confidence"))})
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return {"ok": False, "items": [], "duplicates": 0, "error": f"bad_format: {e}"}
    dup, keep = 0, {}
    for it in items:
        k = (it["side"], it["mechanism"])
        if k in keep:
            dup += 1
            inner = it["at_c"] > keep[k]["at_c"] if it["side"] == "below" else it["at_c"] < keep[k]["at_c"]
            if not inner:
                continue
        keep[k] = it
    return {"ok": True, "items": list(keep.values()), "duplicates": dup, "error": None}


def to_statement(boundaries) -> list[tuple[str, float]]:
    """Convert parsed boundaries to the (side, at) list used by regime_rules / risk_limits."""
    return [(s, a) for s, a, _ in boundaries]


def score_side(truth, stmt, side) -> str:
    t = [a for s, a in truth if s == side]
    p = [a for s, a in stmt if s == side]
    if not t and not p:
        return "correct"
    if t and not p:
        return "miss"
    if p and not t:
        return "false_alarm"
    return "correct" if abs(p[0] - t[0]) <= TOL_C else "location"


def score(world: str, stmt) -> dict:
    truth = TRUTH[world]
    cold, hot = score_side(truth, stmt, "below"), score_side(truth, stmt, "above")
    return {"cold": cold, "hot": hot, "all_correct": cold == "correct" and hot == "correct"}


def majority(statements: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Sides named by more than half of the samples, located at the median."""
    n = len(statements)
    out = []
    for side in ("below", "above"):
        vals = sorted(a for st in statements for s, a in st if s == side)
        if len(vals) * 2 > n:
            mid = len(vals) // 2
            out.append((side, vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])))
    return out


def agreement(stmt, reference) -> bool:
    """Same sides as the reference, each within TOL_C."""
    if Counter(s for s, _ in stmt) != Counter(s for s, _ in reference):
        return False
    ref = dict(reference)
    return all(abs(a - ref[s]) <= TOL_C for s, a in stmt)
