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
    m = re.search(r"\{.*\}", text or "", re.S)
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
