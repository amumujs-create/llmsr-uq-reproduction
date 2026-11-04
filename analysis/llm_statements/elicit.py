"""Prompting-strategy experiment (docs/research/llm-statement-collection.md §12).

S1: same question, step-by-step reasoning before the answer.
S2: mechanism decomposition; the boundary is computed from the model's answers, not asked directly.
Same descriptions (L1, L2) as the main collection; no extra facts.

Usage: python elicit.py --base-url http://localhost:11434/v1 --model mistral:7b --tag t --out runs/strategy/t.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect import call_openai  # noqa: E402
from prompts import QUESTION, SYSTEM, WORLDS, describe  # noqa: E402

RANGE = (2.0, 37.0)

S1_QUESTION = QUESTION.replace(
    "Reply with JSON only, in this form:\n{",
    "First reason step by step about each mechanism in the plant that could change how output responds to "
    "ambient temperature, then answer. Reply with JSON only, in this form:\n{\"reasoning\": \"<your step-by-step "
    "reasoning>\", ")

S2_QUESTION = (
    "Answer each item separately for this plant over the 2 to 37 degC ambient range. Give your best numerical "
    "estimates even if you are unsure. Reply with JSON only, in this form:\n"
    "{\"gt_generator_limit\": {\"reached_in_range\": true or false, \"gt_mw_at_15c\": <MW>, \"limit_mw\": <MW>, "
    "\"gt_mw_per_degc\": <change of gas-turbine output per degC of ambient temperature, negative if output falls "
    "as temperature rises>},\n"
    " \"inlet_anti_icing_heating\": {\"active\": true or false, \"below_c\": <degC below which it operates>},\n"
    " \"inlet_evaporative_cooling\": {\"active\": true or false, \"above_c\": <degC above which it operates>},\n"
    " \"exhaust_or_steam_side_limit\": {\"active\": true or false, \"above_c\": <degC above which it holds output>}}"
)


def messages(world: str, level: str, strategy: str) -> list[dict]:
    q = S1_QUESTION if strategy == "S1" else S2_QUESTION
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{describe(world, level)}\n\n{q}"}]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def assemble_s2(text: str) -> dict:
    """Turn S2 mechanism answers into a statement [(side, at)], keeping the innermost boundary per side."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"ok": False, "stmt": [], "error": "no_json", "parts": {}}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"ok": False, "stmt": [], "error": f"bad_format: {e}", "parts": {}}
    cand, parts = [], {}
    g = obj.get("gt_generator_limit") or {}
    if g.get("reached_in_range") is True:
        p15, lim, sens = _num(g.get("gt_mw_at_15c")), _num(g.get("limit_mw")), _num(g.get("gt_mw_per_degc"))
        if None not in (p15, lim, sens) and sens != 0:
            at = 15.0 - (lim - p15) / abs(sens)
            parts["gt_limit"] = at
            if RANGE[0] < at < RANGE[1]:
                cand.append(("below", at))
    for key, side, field in (("inlet_anti_icing_heating", "below", "below_c"),
                             ("inlet_evaporative_cooling", "above", "above_c"),
                             ("exhaust_or_steam_side_limit", "above", "above_c")):
        d = obj.get(key) or {}
        at = _num(d.get(field))
        if d.get("active") is True and at is not None:
            parts[key] = at
            if RANGE[0] < at < RANGE[1]:
                cand.append((side, at))
    stmt = []
    for side in ("below", "above"):
        vals = [a for s, a in cand if s == side]
        if vals:
            stmt.append((side, max(vals) if side == "below" else min(vals)))
    return {"ok": True, "stmt": stmt, "error": None, "parts": parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    extra = json.loads(a.extra_body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    tasks = []
    for strategy in ("S1", "S2"):
        for level in ("L1", "L2"):
            for world in WORLDS:
                for k in range(a.samples + 1):
                    cid = f"{a.tag}|{strategy}|{world}|{level}|{k}"
                    if cid not in done:
                        tasks.append((cid, strategy, world, level, k, 0.0 if k == a.samples else 0.7))

    def one(t):
        cid, strategy, world, level, k, temp = t
        msgs = messages(world, level, strategy)
        t0 = time.time()
        try:
            text, reasoning = call_openai(a.base_url, a.model, msgs, temp, k, a.max_tokens, extra)
            err = None
        except Exception as e:
            text, reasoning, err = None, None, f"{type(e).__name__}: {e}"
        return {"id": cid, "tag": a.tag, "model": a.model, "strategy": strategy, "world": world, "level": level,
                "data": "D0", "setting": None, "sample": k, "temperature": temp, "messages": msgs,
                "response": text, "reasoning": reasoning, "error": err, "seconds": round(time.time() - t0, 3)}

    with out.open("a") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        for i, rec in enumerate(ex.map(one, tasks), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
