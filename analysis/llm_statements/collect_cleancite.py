"""EXPLORATION stage (docs/research/ccpp-twin-explore-cite.md): the clean-world questions with one change each.

Every claimed boundary (S0C) or device (S2C) must carry "evidence": one sentence of the plant description copied word
for word, or null. Descriptions are the clean-world ones unchanged, plus the device-free base plant at L2 (the plant
with the most false alarms so far). Development models only (models.tsv). Not a registered verdict.
A second, simpler variant (S2N) has no evidence field: each device item may be answered "not stated" instead of
true or false. It separates the effect of quoting from the effect of being allowed to say the description is silent.
7 worlds x (21 S0C + 11 S2C + 11 S2N) = 301 calls per model.

quote_ok() is the mechanical check fixed before any answer is collected: the evidence must be a word-for-word part of
the description and at least MIN_WORDS words long. It does not judge whether the sentence is relevant.

Usage: python collect_cleancite.py --base-url http://localhost:8000/v1 --model <served> --tag <tag> --out runs/cleancite/<tag>.jsonl
       python collect_cleancite.py --check runs/cleancite/<tag>.jsonl [--smoke]
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
from assemble import _json  # noqa: E402
from collect import call_openai  # noqa: E402
from collect_cleanworld import CLEAN_WORLDS, describe_clean  # noqa: E402
from elicit import S2_QUESTION  # noqa: E402
from parse_score import parse  # noqa: E402
from prompts import QUESTION, SYSTEM, describe  # noqa: E402

BASE_WORLD = "base"
WORLDS = (BASE_WORLD,) + tuple(CLEAN_WORLDS)
LEVEL = "L2"
SAMPLES = {"S0C": 20, "S2C": 10, "S2N": 10}        # plus one temperature-0 call each
TEMP = 0.7
N_CALLS = len(WORLDS) * sum(n + 1 for n in SAMPLES.values())
MIN_WORDS = 4
S2_ITEMS = ("gt_generator_limit", "inlet_anti_icing_heating", "inlet_evaporative_cooling",
            "exhaust_or_steam_side_limit")
S2_FLAG = {"gt_generator_limit": "reached_in_range", "inlet_anti_icing_heating": "active",
           "inlet_evaporative_cooling": "active", "exhaust_or_steam_side_limit": "active"}

EVIDENCE_FIELD = ', "evidence": "<sentence>" or null'
EVIDENCE_NOTE = ('"evidence" is the one sentence of the plant description above that this item rests on, copied word '
                 "for word, or null if no sentence of the description says so. ")


def _insert(text: str, anchor: str, addition: str, before: bool = False) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"expected exactly one {anchor!r}")
    return text.replace(anchor, addition + anchor if before else anchor + addition)


def _s0c() -> str:
    q = _insert(QUESTION, '"confidence": <0 to 1>', EVIDENCE_FIELD)
    return _insert(q, "Use an empty list if you expect no such temperature.", EVIDENCE_NOTE, before=True)


def _s2c() -> str:
    q = S2_QUESTION
    for tail in ("negative if output falls as temperature rises>", "<degC below which it operates>",
                 "<degC above which it operates>", "<degC above which it holds output>"):
        q = _insert(q, tail, EVIDENCE_FIELD)
    return _insert(q, "Reply with JSON only, in this form:\n", EVIDENCE_NOTE, before=True)


NOT_STATED = 'true, false or "not stated"'
NOT_STATED_NOTE = 'Answer "not stated" for an item the plant description above says nothing about. '


def _s2n() -> str:
    if S2_QUESTION.count("true or false") != len(S2_ITEMS):
        raise ValueError("expected one 'true or false' per S2 item")
    q = S2_QUESTION.replace("true or false", NOT_STATED)
    return _insert(q, "Reply with JSON only, in this form:\n", NOT_STATED_NOTE, before=True)


S0C_QUESTION = _s0c()
S2C_QUESTION = _s2c()
S2N_QUESTION = _s2n()
QUESTIONS = {"S0C": S0C_QUESTION, "S2C": S2C_QUESTION, "S2N": S2N_QUESTION}


def describe_cite(world: str) -> str:
    return describe(BASE_WORLD, LEVEL) if world == BASE_WORLD else describe_clean(world)


def messages(world: str, kind: str) -> list[dict]:
    q = QUESTIONS[kind]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{describe_cite(world)}\n\n{q}"}]


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    return text.strip(" \"'`.“”‘’")


def quote_ok(evidence, description: str) -> bool:
    """True when evidence is a word-for-word part of the description, at least MIN_WORDS words long."""
    if not isinstance(evidence, str):
        return False
    e = _norm(evidence)
    return len(e.split()) >= MIN_WORDS and e in _norm(description)


def s0c_evidence(text: str):
    """[(side, at_c, evidence)] for one S0C answer, every boundary kept; None when the answer does not parse."""
    if not parse(text)["ok"]:
        return None
    obj = _json(text)
    return [(str(b["side"]).strip().lower(), float(b["at_c"]), b.get("evidence")) for b in obj["boundaries"]]


def s2c_evidence(obj: dict) -> dict:
    """{item: (claimed, evidence)} for one S2 / S2C / S2N answer; only a JSON true counts as claimed."""
    out = {}
    for item in S2_ITEMS:
        d = obj.get(item) or {}
        out[item] = (d.get(S2_FLAG[item]) is True, d.get("evidence"))
    return out


def tasks(tag: str, samples: dict | None = None) -> list[tuple]:
    samples = SAMPLES if samples is None else samples
    out = []
    for kind in SAMPLES:
        for world in WORLDS:
            n = samples[kind]
            for k in range(n + 1):
                out.append((f"{tag}|{kind}|{world}|{LEVEL}|{k}", kind, world, k, 0.0 if k == n else TEMP))
    return out


def check(path: str, smoke: bool) -> int:
    """Counts only (errors, unparsed answers, answers with an evidence field); prints no answer content."""
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"]]
    s0 = [r for r in recs if r["strategy"] == "S0C" and not r["error"]]
    s2 = [r for r in recs if r["strategy"] == "S2C" and not r["error"]]
    sn = [r for r in recs if r["strategy"] == "S2N" and not r["error"]]
    sn_bad = [r for r in sn if _json(r["response"]) is None]
    s0_bad = [r for r in s0 if not parse(r["response"])["ok"]]
    s2_bad = [r for r in s2 if _json(r["response"]) is None]
    s0_ev = sum('"evidence"' in (r["response"] or "") for r in s0)
    s2_ev = sum('"evidence"' in (r["response"] or "") for r in s2)
    print(f"cleancite: {len(recs)} calls, {len(err)} errors, S0C unparsed {len(s0_bad)}/{len(s0)}, "
          f"S2C unparsed {len(s2_bad)}/{len(s2)}, evidence field present S0C {s0_ev}/{len(s0)} S2C {s2_ev}/{len(s2)}, "
          f"S2N unparsed {len(sn_bad)}/{len(sn)}")
    if smoke and (err or not s0 or not s2 or not sn or len(s0_bad) == len(s0) or len(s2_bad) == len(s2)
                  or len(sn_bad) == len(sn)):
        bad = err + s0_bad + s2_bad + sn_bad
        if bad:
            print("error:", bad[0]["error"])
            print("response head:", (bad[0]["response"] or "")[:400])
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None, help="print error / parse counts for a collected file and exit")
    ap.add_argument("--smoke", action="store_true", help="with --check: exit 1 when nothing usable came back")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=None, help="override both sample counts (0 = one call each)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.check, a.smoke))
    if not (a.model and a.tag and a.out):
        ap.error("--model, --tag and --out are required")
    extra = json.loads(a.extra_body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    samples = None if a.samples is None else {k: a.samples for k in SAMPLES}
    todo = [t for t in tasks(a.tag, samples) if t[0] not in done]

    def one(t):
        cid, kind, world, k, temp = t
        msgs = messages(world, kind)
        t0 = time.time()
        try:
            text, reasoning = call_openai(a.base_url, a.model, msgs, temp, k, a.max_tokens, extra)
            err = None
        except Exception as e:  # recorded, not retried (same policy as collect.py)
            text, reasoning, err = None, None, f"{type(e).__name__}: {e}"
        return {"id": cid, "tag": a.tag, "model": a.model, "stage": "cleancite", "strategy": kind, "world": world,
                "level": LEVEL, "data": "D0", "setting": None, "sample": k, "temperature": temp, "messages": msgs,
                "response": text, "reasoning": reasoning, "error": err, "seconds": round(time.time() - t0, 3)}

    with out.open("a") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
