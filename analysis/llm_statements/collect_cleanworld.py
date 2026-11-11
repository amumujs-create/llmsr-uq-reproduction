"""Clean-world stage (docs/research/ccpp-twin-clean-collection.md): the main question (S0) and the device question
(S2) for six plants whose stated limit or set temperature differs from the ones already asked.

Only the L2 description is new (it carries the number). The L1 descriptions of these plants are word for word the
ones already collected (cap, evap, chill), so L1 is not asked again. Question wording is QUESTION / S2_QUESTION
unchanged. Each L2 sentence is an existing sentence with one number replaced.
6 worlds x (21 S0 + 11 S2) = 192 calls per model.

Usage: python collect_cleanworld.py --base-url http://localhost:8000/v1 --model <served> --tag <tag> --out runs/cleanworld/<tag>.jsonl
       python collect_cleanworld.py --check runs/cleanworld/<tag>.jsonl [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble import _json  # noqa: E402
from collect import call_openai  # noqa: E402
from elicit import S2_QUESTION  # noqa: E402
from parse_score import parse  # noqa: E402
from prompts import L0, L1, L1_NEW, L2_CAP, L2_EVAP, L2_GT, L2_NEW, QUESTION, SYSTEM  # noqa: E402


def _swap(sentence: str, old: str, new: str) -> str:
    if sentence.count(old) != 1:
        raise ValueError(f"expected exactly one {old!r} in {sentence!r}")
    return sentence.replace(old, new)


# world -> (L1 sentence, L2 sentence). The number is the only change from an existing sentence.
CLEAN_WORLDS = {
    "cap353": (L1["cap"], _swap(L2_CAP["cap_strong"], "357", "353")),
    "cap350": (L1["cap"], _swap(L2_CAP["cap_strong"], "357", "350")),
    "cap346": (L1["cap"], _swap(L2_CAP["cap_strong"], "357", "346")),
    "evap22": (L1["evap"], _swap(L2_EVAP, "24", "22")),
    "evap26": (L1["evap"], _swap(L2_EVAP, "24", "26")),
    "chill24": (L1_NEW["chill"], _swap(L2_NEW["chill"], "20", "24")),
}
LEVEL = "L2"
SAMPLES = {"S0": 20, "S2": 10}          # plus one temperature-0 call each
TEMP = 0.7
N_CALLS = len(CLEAN_WORLDS) * sum(n + 1 for n in SAMPLES.values())


def describe_clean(world: str) -> str:
    l1, l2 = CLEAN_WORLDS[world]
    return " ".join([L0, l1, L2_GT, l2])


def messages(world: str, kind: str) -> list[dict]:
    q = QUESTION if kind == "S0" else S2_QUESTION
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{describe_clean(world)}\n\n{q}"}]


def tasks(tag: str, samples: dict | None = None) -> list[tuple]:
    samples = SAMPLES if samples is None else samples
    out = []
    for kind in ("S0", "S2"):
        for world in CLEAN_WORLDS:
            n = samples[kind]
            for k in range(n + 1):
                out.append((f"{tag}|{kind}|{world}|{LEVEL}|{k}", kind, world, k, 0.0 if k == n else TEMP))
    return out


def check(path: str, smoke: bool) -> int:
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"]]
    s0 = [r for r in recs if r["strategy"] == "S0" and not r["error"]]
    s2 = [r for r in recs if r["strategy"] == "S2" and not r["error"]]
    s0_bad = [r for r in s0 if not parse(r["response"])["ok"]]
    s2_bad = [r for r in s2 if _json(r["response"]) is None]
    print(f"cleanworld: {len(recs)} calls, {len(err)} errors, S0 unparsed {len(s0_bad)}/{len(s0)}, "
          f"S2 unparsed {len(s2_bad)}/{len(s2)}")
    if smoke and (err or not s0 or not s2 or len(s0_bad) == len(s0) or len(s2_bad) == len(s2)):
        bad = err + s0_bad + s2_bad
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
        return {"id": cid, "tag": a.tag, "model": a.model, "stage": "cleanworld", "strategy": kind, "world": world,
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
