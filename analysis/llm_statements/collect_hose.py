"""Tool-hose stage (docs/research/robot-twin-hose-llm-preregistration.md §5).

6 prompts (L0, L1 world-independent; L2 x 4 braid-angle worlds) x (20 at temperature 0.7 + 1 at 0) = 126 calls.

Usage: python collect_hose.py --base-url http://localhost:8000/v1 --model <served> --tag <tag> --out runs/hose/<tag>.jsonl
       python collect_hose.py --backend mock --tag mock --out /tmp/hose_mock.jsonl     # dry run, no network
       python collect_hose.py --check runs/hose/<tag>.jsonl [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect import call_openai  # noqa: E402
from prompts_hose import WORLDS, jobs, messages  # noqa: E402
from score_hose import parse  # noqa: E402

SAMPLES, TEMP = 20, 0.7
N_CALLS = len(jobs()) * (SAMPLES + 1)


def tasks(tag: str, samples: int = SAMPLES) -> list[tuple]:
    out = []
    for world, level in jobs():
        for k in range(samples + 1):
            out.append((f"{tag}|hose|{world}|{level}|{k}", world, level, k, 0.0 if k == samples else TEMP))
    return out


def call_mock(world: str, level: str, seed: int) -> str:
    """Pipeline check only: 'shorter' by default, follows the braid rule at L2."""
    rng = np.random.default_rng(seed)
    pct = -2.0 + rng.normal(0, 0.5)
    if level == "L2":
        pct = 100 * WORLDS[world][1] + rng.normal(0, 0.5)
    d = "shorter" if pct < -0.5 else "longer" if pct > 0.5 else "unchanged"
    r = {"shorter": "narrower", "longer": "wider", "unchanged": "unchanged"}[d]
    return json.dumps({"length_change_pct": round(float(pct), 2), "direction": d, "j6_range": r,
                       "confidence": 0.5, "rationale": "mock"})


def check(path: str, smoke: bool) -> int:
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"]]
    ok = [r for r in recs if not r["error"]]
    bad = [r for r in ok if not parse(r["response"])["ok"]]
    print(f"hose: {len(recs)} calls, {len(err)} errors, unparsed {len(bad)}/{len(ok)}")
    if smoke and (err or not ok or len(bad) == len(ok)):
        r = (err + bad)[0]
        print("error:", r["error"])
        print("response head:", (r["response"] or "")[:400])
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None, help="print error / parse counts for a collected file and exit")
    ap.add_argument("--smoke", action="store_true", help="with --check: exit 1 when nothing usable came back")
    ap.add_argument("--backend", choices=("openai", "mock"), default="openai")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=SAMPLES, help="temperature-0.7 samples per prompt (0 = smoke)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.check, a.smoke))
    if not (a.tag and a.out):
        ap.error("--tag and --out are required")
    extra = json.loads(a.extra_body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [t for t in tasks(a.tag, a.samples) if t[0] not in done]

    def one(t):
        cid, world, level, k, temp = t
        msgs = messages("shrink4" if world == "any" else world, level)   # L0/L1 text does not depend on the world
        t0 = time.time()
        try:
            if a.backend == "mock":
                text, reasoning = call_mock(world, level, k), None
            else:
                text, reasoning = call_openai(a.base_url, a.model, msgs, temp, k, a.max_tokens, extra)
            err = None
        except Exception as e:  # recorded, not retried (same policy as collect.py)
            text, reasoning, err = None, None, f"{type(e).__name__}: {e}"
        return {"id": cid, "tag": a.tag, "model": a.model, "stage": "hose", "world": world, "level": level,
                "data": "D0", "sample": k, "temperature": temp, "messages": msgs, "response": text,
                "reasoning": reasoning, "error": err, "seconds": round(time.time() - t0, 3)}

    with out.open("a") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
