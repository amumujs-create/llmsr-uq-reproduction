"""EXPLORATION stage (docs/research/ccpp-twin-indirect-spec-design.md): indirect specifications.

Nine plants differ only in the generator rating (MVA) and power factor. The limit of the gas turbine generator is
rating x power factor; whether and where the cold-side kink exists follows from that number in the twin. Three
information tiers:
  T0  the plant sentence only (one description, asked once);
  T2  indirect: rating and power factor, no sentence about behaviour and no limit in MW;
  T3  direct: the behaviour sentence and the limit in MW (the wording already used in the clean-world stage).
Questions are the existing ones, unchanged: QUESTION (S0) and collect_cleancite.S2N_QUESTION (S2N).
Development models only (models.tsv). Not a registered verdict.
(1 + 9 + 9) descriptions x (21 S0 + 11 S2N) = 608 calls per model.

Usage: python collect_spec.py --base-url http://localhost:8000/v1 --model <served> --tag <tag> --out runs/spec/<tag>.jsonl
       python collect_spec.py --check runs/spec/<tag>.jsonl [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble import _json  # noqa: E402
from collect import call_openai  # noqa: E402
from collect_cleancite import S2N_QUESTION  # noqa: E402
from collect_cleanworld import _swap  # noqa: E402
from parse_score import parse  # noqa: E402
from prompts import L0, L1, L2_CAP, L2_GT, QUESTION, SYSTEM  # noqa: E402

# plant -> (rating in MVA, power factor). Fixed in the design (§3) before any answer is collected.
PLANTS = {
    "g415": (415, 0.85), "g407": (407, 0.85), "g392p": (392, 0.90), "g400p": (400, 0.90), "g400": (400, 0.85),
    "g425": (425, 0.85), "n450": (450, 0.85), "n425p": (425, 0.90), "n470": (470, 0.85),
}
KINK = ("g415", "g407", "g392p", "g400p", "g400", "g425")       # limit is reached inside 2 to 37 degC
NO_KINK = ("n450", "n425p", "n470")
T0_WORLD = "none"
TIERS = ("T0", "T2", "T3")
SAMPLES = {"S0": 20, "S2N": 10}        # plus one temperature-0 call each
TEMP = 0.7
QUESTIONS = {"S0": QUESTION, "S2N": S2N_QUESTION}
N_CALLS = (1 + 2 * len(PLANTS)) * sum(n + 1 for n in SAMPLES.values())

NO_AC = "The plant has no inlet air conditioning."
RATING = "The gas turbine generator is rated {mva} MVA at a power factor of {pf:.2f}."
if not (L1["cap"].startswith(NO_AC) and L1["base"].startswith(NO_AC[:-1])):
    raise ValueError("NO_AC must be the opening words of the existing L1 sentences")


def limit_mw(plant: str) -> float:
    mva, pf = PLANTS[plant]
    return mva * pf


def stated_mw(plant: str) -> int:
    """The limit as written at T3: rounded half up to a whole MW."""
    return int(math.floor(limit_mw(plant) + 0.5))


def worlds(tier: str) -> tuple:
    return (T0_WORLD,) if tier == "T0" else tuple(PLANTS)


def describe_spec(world: str, tier: str) -> str:
    if tier == "T0":
        if world != T0_WORLD:
            raise ValueError("T0 has one description")
        return L0
    mva, pf = PLANTS[world]
    if tier == "T2":
        return " ".join([L0, NO_AC, L2_GT, RATING.format(mva=mva, pf=pf)])
    if tier == "T3":
        l1 = L1["cap"] if world in KINK else L1["base"]
        return " ".join([L0, l1, L2_GT, _swap(L2_CAP["cap_strong"], "357", str(stated_mw(world)))])
    raise ValueError(f"unknown tier {tier!r}")


def messages(world: str, tier: str, kind: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{describe_spec(world, tier)}\n\n{QUESTIONS[kind]}"}]


def tasks(tag: str, samples: dict | None = None) -> list[tuple]:
    samples = SAMPLES if samples is None else samples
    out = []
    for tier in TIERS:
        for kind in SAMPLES:
            for world in worlds(tier):
                n = samples[kind]
                for k in range(n + 1):
                    out.append((f"{tag}|{kind}|{world}|{tier}|{k}", kind, world, tier, k, 0.0 if k == n else TEMP))
    return out


def check(path: str, smoke: bool) -> int:
    """Counts only (errors, unparsed answers); prints no answer content."""
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"]]
    s0 = [r for r in recs if r["strategy"] == "S0" and not r["error"]]
    sn = [r for r in recs if r["strategy"] == "S2N" and not r["error"]]
    s0_bad = [r for r in s0 if not parse(r["response"])["ok"]]
    sn_bad = [r for r in sn if _json(r["response"]) is None]
    print(f"spec: {len(recs)} calls, {len(err)} errors, S0 unparsed {len(s0_bad)}/{len(s0)}, "
          f"S2N unparsed {len(sn_bad)}/{len(sn)}")
    if smoke and (err or not s0 or not sn or len(s0_bad) == len(s0) or len(sn_bad) == len(sn)):
        bad = err + s0_bad + sn_bad
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
        cid, kind, world, tier, k, temp = t
        msgs = messages(world, tier, kind)
        t0 = time.time()
        try:
            text, reasoning = call_openai(a.base_url, a.model, msgs, temp, k, a.max_tokens, extra)
            err = None
        except Exception as e:  # recorded, not retried (same policy as collect.py)
            text, reasoning, err = None, None, f"{type(e).__name__}: {e}"
        return {"id": cid, "tag": a.tag, "model": a.model, "stage": "spec", "strategy": kind, "world": world,
                "level": tier, "data": "D0", "setting": None, "sample": k, "temperature": temp, "messages": msgs,
                "response": text, "reasoning": reasoning, "error": err, "seconds": round(time.time() - t0, 3)}

    with out.open("a") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
