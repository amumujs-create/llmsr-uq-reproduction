"""Collect balance-point statements for BDG2 building descriptions (docs/research/bdg2-regime-extrapolation.md §5).

One job per unique description (L1, L2) plus the model-default job M (no building details). Same record format and
OpenAI-compatible client as analysis/llm_statements/collect.py; re-running skips finished ids.

Usage:
  python3 collect_bdg2.py --base-url http://localhost:8000/v1 --model <served> --tag <tag> --out runs/bdg2/<tag>.jsonl
  python3 collect_bdg2.py --backend mock --out /tmp/mock.jsonl
  python3 collect_bdg2.py --memcheck --base-url ... --model ... --tag <tag> --out runs/bdg2/memcheck_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llm_statements"))
from collect import call_openai  # noqa: E402

N_SAMPLES, TEMP = 6, 0.7   # 6 sampled + 1 greedy = 7 calls per description (§5)

SYSTEM = ("You are an experienced building energy engineer. Answer with a single JSON object and nothing else.")

BANNED = ("flat", "plateau", "saturat", "regime", "level off", "levels off", "breakpoint", "piecewise",
          "balance point", "balance-point", "base load", "baseload", "change-point", "changepoint", "degree-day",
          "degree day")

INTRO = ("A non-residential building is served by a central chilled-water system. Daily chilled-water consumption "
         "(cooling energy delivered to the building) is recorded for two years together with the daily mean "
         "outdoor air temperature.")

QUESTION = (
    "Question: over the outdoor temperatures this building sees, is there a daily mean outdoor temperature beyond "
    "which the daily chilled-water consumption stops responding much to outdoor temperature (changes far less per "
    "degC than elsewhere)? There may be none, one on the cold side, one on the hot side, or both.\n"
    "Reply with JSON only, in this form:\n"
    '{"boundaries": [{"side": "below" or "above", "at_c": <degC>, "confidence": <0 to 1>}], '
    '"rationale": "<one or two sentences>"}\n'
    '"side": "below" means the weak response holds below at_c; "above" means above at_c. '
    "Use an empty list if you expect no such temperature."
)

MEMCHECK = (
    "Do you know the Building Data Genome Project 2 dataset or the ASHRAE Great Energy Predictor III competition? "
    "If so, which real cities or campuses do the anonymised site names Panther, Fox, Bull, Eagle, Hog and Cockatoo "
    "correspond to? Finally, for any specific building in that dataset, can you state its cooling balance-point "
    "temperature from memory? Reply with JSON: {\"knows_dataset\": true/false, \"site_cities\": {<name>: <city or "
    "null>}, \"building_balance_points\": [{\"building\": <id>, \"tc\": <degC>}], \"notes\": \"<short>\"}"
)


def describe(fields: dict, level: str) -> str:
    c = fields["climate"]
    parts = [INTRO, f"Primary use: {fields['use']}."]
    if level == "L2":
        if fields.get("sub_use"):
            parts.append(f"Specific use: {fields['sub_use']}.")
        parts.append(f"Gross floor area: {fields['sqm_bin']} square metres.")
    parts.append(f"Local climate: the daily mean outdoor temperature averages {c['mean']:.1f} degC over the year; "
                 f"the 5th and 95th percentiles of daily means are {c['p5']:.1f} and {c['p95']:.1f} degC.")
    return " ".join(parts)


def describe_default() -> str:
    """Model-default prompt M: no use, no size, no climate (§4)."""
    return INTRO


def build_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{text}\n\n{QUESTION}"}]


def check_no_leak(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BANNED if w in low]


def jobs(descriptions: dict) -> list[dict]:
    out = [{"level": "M", "desc_id": "M", "messages": build_prompt(describe_default())}]
    for level in ("L1", "L2"):
        for did, d in sorted(descriptions[level].items()):
            out.append({"level": level, "desc_id": did, "messages": build_prompt(describe(d["fields"], level))})
    return out


def call_mock(messages, temperature, seed):
    """Deterministic stand-in: a 'below' boundary that drifts with the stated mean temperature, none for M."""
    rng = np.random.default_rng(seed)
    text = messages[-1]["content"]
    if "Primary use" not in text:
        return json.dumps({"boundaries": [], "rationale": "mock default"}), None
    mean = float(text.split("averages ")[1].split(" degC")[0])
    at = 0.5 * mean + 8 + (rng.normal(0, 1.5) if temperature > 0 else 0.0)
    return json.dumps({"boundaries": [{"side": "below", "at_c": round(at, 1), "confidence": 0.6}],
                       "rationale": "mock"}), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptions", default=str(HERE / "descriptions.json"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--backend", choices=("openai", "mock"), default="openai")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="mock")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--extra-body", default="{}")
    p.add_argument("--memcheck", action="store_true", help="ask the memorisation question once (greedy) and exit")
    a = p.parse_args()
    extra = json.loads(a.extra_body)
    tag = a.tag or a.model
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.memcheck:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": MEMCHECK}]
        text, reasoning = (call_mock(msgs, 0.0, 0) if a.backend == "mock"
                           else call_openai(a.base_url, a.model, msgs, 0.0, 0, a.max_tokens, extra))
        with out.open("a") as f:
            f.write(json.dumps({"id": f"bdg2-memcheck|{tag}", "tag": tag, "model": a.model, "messages": msgs,
                                "response": text, "reasoning": reasoning}) + "\n")
        print(text)
        return
    job_list = jobs(json.load(open(a.descriptions)))
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    tasks = []
    for j in job_list:
        for k in range(a.samples + 1):
            cid = f'bdg2|{tag}|{j["level"]}|{j["desc_id"]}|{k}'
            if cid not in done:
                tasks.append((cid, j, k, 0.0 if k == a.samples else TEMP))

    def one(task):
        cid, j, k, temp = task
        t0 = time.time()
        try:
            text, reasoning = (call_mock(j["messages"], temp, k) if a.backend == "mock"
                               else call_openai(a.base_url, a.model, j["messages"], temp, k, a.max_tokens, extra))
            err = None
        except Exception as e:  # recorded, not retried
            text, reasoning, err = None, None, f"{type(e).__name__}: {e}"
        return {"id": cid, "stage": "bdg2", "model": a.model, "tag": tag, "extra_body": extra, "level": j["level"],
                "desc_id": j["desc_id"], "sample": k, "temperature": temp, "messages": j["messages"],
                "response": text, "reasoning": reasoning, "error": err, "seconds": round(time.time() - t0, 3)}

    from concurrent.futures import ThreadPoolExecutor
    with out.open("a") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        for i, rec in enumerate(ex.map(one, tasks), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done", flush=True)
    print(f"jobs={len(job_list)} calls_per_job={a.samples + 1} -> {out}")


if __name__ == "__main__":
    main()
