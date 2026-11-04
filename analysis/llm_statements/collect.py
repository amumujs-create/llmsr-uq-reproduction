"""Collect regime statements from an OpenAI-compatible chat endpoint (e.g. vLLM on the GPU server).

Usage (see RUN.md):
  python collect.py --base-url http://localhost:8000/v1 --model <served-name> --out runs/<tag>.jsonl --workers 16
  python collect.py <Folds5x2_pp.xlsx> --base-url http://localhost:8000/v1 --model <served-name> --out runs/<tag>.jsonl
  python collect.py <xlsx> --backend mock --out /tmp/mock.jsonl      # dry run, no network
Writes one JSON line per call; re-running skips ids already in the output file.
Design: docs/research/llm-statement-collection.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
from prompts import SETTINGS, WORLDS, build_prompt  # noqa: E402

N_SAMPLES, TEMP = 20, 0.7


def window_summaries(xlsx: str) -> dict:
    """Training-window summaries (w = 2.5, rep 0) for each twin world (real V, noise 4.3) and for real CCPP."""
    import openpyxl
    from ccpp_twin import load_ccpp_inputs, simulate
    from grid import _seed, build_worlds, ols, windows
    x = load_ccpp_inputs(xlsx)
    at = x["AT"]
    feats_real = np.column_stack([at, x["V"], x["AP"], x["RH"]])
    rows = list(openpyxl.load_workbook(xlsx, read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True))
    hdr = [str(h).strip() for h in rows[0]]
    pe_real = np.array([float(r[hdr.index("PE")]) for r in rows[1:] if r[0] is not None])
    worlds = {w["module"]: w for w in build_worlds(x)[0] if w["vmode"] == "realV" and w["noise"] > 0}
    out = {}
    for wd in windows(at):
        if wd["w"] != 2.5:
            continue
        idx = np.random.default_rng(_seed(f'sample|{wd["setting"]}|{wd["w"]}|0')).choice(wd["pool"], wd["n"], replace=False)

        def summ(y):
            b, se = ols(feats_real[idx], y[idx])
            return {"n": int(len(idx)), "at_lo": float(at[idx].min()), "at_hi": float(at[idx].max()),
                    "slope": float(b[1]), "se": float(se[1])}
        for mod, w in worlds.items():
            clean = simulate(at, x["AP"], x["RH"], x["V"], w["cfg"])["pe_clean"]
            noise = np.random.default_rng(_seed(f'noise|{mod}|realV|{w["noise"]}|0')).normal(0, w["noise"], len(at))
            out[(mod, wd["setting"])] = summ(clean + noise)
        out[("real", wd["setting"])] = summ(pe_real)  # training window only
    return out


def jobs(summaries: dict) -> list[dict]:
    """All prompts for one model. L0 + D0 is world-independent and asked once; with D1 the window data differ by world."""
    def job(world, level, data, setting, messages):
        return {"world": world, "level": level, "data": data, "setting": setting, "messages": messages}
    out = [job("any", "L0", "D0", None, build_prompt("base", "L0"))]
    for world in WORLDS:
        for s in SETTINGS:
            out.append(job(world, "L0", "D1", s, build_prompt(world, "L0", summaries[(world, s)])))
        for level in ("L1", "L2"):
            out.append(job(world, level, "D0", None, build_prompt(world, level)))
            for s in SETTINGS:
                out.append(job(world, level, "D1", s, build_prompt(world, level, summaries[(world, s)])))
    out.append(job("real", "L0", "D0", None, build_prompt("base", "L0", real=True)))
    for s in SETTINGS:
        out.append(job("real", "L0", "D1", s, build_prompt("base", "L0", summaries[("real", s)], real=True)))
    return out


def call_openai(base_url, model, messages, temperature, seed, max_tokens, extra=None, timeout=600):
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            **(extra or {})}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    # vLLM reasoning parsers return the reasoning separately; keep it for the record, score only the content.
    return msg.get("content"), msg.get("reasoning_content") or msg.get("reasoning")


def call_mock(messages, temperature, seed):
    rng = np.random.default_rng(seed)
    text = messages[-1]["content"]
    b = []
    if "evaporative" in text:
        b.append({"side": "above", "at_c": float(24 + rng.normal(0, 2)), "confidence": 0.7})
    if "rated limit" in text and "stays below" not in text:
        b.append({"side": "below", "at_c": float(7 + rng.normal(0, 3)), "confidence": 0.5})
    return json.dumps({"boundaries": b, "rationale": "mock"}), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xlsx", nargs="?", default=None,
                   help="CCPP workbook; if omitted, window summaries are read from --summaries (no data needed)")
    p.add_argument("--summaries", default=str(HERE / "window_summaries.json"))
    p.add_argument("--write-summaries", action="store_true", help="compute summaries from xlsx, save, and exit")
    p.add_argument("--workers", type=int, default=1, help="concurrent requests (vLLM batches them)")
    p.add_argument("--backend", choices=("openai", "mock"), default="openai")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="mock")
    p.add_argument("--out", help="output JSONL (required unless --write-summaries)")
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--max-tokens", type=int, default=4096, help="large enough for reasoning models")
    p.add_argument("--extra-body", default="{}", help='e.g. \'{"reasoning_effort": "none"}\' (Ollama) or '
                   '\'{"chat_template_kwargs": {"enable_thinking": false}}\' (vLLM + Qwen)')
    p.add_argument("--data", choices=("D0", "D1", "all"), default="all", help="restrict to one data condition")
    p.add_argument("--tag", default=None, help="model label in ids/output (defaults to --model)")
    a = p.parse_args()
    if a.xlsx:
        summaries = window_summaries(a.xlsx)
        if a.write_summaries:
            json.dump({f"{w}|{s}": v for (w, s), v in summaries.items()}, open(a.summaries, "w"), indent=1)
            print(f"wrote {a.summaries}")
            return
    else:
        summaries = {tuple(k.split("|")): v for k, v in json.load(open(a.summaries)).items()}
    if not a.out:
        p.error("--out is required")
    job_list = [j for j in jobs(summaries) if a.data == "all" or j["data"] == a.data]
    extra = json.loads(a.extra_body)
    tag = a.tag or a.model
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open()}
    tasks = []
    for j in job_list:
        for k in range(a.samples + 1):
            cid = f'{tag}|{j["world"]}|{j["level"]}|{j["data"]}|{j["setting"]}|{k}'
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
        return {"id": cid, "model": a.model, "tag": tag, "extra_body": extra, "world": j["world"],
                "level": j["level"], "data": j["data"], "setting": j["setting"], "sample": k, "temperature": temp,
                "messages": j["messages"], "response": text, "reasoning": reasoning, "error": err,
                "seconds": round(time.time() - t0, 3)}

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
