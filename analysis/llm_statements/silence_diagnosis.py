"""POST-HOC diagnosis (not a verdict): where does the cold-cap statement disappear in the HA assembly?

For every cell (model x level) and cap world, count the S2 answers that survive each step of assemble.s2_parts
for the gt_limit mechanism, and show the S0 majority. Reads the collected answers only; no model calls.
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble import L2_GT_15C_MW, SENS_GATE_PCT, _json, assemble, load  # noqa: E402
from elicit import RANGE, _num  # noqa: E402
from parse_score import TRUTH, majority  # noqa: E402

WORLDS = ("cap_strong", "cap_mild")
CELLS = [(t, lv) for t in ("gemma3-27b", "llama33-70b", "gpt-oss-20b", "gpt-oss-120b", "qwen3-30b-a3b-instruct")
         for lv in ("L1", "L2")]


def trace(obj, level):
    """Step at which one S2 answer drops the cap boundary, plus the numbers it gave."""
    g = obj.get("gt_generator_limit") or {}
    info = {"reached": g.get("reached_in_range"), "p15": _num(g.get("gt_mw_at_15c")), "lim": _num(g.get("limit_mw")),
            "sens": _num(g.get("gt_mw_per_degc")), "at": None, "pct": None}
    if g.get("reached_in_range") is not True:
        return "1 not claimed reached", info
    p15 = L2_GT_15C_MW if level == "L2" else info["p15"]
    if None in (p15, info["lim"], info["sens"]) or info["sens"] == 0 or p15 <= 0:
        return "2 number missing", info
    info["at"] = 15.0 - (info["lim"] - p15) / abs(info["sens"])
    info["pct"] = abs(info["sens"]) / p15 * 100
    if not RANGE[0] < info["at"] < RANGE[1]:
        return "3 implied temperature out of range", info
    if not SENS_GATE_PCT[0] <= info["pct"] <= SENS_GATE_PCT[1]:
        return "4 sensitivity gate", info
    return "5 kept", info


def main():
    root = HERE / "runs"
    raw = defaultdict(list)
    for path in sorted(glob.glob(str(root / "strategy/*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["strategy"] != "S2" or r["level"] not in ("L1", "L2") or r["temperature"] == 0 or r["error"]:
                continue
            obj = _json(r["response"])
            if obj is not None:
                raw[(r.get("tag", r["model"]), r["level"], r["world"])].append(obj)
    s0, s2 = load(root)
    out = ["POST-HOC diagnosis of the silent cold-cap statement. Not a verdict.",
           f"range {RANGE}, sensitivity gate {SENS_GATE_PCT} %/degC, majority needs more than half of the S2 answers."]
    for world in WORLDS:
        out.append(f"\n=== {world} (truth {TRUTH[world]}) ===")
        for tag, lv in CELLS:
            key = (tag, lv, world)
            objs = raw.get(key, [])
            steps, infos = Counter(), []
            for o in objs:
                st, info = trace(o, lv)
                steps[st] += 1
                infos.append((st, info))
            kept = [i["at"] for st, i in infos if st == "5 kept"]
            med = lambda k, cond=lambda s: True: (  # noqa: E731
                round(float(np.median([i[k] for s, i in infos if i[k] is not None and cond(s)])), 2)
                if any(i[k] is not None and cond(s) for s, i in infos) else None)
            s0m = majority(s0[key]) if s0.get(key) else None
            out.append(f"{tag}|{lv}: S2 answers {len(objs)}; " + ", ".join(f"{k}: {v}" for k, v in sorted(steps.items()))
                       + f"; kept median at {round(float(np.median(kept)), 1) if kept else None}"
                       + f"; medians lim {med('lim')} sens {med('sens')} pct {med('pct')} implied at {med('at')}"
                       + f"; S0 answers {len(s0.get(key, []))} majority {s0m}; HA {assemble(s0.get(key, []), s2.get(key, []))}")
    out.append("\n=== share of S2 answers claiming the GT limit is reached in range, by world "
               "(does the claim tell worlds apart?) ===")
    worlds = list(TRUTH)
    out.append(f"{'cell':30s} " + " ".join(f"{w:>14s}" for w in worlds))
    for tag, lv in CELLS:
        vals = []
        for w in worlds:
            objs = raw.get((tag, lv, w), [])
            vals.append(np.mean([(o.get("gt_generator_limit") or {}).get("reached_in_range") is True for o in objs])
                        if objs else float("nan"))
        out.append(f"{tag + '|' + lv:30s} " + " ".join(f"{v:14.2f}" for v in vals))
    text = "\n".join(out) + "\n"
    (root / "assembly" / "silence_diagnosis.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
