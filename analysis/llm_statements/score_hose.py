"""Score the tool-hose statements (docs/research/robot-twin-hose-llm-preregistration.md §6-§7).

Usage: python score_hose.py runs/hose/*.jsonl                     # structure scoring
       python score_hose.py runs/hose/*.jsonl --outcome [--reps 20] # + outcome scoring in the hose twin
                                                                    #   (needs data/robot/h2017.urdf and
                                                                    #    analysis/robot_twin/runs/hose_cable_truth.npz)
Structure: direction of the stated length change vs the world truth (strict error = opposite sign).
Outcome: stated eps (clipped to [-8 %, +4 %], rounded to 0.25 percentage points) goes through the
rules of analysis/robot_twin/hose_gate_mismatch.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from prompts_hose import NEUTRAL_TOL_PCT, NEUTRAL_WRONG_PCT, WORLDS  # noqa: E402

EPS_CLIP = (-0.08, 0.04)
EPS_ROUND = 0.0025
DIR_TOL_PCT = 0.5
OUTCOME_TRUTHS = ("exact", "cable", "cable+wrap")
OUTCOME_SIGMAS = (0.2, 1.0)
OUTCOME_RULES = ("R0", "R2", "safe", "gate_nom", "gate_cal", "gate_cal_infl")
CONTROLS = {"D_shrink": lambda e: -0.04, "P_rand+": lambda e: 0.02, "P_rand-": lambda e: -0.02,
            "ORACLE": lambda e: e}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def parse(text) -> dict:
    """Parse one answer. ok requires a finite length_change_pct; direction / j6_range are checked for consistency."""
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict) or _num(obj.get("length_change_pct")) is None:
        return {"ok": False}
    pct = _num(obj["length_change_pct"])
    sign_dir = "shorter" if pct < -DIR_TOL_PCT else "longer" if pct > DIR_TOL_PCT else "unchanged"
    want_range = {"shorter": "narrower", "longer": "wider", "unchanged": "unchanged"}
    d, r = obj.get("direction"), obj.get("j6_range")
    return {"ok": True, "pct": pct, "direction": d, "j6_range": r, "confidence": _num(obj.get("confidence")),
            "dir_mismatch": d is not None and d != sign_dir,
            "range_mismatch": d in want_range and r is not None and r != want_range[d]}


def structure(pct: float, world: str) -> dict:
    """Direction score of one stated percent against one world (preregistration §6)."""
    truth = 100 * WORLDS[world][1]
    if truth == 0:
        correct, strict = abs(pct) <= NEUTRAL_TOL_PCT, abs(pct) > NEUTRAL_WRONG_PCT
    else:
        correct, strict = np.sign(pct) == np.sign(truth), np.sign(pct) == -np.sign(truth)
    return {"correct": bool(correct), "strict_wrong": bool(strict), "mag_err": abs(pct - truth)}


def load(paths) -> list[dict]:
    recs = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            r["parsed"] = parse(r["response"]) if not r["error"] else {"ok": False}
            recs.append(r)
    return recs


def worlds_of(rec) -> list[str]:
    return list(WORLDS) if rec["world"] == "any" else [rec["world"]]


def structure_table(recs) -> list[str]:
    by = defaultdict(list)
    for r in recs:
        by[(r["tag"], r["level"])].append(r)
    lines = ["structure: p_dir sample / median-of-20 / T0 | strict wrong | median |pct - truth| | "
             "dir/range mismatch | parse fail"]
    for (tag, level), rs in sorted(by.items()):
        ok = [r for r in rs if r["parsed"]["ok"]]
        s = [structure(r["parsed"]["pct"], w) for r in ok if r["temperature"] > 0 for w in worlds_of(r)]
        med, t0 = [], []
        for w_rec in sorted({r["world"] for r in rs}):
            samp = [r["parsed"]["pct"] for r in ok if r["world"] == w_rec and r["temperature"] > 0]
            zero = [r["parsed"]["pct"] for r in ok if r["world"] == w_rec and r["temperature"] == 0]
            for w in (list(WORLDS) if w_rec == "any" else [w_rec]):
                if samp:
                    med.append(structure(float(np.median(samp)), w)["correct"])
                if zero:
                    t0.append(structure(zero[0], w)["correct"])
        f = lambda xs: f'{np.mean(xs):.2f}' if len(xs) else ' -  '  # noqa: E731
        lines.append(f"   {tag:24s} {level}: {f([x['correct'] for x in s])} / {f(med)} / {f(t0)} | "
                     f"{f([x['strict_wrong'] for x in s])} | "
                     f"{np.median([x['mag_err'] for x in s]) if s else float('nan'):5.2f} | "
                     f"{f([r['parsed']['dir_mismatch'] for r in ok])}/{f([r['parsed']['range_mismatch'] for r in ok])} | "
                     f"{1 - len(ok) / max(len(rs), 1):.2f} (n={len(rs)})")
    lines.append("default direction across L2 worlds (median of samples): ")
    for tag in sorted({r["tag"] for r in recs}):
        dirs = []
        for w in WORLDS:
            samp = [r["parsed"]["pct"] for r in recs if r["tag"] == tag and r["level"] == "L2" and r["world"] == w
                    and r["parsed"]["ok"] and r["temperature"] > 0]
            dirs.append('-' if not samp else 'S' if np.median(samp) < -DIR_TOL_PCT else 'L' if np.median(samp) > DIR_TOL_PCT else '0')
        lines.append(f"   {tag:24s} " + ' '.join(f'{w}:{d}' for w, d in zip(WORLDS, dirs))
                     + f"  -> {'same direction in every world' if len(set(dirs)) == 1 else 'varies'}")
    return lines


def to_eps(pct: float) -> float:
    e = min(max(pct / 100.0, EPS_CLIP[0]), EPS_CLIP[1])
    return round(round(e / EPS_ROUND) * EPS_ROUND, 6)


def statement_sets(recs) -> dict:
    """world -> {label: eps_s}. Labels tag|level|med, tag|level|t0, tag|level|s<k>, plus controls."""
    out = {w: {} for w in WORLDS}
    clipped = [0, 0]
    for r in recs:
        if not r["parsed"]["ok"]:
            continue
        e = to_eps(r["parsed"]["pct"])
        clipped[0] += e in EPS_CLIP; clipped[1] += 1
        kind = "t0" if r["temperature"] == 0 else f's{r["sample"]}'
        for w in worlds_of(r):
            out[w][f'{r["tag"]}|{r["level"]}|{kind}'] = e
    for w in WORLDS:
        groups = defaultdict(list)
        for lab, e in out[w].items():
            tag, level, kind = lab.split("|")
            if kind.startswith("s"):
                groups[(tag, level)].append(e)
        for (tag, level), es in groups.items():
            out[w][f'{tag}|{level}|med'] = to_eps(100 * float(np.median(es)))
        for name, fn in CONTROLS.items():
            out[w][f'{name}|-|ctl'] = round(fn(WORLDS[w][1]), 6)
    return out, clipped


def outcome(recs, reps, rng_seed=20261115) -> list[str]:
    sys.path.insert(0, str(ROOT / "analysis" / "robot_twin"))
    import hose_gate_mismatch as g
    from h2017_dynamics import load_urdf
    chain = load_urdf(ROOT / "data" / "robot" / "h2017.urdf")
    sweeps = np.load(ROOT / "analysis" / "robot_twin" / "runs" / "hose_cable_truth.npz")
    sets, clipped = statement_sets(recs)
    eps2world = {round(v[1], 6): w for w, v in WORLDS.items()}
    lines = [f"outcome: reps {reps}; stated eps clipped to {EPS_CLIP} in {clipped[0]}/{clipped[1]} answers; "
             "cells: MAE deg / overshoot rate (>2 deg) / loss c=3; sample = pooled over the 20 samples"]
    rng = np.random.default_rng(rng_seed)
    for truth in OUTCOME_TRUTHS:
        for sigma in OUTCOME_SIGMAS:
            rows = g.run(chain, sweeps, reps, rng, sigma, truth, statements_fn=lambda e: sets[eps2world[round(e, 6)]])
            agg = defaultdict(list)
            for lab, rule, err, _acc in rows:
                tag, level, kind = lab.split("|")
                agg[(tag, level, "sample" if kind.startswith("s") else kind, rule)].append(err)
            lines.append(f"-- truth {truth}, sigma {sigma} Nm")
            lines.append(f"   {'statement':34s}" + ''.join(f'{r:>20s}' for r in OUTCOME_RULES))
            for key in sorted({k[:3] for k in agg}):
                cells = []
                for rule in OUTCOME_RULES:
                    e = np.array(agg[key + (rule,)])
                    loss = np.mean(np.maximum(-e, 0) + 3 * np.maximum(e, 0))
                    cells.append(f'{np.mean(np.abs(e)):5.1f}/{np.mean(e > 2.0):4.2f}/{loss:5.1f}')
                lines.append(f"   {'|'.join(key):34s}" + ''.join(f'{c:>20s}' for c in cells))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outcome", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args()
    recs = load(a.files)
    lines = structure_table(recs)
    if a.outcome:
        lines += [""] + outcome(recs, a.reps)
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
