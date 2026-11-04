"""Second scoring of LLM answers against literature ranges (docs/research/ccpp-domain-check.md §8).

Exploratory: the ranges were fixed after seeing the gpt-oss-120b answers. Twin-truth scoring is unchanged.
Usage: python lit_rescore.py --out runs/lit_rescore
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_score import parse, score, to_statement  # noqa: E402

SENS_PCT = (0.5, 0.9)        # gas-turbine output change, % per degC (domain-check §2)
SENS_PCT_LOOSE = (0.4, 1.2)
ICE_RANGE = (-15.0, 8.0)     # anti-icing operating band, degC (domain-check §4)
CAP_WORLDS = ("cap_mild", "cap_strong", "evap+cap_mild")
MODELS = {"pilot-mistral7b": "mistral 7B", "pilot-qwen35-9b-nothink": "qwen3.5 9B", "pilot-llama31-8b": "llama3.1 8B",
          "pilot-gemma3-12b": "gemma3 12B", "gpt-oss-120b": "gpt-oss-120b",
          "gpt-oss-20b": "gpt-oss-20b", "qwen3-30b-a3b-instruct": "qwen3 30B"}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _json(text):
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def s2_rows():
    for path in glob.glob(str(HERE / "runs/strategy/*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["strategy"] != "S2" or r["temperature"] == 0 or r["error"]:
                continue
            o = _json(r["response"])
            if o is None:
                continue
            g = o.get("gt_generator_limit") or {}
            sens, p15 = _num(g.get("gt_mw_per_degc")), _num(g.get("gt_mw_at_15c"))
            if r["level"] == "L2":
                p15 = 343.0  # given in the prompt
            pct = abs(sens) / p15 * 100 if sens is not None and p15 else None
            ice = o.get("inlet_anti_icing_heating") or {}
            ice_at = _num(ice.get("below_c"))
            ice_on = ice.get("active") is True
            yield {"model": MODELS.get(r["tag"], r["tag"]), "level": r["level"], "world": r["world"], "sens_pct": pct,
                   "sens_lit": pct is not None and SENS_PCT[0] <= pct <= SENS_PCT[1],
                   "sens_lit_loose": pct is not None and SENS_PCT_LOOSE[0] <= pct <= SENS_PCT_LOOSE[1],
                   "sens_twin": sens is not None and abs(abs(sens) / (p15 or 1) * 343 - 2.2) <= 0.7,
                   "ice_on": ice_on,
                   "ice_plausible": (not ice_on) or (ice_at is not None and ICE_RANGE[0] <= ice_at <= ICE_RANGE[1])}


def s0_rows():
    paths = glob.glob(str(HERE / "runs/pilot/*.jsonl")) + glob.glob(str(HERE / "runs/*.jsonl"))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if (r["world"] in ("any", "real") or r.get("data") != "D0" or r["level"] not in ("L1", "L2")
                    or r["temperature"] == 0 or r["error"]):
                continue
            p = parse(r["response"])
            if not p["ok"]:
                continue
            stmt = to_statement(p["boundaries"])
            s = score(r["world"], stmt)
            cold_at = [a for side, a in stmt if side == "below"]
            # A cold false alarm is literature-plausible if it sits in the anti-icing band (<= 8 degC).
            cold_plaus = s["cold"] == "false_alarm" and cold_at and cold_at[0] <= ICE_RANGE[1]
            lit_ok = s["all_correct"] or (cold_plaus and s["hot"] == "correct")
            yield {"model": MODELS.get(r.get("tag", r["model"]), r.get("tag")), "level": r["level"], "world": r["world"],
                   "twin_ok": s["all_correct"], "cold": s["cold"], "cold_plausible_fa": bool(cold_plaus),
                   "lit_ok": bool(lit_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs/lit_rescore"))
    a = ap.parse_args()
    lines, out = [], {}
    s2 = pd.DataFrame(list(s2_rows()))
    t = s2.groupby(["model", "level"]).agg(n=("sens_lit", "size"), sens_pct_median=("sens_pct", "median"),
                                           sens_twin=("sens_twin", "mean"), sens_lit=("sens_lit", "mean"),
                                           sens_lit_loose=("sens_lit_loose", "mean"), ice_on=("ice_on", "mean"),
                                           ice_plausible=("ice_plausible", "mean")).round(2)
    out["s2"] = {"|".join(k): v for k, v in t.to_dict("index").items()}
    lines += ["=== S2 quantities: twin vs literature scoring ===", t.to_string()]

    s0 = pd.DataFrame(list(s0_rows()))
    fa = s0[s0.cold == "false_alarm"]
    t0 = s0.groupby(["model", "level"]).agg(n=("twin_ok", "size"), p_twin=("twin_ok", "mean"),
                                            p_lit_upper=("lit_ok", "mean")).round(2)
    t0["cold_false_alarm"] = s0.assign(f=s0.cold == "false_alarm").groupby(["model", "level"]).f.mean().round(2)
    t0["of_which_plausible"] = fa.groupby(["model", "level"]).cold_plausible_fa.mean().round(2)
    out["s0"] = {"|".join(k): v for k, v in t0.fillna(0).to_dict("index").items()}
    lines += ["\n=== S0 statements: twin p vs literature-adjusted upper bound ===", t0.fillna(0).to_string()]
    text = "\n".join(lines)
    Path(a.out + ".txt").write_text(text)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
