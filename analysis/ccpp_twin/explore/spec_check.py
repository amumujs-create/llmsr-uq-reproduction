"""EXPLORATION (not a verdict): do models derive the cold-side kink from the generator rating and power factor?

docs/research/ccpp-twin-indirect-spec-design.md §6 and §8. Development models only (models.tsv). Answer level only;
no prediction error is computed here. Temperature-0 calls are left out.

Measures (per tier, question, model and plant group):
  A  share of answers claiming a cold-side boundary / device (kink plants: true, no-kink plants: false)
  B  S2N: share whose limit_mw is within LIMIT_TOL_MW of rating x power factor
  C  position error against the twin (kink plants, answers that claim the generator limit / a cold boundary)
  D  the three control pairs (§3)
  E  share of answers claiming a hot-side boundary / device (always false here)
  F  S2N: share answering "not stated" for the generator item
The reading in §8 uses tier T2, question S2N. Thresholds are the ones written there.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/spec_check.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import ccpp_twin as tw  # noqa: E402
import clean_worlds as cw  # noqa: E402
import collect_spec as sp  # noqa: E402
from assemble import _json  # noqa: E402
from collect_cleancite import S2_FLAG  # noqa: E402
from elicit import assemble_s2  # noqa: E402
from parse_score import parse  # noqa: E402

# Medians of the CCPP inputs (AP mbar, RH %, V cm Hg); the kink position of a plant is where the uncapped
# gas-turbine output of the twin crosses the limit under these conditions.
MEDIAN_AP, MEDIAN_RH, MEDIAN_V = 1012.94, 74.975, 52.08
AT_RANGE = (1.81, 37.11)
LIMIT_TOL_MW = 1.0
POS_TOL_C = 2.0
GEN, ICE = "gt_generator_limit", "inlet_anti_icing_heating"
HOT_ITEMS = ("inlet_evaporative_cooling", "exhaust_or_steam_side_limit")
# §8 thresholds (written before collection).
PASS_KINK, PASS_NOKINK, PASS_POS, PASS_MODELS = 0.80, 0.20, 0.70, 3
FOLD_NOKINK, FOLD_KINK = 0.50, 0.50
# §3 control pairs. Thresholds fixed with the code, before collection: a position must move by at least half the
# expected distance, a claim share by at least 0.5, and equal limits must give positions within POS_TOL_C.
PAIR_MOVE = ("g400", "g400p")          # same rating, power factor 0.85 -> 0.90: kink moves colder
PAIR_VANISH = ("g425", "n425p")        # same rating, power factor 0.85 -> 0.90: kink disappears
PAIR_SAME = ("g415", "g392p")          # different rating and power factor, same limit


def kink_position(limit_mw: float):
    """Ambient temperature (degC) at which the twin's uncapped GT output equals limit_mw, or None if it never does
    inside the CCPP temperature range."""
    at = np.linspace(AT_RANGE[0], AT_RANGE[1], 3531)
    gt = tw.simulate(at, np.full_like(at, MEDIAN_AP), np.full_like(at, MEDIAN_RH), np.full_like(at, MEDIAN_V),
                     tw.with_modules())["p_gt"]
    if not gt.min() < limit_mw < gt.max():
        return None
    return float(np.interp(limit_mw, gt[::-1], at[::-1]))        # gt falls as at rises


def twin_config(plant: str) -> tw.TwinConfig:
    """The twin configuration a plant stands for (used when prediction tasks are built in the second step)."""
    return tw.with_modules(**cw.SHARP_CAP, gt_cap_mw=sp.limit_mw(plant))


def dev_tags(path):
    return [line.split("\t")[0] for line in open(path) if line.strip() and not line.startswith("#")]


def read_s0(text):
    """One S0 answer -> dict, or None if unparsed. Position is the innermost cold-side boundary."""
    p = parse(text)
    if not p["ok"]:
        return None
    below = [at for side, at, _ in p["boundaries"] if side == "below"]
    above = [at for side, at, _ in p["boundaries"] if side == "above"]
    return dict(cold=bool(below), hot=bool(above), gen=bool(below), position=max(below) if below else None,
                limit=None, not_stated=None)


def read_s2n(text):
    """One S2N answer -> dict, or None if unparsed. Only a JSON true is a claim, as in the scored rule."""
    obj = _json(text)
    if not isinstance(obj, dict):
        return None
    flag = {item: (obj.get(item) or {}).get(S2_FLAG[item]) if isinstance(obj.get(item), dict) else None
            for item in S2_FLAG}
    g = obj.get(GEN) if isinstance(obj.get(GEN), dict) else {}
    try:
        limit = float(g.get("limit_mw"))
    except (TypeError, ValueError):
        limit = None
    gen = flag[GEN] is True
    ns = isinstance(flag[GEN], str) and flag[GEN].strip().lower().replace("_", " ") == "not stated"
    return dict(cold=gen or flag[ICE] is True, hot=any(flag[i] is True for i in HOT_ITEMS), gen=gen,
                position=assemble_s2(text)["parts"].get("gt_limit") if gen else None, limit=limit, not_stated=ns)


def load(runs, tags):
    recs = []
    for tag in tags:
        path = Path(runs) / "spec" / f"{tag}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing answers: {path}")
        for line in open(path):
            r = json.loads(line)
            if r["temperature"] == 0 or r["error"]:
                continue
            recs.append(r)
    return recs


def rows(recs):
    truth = {p: kink_position(sp.limit_mw(p)) for p in sp.PLANTS}
    out, unparsed = [], Counter()
    for r in recs:
        a = read_s0(r["response"]) if r["strategy"] == "S0" else read_s2n(r["response"])
        if a is None:
            unparsed[(r["tag"], r["level"], r["strategy"])] += 1
            continue
        plant = r["world"]
        group = "T0" if plant == sp.T0_WORLD else "kink" if plant in sp.KINK else "nokink"
        k = truth.get(plant)
        err = abs(a["position"] - k) if a["position"] is not None and k is not None else None
        target = None if plant == sp.T0_WORLD else sp.limit_mw(plant)
        limit_ok = None if a["limit"] is None or target is None else abs(a["limit"] - target) <= LIMIT_TOL_MW
        if r["strategy"] == "S2N" and target is not None and a["limit"] is None:
            limit_ok = False                                    # no usable number counts as not computed
        out.append(dict(tag=r["tag"], tier=r["level"], kind=r["strategy"], plant=plant, group=group, cold=a["cold"],
                        hot=a["hot"], gen=a["gen"], position=a["position"], pos_err=err,
                        pos_ok=None if err is None else err <= POS_TOL_C, limit_ok=limit_ok,
                        not_stated=a["not_stated"]))
    return pd.DataFrame(out), unparsed, truth


def model_table(df, tier="T2", kind="S2N"):
    """Per model: the three §8 quantities at one tier and question."""
    d = df[(df["tier"] == tier) & (df["kind"] == kind)]
    out = {}
    for tag, g in d.groupby("tag"):
        kink, nokink = g[g["group"] == "kink"], g[g["group"] == "nokink"]
        claims = kink[kink["gen"] & kink["pos_ok"].notna()]
        out[tag] = dict(kink_claim=kink["cold"].mean(), nokink_claim=nokink["cold"].mean(),
                        pos_within=claims["pos_ok"].astype(float).mean() if len(claims) else float("nan"),
                        pos_err_median=claims["pos_err"].median() if len(claims) else float("nan"),
                        n_claims=len(claims))
    return pd.DataFrame(out).T


def passes(row) -> bool:
    return bool(row["kink_claim"] >= PASS_KINK and row["nokink_claim"] <= PASS_NOKINK
                and row["pos_within"] >= PASS_POS)


def verdict(df, tier="T2", kind="S2N"):
    """§8 reading: (label, per-model table with a 'passes' column, pooled row)."""
    mt = model_table(df, tier, kind)
    mt["passes"] = [passes(r) for _, r in mt.iterrows()]
    pooled = model_table(df.assign(tag="pooled"), tier, kind).iloc[0]
    if passes(pooled) and int(mt["passes"].sum()) >= PASS_MODELS:
        label = "PASS"
    elif pooled["kink_claim"] >= PASS_KINK and pooled["nokink_claim"] >= FOLD_NOKINK:
        label = "FOLD: claims without calculation"
    elif pooled["kink_claim"] < FOLD_KINK:
        label = "FOLD: no claim"
    else:
        label = "IN BETWEEN"
    return label, mt, pooled


def pair_table(df, truth, tier="T2", kind="S2N"):
    """Per model: do the answers follow the three control pairs?"""
    d = df[(df["tier"] == tier) & (df["kind"] == kind)]
    need = 0.5 * (truth[PAIR_MOVE[0]] - truth[PAIR_MOVE[1]])
    out = {}
    for tag, g in d.groupby("tag"):
        pos = g[g["gen"]].groupby("plant")["position"].median()
        claim = g.groupby("plant")["cold"].mean()
        move = pos.get(PAIR_MOVE[0], np.nan) - pos.get(PAIR_MOVE[1], np.nan)
        vanish = claim.get(PAIR_VANISH[0], np.nan) - claim.get(PAIR_VANISH[1], np.nan)
        same = abs(pos.get(PAIR_SAME[0], np.nan) - pos.get(PAIR_SAME[1], np.nan))
        out[tag] = dict(move_c=move, move_ok=bool(move >= need), vanish=vanish, vanish_ok=bool(vanish >= 0.5),
                        same_c=same, same_ok=bool(same <= POS_TOL_C))
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE.parent.parent / "llm_statements" / "runs"))
    ap.add_argument("--models", default=str(HERE.parent.parent / "llm_statements" / "models.tsv"))
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_spec"))
    a = ap.parse_args()
    tags = dev_tags(a.models)
    recs = load(a.runs, tags)
    df, unparsed, truth = rows(recs)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "answers.csv.gz", index=False)
    pd.set_option("display.width", 220)
    label, mt, pooled = verdict(df)
    plants = df[df["group"] != "T0"]
    t0 = df[df["group"] == "T0"]
    s2n = plants[plants["kind"] == "S2N"]
    gen_claims = plants[plants["gen"] & plants["pos_err"].notna()].assign(pos_ok=lambda x: x["pos_ok"].astype(float))
    L = ["EXPLORATION (not a verdict). Indirect specifications: generator rating and power factor. "
         f"models={tags}", f"answers {len(recs)}, unparsed {dict(unparsed)}", "",
         "=== plants: limit (MW) and kink position in the twin (degC; None = no kink in range) ===",
         pd.DataFrame({"limit_mw": {p: sp.limit_mw(p) for p in sp.PLANTS}, "kink_c": truth}).round(2).to_string(), "",
         f"=== §8 reading (tier T2, question S2N): {label} ===",
         "per model:", mt.astype(float).round(3).to_string(),
         "pooled:", pooled.astype(float).round(3).to_string(), "",
         "=== A, E: share of answers claiming the cold side / the hot side (rows: tier, question, plant group) ===",
         plants.pivot_table(index=["tier", "kind", "group"], values=["cold", "hot"]).round(3).to_string(), "",
         "=== A by model (cold-side claims) ===",
         plants.pivot_table(index=["tier", "kind", "group"], columns="tag", values="cold").round(3).to_string(), "",
         "=== A by plant (cold-side claims, pooled over models) ===",
         plants.pivot_table(index="plant", columns=["tier", "kind"], values="cold").reindex(list(sp.PLANTS))
         .round(3).to_string(), "",
         "=== tier T0 (no specification): share of answers claiming each side, by model ===",
         t0.pivot_table(index=["kind", "tag"], values=["cold", "hot"]).round(3).to_string(), "",
         f"=== B: S2N limit_mw within {LIMIT_TOL_MW} MW of rating x power factor (all parsed answers) ===",
         s2n.assign(limit_ok=s2n["limit_ok"].astype(float))
         .pivot_table(index=["tier", "group"], columns="tag", values="limit_ok").round(3).to_string(), "",
         f"=== C: position error, kink plants, answers that claim (share within {POS_TOL_C} degC; median degC) ===",
         gen_claims.pivot_table(index=["tier", "kind"], columns="tag", values="pos_ok").round(3).to_string(),
         gen_claims.pivot_table(index=["tier", "kind"], columns="tag", values="pos_err", aggfunc="median")
         .round(2).to_string(), "",
         "=== D: control pairs at T2 / S2N (move_c expected "
         f"{truth[PAIR_MOVE[0]] - truth[PAIR_MOVE[1]]:.1f}; vanish expected 1; same_c expected 0) ===",
         pair_table(df, truth).to_string(), "",
         "=== D: control pairs at T2 / S0 ===", pair_table(df, truth, kind="S0").to_string(), "",
         "=== F: S2N share answering 'not stated' for the generator item ===",
         s2n.assign(not_stated=s2n["not_stated"].astype(float))
         .pivot_table(index=["tier", "group"], columns="tag", values="not_stated").round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
