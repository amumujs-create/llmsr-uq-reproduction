"""EXPLORATION (not a verdict): does asking for a quoted sentence of evidence reduce false alarms?

docs/research/ccpp-twin-explore-cite.md. Development models only (models.tsv). Compares, answer by answer:
  before : runs/cleanworld/<tag>.jsonl  (S0, S2; the questions without the evidence field)
  after  : runs/cleancite/<tag>.jsonl   (S0C, S2C; same descriptions, evidence field added)
  checked: after, keeping a claim only when its evidence passes collect_cleancite.quote_ok

A claim "on a side" is an S0 boundary with that side, or an S2 device that points to that side. The true side of a
plant is the side its device acts on; the base plant has no true side. Temperature-0 calls are left out, as in
clean_diag.py. The base plant has no "before" column here (its earlier answers come from other stages).

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/cite_check.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import clean_worlds as cw  # noqa: E402
import collect_cleancite as ct  # noqa: E402
import outside_fallback as fb  # noqa: E402
from assemble import _json  # noqa: E402
from parse_score import parse  # noqa: E402

SIDE_WORD = {"cold": "below", "hot": "above"}
ITEM_OF = {m: key for m, (key, _) in fb.CLAIM_KEY.items()}          # mechanism name -> S2 item name
ITEM_SIDE = {ITEM_OF[m]: side for m, side in fb.SIDE.items()}       # S2 item name -> "below" / "above"


def true_side(world):
    return None if world == ct.BASE_WORLD else SIDE_WORD[cw.SIDE[world]]


def dev_tags(path):
    return [line.split("\t")[0] for line in open(path) if line.strip() and not line.startswith("#")]


def sides_s0(text, description=None):
    """Sides claimed by one S0 / S0C answer, or None if unparsed. With a description, only quoted claims count."""
    if not parse(text)["ok"]:
        return None
    out = set()
    for b in _json(text)["boundaries"]:
        if description is None or ct.quote_ok(b.get("evidence"), description):
            out.add(str(b["side"]).strip().lower())
    return out


def items_s2(text, description=None):
    """S2 items claimed by one S2 / S2C answer, or None if unparsed. With a description, only quoted claims count."""
    obj = _json(text)
    if obj is None:
        return None
    return {item for item, (claimed, ev) in ct.s2c_evidence(obj).items()
            if claimed and (description is None or ct.quote_ok(ev, description))}


def load(runs, stage, tags):
    recs = []
    for tag in tags:
        path = Path(runs) / stage / f"{tag}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing answers: {path}")
        for line in open(path):
            r = json.loads(line)
            if r["temperature"] == 0 or r["error"] or r["world"] not in ct.WORLDS:
                continue
            recs.append(r)
    return recs


def rows(recs, variant):
    """One row per parsed answer and side: claimed or not, and whether that side is the plant's true side."""
    out, unparsed, quoted = [], Counter(), Counter()
    for r in recs:
        kind = r["strategy"][:2]                                      # S0C -> S0, S2C -> S2
        desc = ct.describe_cite(r["world"]) if variant == "checked" else None
        if kind == "S0":
            sides, items = sides_s0(r["response"], desc), None
        else:
            items = items_s2(r["response"], desc)
            sides = None if items is None else {ITEM_SIDE[i] for i in items}
        if sides is None:
            unparsed[(r["tag"], kind)] += 1
            continue
        if variant == "checked" and kind == "S2":
            for i in items:
                quoted[i] += 1
        for side in ("below", "above"):
            ts = true_side(r["world"])
            out.append(dict(tag=r["tag"], world=r["world"], kind=kind, variant=variant, side=side,
                            truth="true" if side == ts else "false", claimed=side in sides,
                            base=r["world"] == ct.BASE_WORLD))
    return out, unparsed, quoted


def item_table(recs):
    """Share of S2 / S2C answers claiming each item, split by whether the item's side is the true side."""
    n, c = Counter(), Counter()
    for r in recs:
        if not r["strategy"].startswith("S2"):
            continue
        items = items_s2(r["response"])
        if items is None:
            continue
        ts = true_side(r["world"])
        for item, side in ITEM_SIDE.items():
            key = (item, "true" if side == ts else "false")
            n[key] += 1
            c[key] += item in items
    return pd.Series({k: c[k] / n[k] for k in sorted(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE.parent.parent / "llm_statements" / "runs"))
    ap.add_argument("--models", default=str(HERE.parent.parent / "llm_statements" / "models.tsv"))
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_cite"))
    a = ap.parse_args()
    tags = dev_tags(a.models)
    before = load(a.runs, "cleanworld", tags)
    after = load(a.runs, "cleancite", tags)
    parts, notes = [], []
    for recs, variant in ((before, "before"), (after, "after"), (after, "checked")):
        r, unparsed, quoted = rows(recs, variant)
        parts += r
        notes.append(f"{variant}: answers {len(recs)}, unparsed {dict(unparsed)}")
        if variant == "checked":
            notes.append(f"checked: S2C claims that passed the quote check, by item: {dict(quoted)}")
    df = pd.DataFrame(parts)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "claims.csv.gz", index=False)
    pd.set_option("display.width", 220)
    order = ["before", "after", "checked"]
    device = df[~df["base"]]
    L = ["EXPLORATION (not a verdict). Share of answers claiming a boundary / device on a side. "
         f"models={tags}", *notes, "",
         "=== device plants, pooled over models (rows: question, true/false side) ===",
         device.pivot_table(index=["kind", "truth"], columns="variant", values="claimed").reindex(columns=order)
         .round(3).to_string(), "",
         "=== device plants, by model ===",
         device.pivot_table(index=["tag", "kind", "truth"], columns="variant", values="claimed")
         .reindex(columns=order).round(3).to_string(), "",
         "=== device plants, by plant (false side only) ===",
         device[device["truth"] == "false"].pivot_table(index=["world", "kind"], columns="variant", values="claimed")
         .reindex(columns=order).round(3).to_string(), "",
         "=== base plant (no device; every claim is a false alarm), by model and side ===",
         df[df["base"]].pivot_table(index=["tag", "kind", "side"], columns="variant", values="claimed")
         .reindex(columns=[v for v in order if v != "before"]).round(3).to_string(), "",
         "=== S2 items claimed, share of answers (before = S2, after = S2C, no quote check) ===",
         pd.DataFrame({"before": item_table(before), "after": item_table(after)}).round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
