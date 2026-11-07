"""Check a shape-stage file: call errors, parse failures, null locations (ccpp-shape-elicitation.md §3).

Usage: python check_shape.py <runs/shape/tag.jsonl> [--smoke]
--smoke exits 1 on any call error or when more than half of the answers fail to parse (§3).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_score import parse_shape  # noqa: E402
from score_shape import null_location  # noqa: E402


def check(path):
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"] is not None]
    bad = [r for r in recs if r["error"] is None and not parse_shape(r["response"])["ok"]]
    nulls = [r for r in bad if null_location(r["response"])]
    return recs, err, bad, nulls


def main():
    path, smoke = sys.argv[1], "--smoke" in sys.argv[2:]
    recs, err, bad, nulls = check(path)
    print(f"{'smoke' if smoke else 'shape'}: calls={len(recs)} call_errors={len(err)} "
          f"parse_failures={len(bad)} (null_location={len(nulls)})")
    if smoke and (err or 2 * len(bad) > len(recs)):
        r = (err or bad)[0]
        print("error:", r["error"])
        print("response head:", (r["response"] or "")[:400])
        sys.exit(1)


if __name__ == "__main__":
    main()
