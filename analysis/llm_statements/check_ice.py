"""Check an anti-icing stage file: call errors and parse failures (ccpp-anti-icing-world.md §9 step 6).

Usage: python check_ice.py <runs/ice/tag.jsonl> [--smoke]
Ice-world answers are parsed with parse_mech, re-asked original worlds with parse. --smoke exits 1 on any failure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_score import parse, parse_mech  # noqa: E402
from prompts import ICE_WORLDS  # noqa: E402


def check(path):
    recs = [json.loads(line) for line in open(path)]
    err = [r for r in recs if r["error"] is not None]
    bad = [r for r in recs if r["error"] is None and
           not (parse_mech if r["world"] in ICE_WORLDS else parse)(r["response"])["ok"]]
    return recs, err, bad


def main():
    path, smoke = sys.argv[1], "--smoke" in sys.argv[2:]
    recs, err, bad = check(path)
    print(f"{'smoke' if smoke else 'ice'}: calls={len(recs)} call_errors={len(err)} parse_failures={len(bad)}")
    if smoke and (err or bad):
        r = (err or bad)[0]
        print("error:", r["error"])
        print("response head:", (r["response"] or "")[:400])
        sys.exit(1)


if __name__ == "__main__":
    main()
