#!/usr/bin/env bash
# Collect regime statements from one model served by an OpenAI-compatible server (vLLM / Ollama).
# Usage: bash analysis/llm_statements/run_server.sh <served-model-name> <tag> [extra-body-json] [base-url]
# Example (vLLM, Qwen with thinking off):
#   bash analysis/llm_statements/run_server.sh Qwen/xxx qwen-30b-nothink '{"chat_template_kwargs": {"enable_thinking": false}}'
set -euo pipefail
MODEL="$1"; TAG="$2"; EXTRA="${3:-}"; BASE="${4:-http://localhost:8000/v1}"
[ -z "$EXTRA" ] && EXTRA='{}'
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/runs/$TAG.jsonl"
WORKERS="${WORKERS:-16}"
MAXTOK="${MAXTOK:-4096}"

echo "[1/3] smoke test: 12 D0 prompts x 1 call"
SMOKE="$HERE/runs/smoke-$TAG.jsonl"; rm -f "$SMOKE"
python3 "$HERE/collect.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
  --max-tokens "$MAXTOK" --data D0 --samples 0 --out "$SMOKE" >/dev/null
python3 - "$SMOKE" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1].rsplit("/runs/", 1)[0])
from parse_score import parse
recs = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in recs if r["error"] or not parse(r["response"])["ok"]]
print(f"smoke: {len(recs)} calls, {len(bad)} failed")
if bad:
    r = bad[0]
    print("error:", r["error"]); print("response head:", (r["response"] or "")[:400])
    sys.exit("smoke test failed: fix server/model/--extra-body/MAXTOK before the full run")
EOF

if [ "${SMOKE_ONLY:-0}" = "1" ]; then echo "smoke only: done"; exit 0; fi

echo "[2/3] full collection -> $OUT (workers=$WORKERS, max_tokens=$MAXTOK); re-run to resume"
python3 "$HERE/collect.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
  --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$OUT"

echo "[3/3] quick check"
python3 - "$OUT" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1].rsplit("/runs/", 1)[0])
from parse_score import parse
recs = [json.loads(l) for l in open(sys.argv[1])]
err = sum(r["error"] is not None for r in recs)
bad = sum(r["error"] is None and not parse(r["response"])["ok"] for r in recs)
print(f"calls={len(recs)} call_errors={err} parse_failures={bad} mean_seconds={sum(r['seconds'] for r in recs)/len(recs):.1f}")
EOF
