#!/usr/bin/env bash
# BDG2 collection on the GPU server (docs/research/bdg2-regime-extrapolation.md §5, §9), one model at a time from
# analysis/llm_statements/models.tsv: start vLLM, memorisation check (1 call), collection (215 jobs x 7 calls = 1,505),
# environment record, stop vLLM. Finished parts are skipped; re-run after a crash.
#
# Usage:  bash analysis/bdg2/run_bdg2.sh              # all models
#         ONLY=gemma3-27b bash analysis/bdg2/run_bdg2.sh
# Env:    TP (default 4), PORT (8000), WORKERS (16), MAXTOK (4096), EXTRA_VARGS
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLM="$HERE/../llm_statements"
TP="${TP:-4}"; PORT="${PORT:-8000}"; WORKERS="${WORKERS:-16}"; MAXTOK="${MAXTOK:-4096}"; ONLY="${ONLY:-}"
EXTRA_VARGS="${EXTRA_VARGS:-}"
BASE="http://localhost:$PORT/v1"
N_CALLS=$(( (1 + $(python3 -c "import json;d=json.load(open('$HERE/descriptions.json'));print(len(d['L1'])+len(d['L2']))")) * 7 ))
mkdir -p "$HERE/runs/bdg2" "$HERE/runs/env" "$HERE/runs/logs"

lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
VLLM_PID=""
stop_server() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""; }
trap stop_server EXIT

summary=()
while IFS=$'\t' read -r TAG MODEL EXTRA VARGS <&3; do
  [ -z "${TAG:-}" ] || [[ "$TAG" == \#* ]] && continue
  [ -n "$ONLY" ] && [ "$ONLY" != "$TAG" ] && continue
  [ "$EXTRA" = "-" ] && EXTRA='{}'
  [ "$VARGS" = "-" ] && VARGS=""
  OUT="$HERE/runs/bdg2/$TAG.jsonl"; MEM="$HERE/runs/bdg2/memcheck_$TAG.jsonl"; LOG="$HERE/runs/logs/vllm-bdg2-$TAG.log"
  if [ "$(lines "$OUT")" -ge "$N_CALLS" ] && [ "$(lines "$MEM")" -ge 1 ]; then
    echo "== $TAG: already complete, skip"; summary+=("$TAG: complete (skipped)"); continue
  fi
  echo "== $TAG ($MODEL): starting vLLM (TP=$TP), log $LOG"
  # shellcheck disable=SC2086
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" --max-model-len 16384 $VARGS $EXTRA_VARGS > "$LOG" 2>&1 < /dev/null &
  VLLM_PID=$!
  ok=0
  for _ in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then ok=1; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 10
  done
  if [ "$ok" != 1 ]; then
    echo "!! $TAG: vLLM did not start. Last log lines:"; tail -15 "$LOG"
    stop_server; summary+=("$TAG: FAILED to start vLLM"); continue
  fi
  {
    echo "tag=$TAG"; echo "model=$MODEL"; echo "extra_body=$EXTRA"; echo "stage=bdg2"
    echo "vllm_args=--tensor-parallel-size $TP --max-model-len 16384 $VARGS $EXTRA_VARGS"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host=$(hostname)"
    echo "vllm=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    echo "hf_revision=$(python3 -c "from huggingface_hub import model_info; print(model_info('$MODEL').sha)" 2>/dev/null)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'
  } > "$HERE/runs/env/$TAG-bdg2.txt"
  if [ "$(lines "$MEM")" -lt 1 ]; then
    echo "-- $TAG: memorisation check"
    python3 "$HERE/collect_bdg2.py" --memcheck --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --out "$MEM" | tail -3
  fi
  echo "-- $TAG: collection ($N_CALLS calls)"
  python3 "$HERE/collect_bdg2.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
    --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$OUT" | tail -1
  stop_server
  summary+=("$TAG: $(lines "$OUT")/$N_CALLS, memcheck $(lines "$MEM")/1")
done 3< "$LLM/models.tsv"

echo; echo "== summary"; printf '%s\n' "${summary[@]}"
