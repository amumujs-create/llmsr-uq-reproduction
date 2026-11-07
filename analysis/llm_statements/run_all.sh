#!/usr/bin/env bash
# Run every LLM collection the study needs, one model at a time (docs/research/llm-statement-collection.md).
# For each model in models.tsv: start vLLM, smoke test, main collection (1,596 calls), strategy experiment
# (S1/S2, 220 calls), environment record, stop vLLM. Finished parts are skipped, so just re-run after a crash.
#
# Usage:  bash analysis/llm_statements/run_all.sh            # all models in models.tsv
#         ONLY=gpt-oss-20b bash analysis/llm_statements/run_all.sh
# Env:    TP (tensor-parallel GPUs, default 4), PORT (8000), WORKERS (16), MAXTOK (4096), STRAT_SAMPLES (10),
#         EXTRA_VARGS (extra vllm serve arguments for every model, e.g. --enforce-eager),
#         ICE (default 0; 1 adds the anti-icing stage, 630 calls -> runs/ice/<tag>.jsonl,
#              docs/research/ccpp-anti-icing-world.md §6). With ICE=0 the script behaves exactly as before.
#         SHAPE (default 0; 1 adds the beyond-boundary shape stage, 252 calls -> runs/shape/<tag>.jsonl,
#                docs/research/ccpp-shape-elicitation.md §3). With SHAPE=0 nothing changes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="${TP:-4}"; PORT="${PORT:-8000}"; WORKERS="${WORKERS:-16}"; MAXTOK="${MAXTOK:-4096}"
STRAT_SAMPLES="${STRAT_SAMPLES:-10}"; ONLY="${ONLY:-}"; EXTRA_VARGS="${EXTRA_VARGS:-}"; ICE="${ICE:-0}"
SHAPE="${SHAPE:-0}"
BASE="http://localhost:$PORT/v1"
MAIN_N=1596
STRAT_N=$((2 * 2 * 5 * (STRAT_SAMPLES + 1)))
ICE_N=630
SHAPE_N=252
mkdir -p "$HERE/runs/strategy" "$HERE/runs/env" "$HERE/runs/logs"
[ "$ICE" = 1 ] && mkdir -p "$HERE/runs/ice"
[ "$SHAPE" = 1 ] && mkdir -p "$HERE/runs/shape"

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
  MAIN="$HERE/runs/$TAG.jsonl"; STRAT="$HERE/runs/strategy/$TAG.jsonl"; LOG="$HERE/runs/logs/vllm-$TAG.log"
  ICEF="$HERE/runs/ice/$TAG.jsonl"
  ice_done=1; [ "$ICE" = 1 ] && [ "$(lines "$ICEF")" -lt "$ICE_N" ] && ice_done=0
  SHAPEF="$HERE/runs/shape/$TAG.jsonl"
  shape_done=1; [ "$SHAPE" = 1 ] && [ "$(lines "$SHAPEF")" -lt "$SHAPE_N" ] && shape_done=0
  if [ "$(lines "$MAIN")" -ge "$MAIN_N" ] && [ "$(lines "$STRAT")" -ge "$STRAT_N" ] && [ "$ice_done" = 1 ] \
     && [ "$shape_done" = 1 ]; then
    echo "== $TAG: already complete, skip"; summary+=("$TAG: complete (skipped)"); continue
  fi
  # Keep the original environment record when only the anti-icing stage is left to run.
  ENVF="$HERE/runs/env/$TAG.txt"
  if [ "$(lines "$MAIN")" -ge "$MAIN_N" ] && [ "$(lines "$STRAT")" -ge "$STRAT_N" ]; then ENVF="$HERE/runs/env/$TAG-ice.txt"; fi
  # Shape-only runs get their own record too, so the anti-icing record is not overwritten.
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$SHAPE" = 1 ] && [ "$ice_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-shape.txt"; fi
  echo "== $TAG ($MODEL): starting vLLM (TP=$TP), log $LOG"
  # shellcheck disable=SC2086
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" --max-model-len 16384 $VARGS $EXTRA_VARGS > "$LOG" 2>&1 < /dev/null &
  VLLM_PID=$!
  ok=0
  for _ in $(seq 1 360); do   # up to 60 min for download + load
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then ok=1; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 10
  done
  if [ "$ok" != 1 ]; then
    echo "!! $TAG: vLLM did not start. Last log lines:"; tail -15 "$LOG"
    stop_server; summary+=("$TAG: FAILED to start vLLM (see runs/logs/vllm-$TAG.log)"); continue
  fi

  {
    echo "tag=$TAG"; echo "model=$MODEL"; echo "extra_body=$EXTRA"; echo "vllm_args=--tensor-parallel-size $TP --max-model-len 16384 $VARGS $EXTRA_VARGS"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host=$(hostname)"
    echo "vllm=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    echo "hf_revision=$(python3 -c "from huggingface_hub import model_info; print(model_info('$MODEL').sha)" 2>/dev/null)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'
  } > "$ENVF"

  if [ "$(lines "$MAIN")" -lt "$MAIN_N" ]; then
    echo "-- $TAG: main collection"
    SMOKE_FAIL=0
    WORKERS="$WORKERS" MAXTOK="$MAXTOK" bash "$HERE/run_server.sh" "$MODEL" "$TAG" "$EXTRA" "$BASE" || SMOKE_FAIL=1
    if [ "$SMOKE_FAIL" = 1 ]; then
      stop_server; summary+=("$TAG: main collection FAILED (smoke test or run; see output above)"); continue
    fi
  fi
  if [ "$(lines "$STRAT")" -lt "$STRAT_N" ]; then
    echo "-- $TAG: strategy experiment (S1/S2, $STRAT_N calls)"
    python3 "$HERE/elicit.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --samples "$STRAT_SAMPLES" \
      --max-tokens "$MAXTOK" --extra-body "$EXTRA" --workers "$WORKERS" --out "$STRAT" | tail -1
  fi
  if [ "$ICE" = 1 ] && [ "$(lines "$ICEF")" -lt "$ICE_N" ]; then
    echo "-- $TAG: anti-icing stage ($ICE_N calls)"
    ISMOKE="$HERE/runs/logs/smoke-ice-$TAG.jsonl"; rm -f "$ISMOKE"
    python3 "$HERE/collect.py" --stage ice --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --data D0 --samples 0 --out "$ISMOKE" > /dev/null
    if python3 "$HERE/check_ice.py" "$ISMOKE" --smoke; then
      python3 "$HERE/collect.py" --stage ice --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$ICEF" | tail -1
      python3 "$HERE/check_ice.py" "$ICEF"
    else
      summary+=("$TAG: anti-icing smoke test FAILED (see output above)")
    fi
  fi
  if [ "$SHAPE" = 1 ] && [ "$(lines "$SHAPEF")" -lt "$SHAPE_N" ]; then
    echo "-- $TAG: shape stage ($SHAPE_N calls)"
    SSMOKE="$HERE/runs/logs/smoke-shape-$TAG.jsonl"; rm -f "$SSMOKE"
    python3 "$HERE/collect.py" --stage shape --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$SSMOKE" > /dev/null
    if python3 "$HERE/check_shape.py" "$SSMOKE" --smoke; then
      python3 "$HERE/collect.py" --stage shape --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$SHAPEF" | tail -1
      python3 "$HERE/check_shape.py" "$SHAPEF"
    else
      summary+=("$TAG: shape smoke test FAILED (see output above)")
    fi
  fi
  stop_server
  if [ "$SHAPE" = 1 ]; then
    summary+=("$TAG: shape $(lines "$SHAPEF")/$SHAPE_N")
  fi
  if [ "$ICE" = 1 ]; then
    summary+=("$TAG: main $(lines "$MAIN")/$MAIN_N, strategy $(lines "$STRAT")/$STRAT_N, ice $(lines "$ICEF")/$ICE_N")
  else
    summary+=("$TAG: main $(lines "$MAIN")/$MAIN_N, strategy $(lines "$STRAT")/$STRAT_N")
  fi
done 3< "$HERE/models.tsv"

echo; echo "===== summary ====="; printf '%s\n' "${summary[@]}"
echo "Next: git add analysis/llm_statements/runs && git commit -m 'LLM runs' && git push"
