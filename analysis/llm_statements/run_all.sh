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
#         SHAPE2 (default 0; 1 adds shape stage 2, 210 calls -> runs/shape2/<tag>.jsonl,
#                 docs/research/ccpp-shape-elicitation-2.md §3). With SHAPE2=0 nothing changes.
#         NEWWORLD (default 0; 1 adds the new-world stage, 128 calls -> runs/newworld/<tag>.jsonl,
#                   docs/research/ccpp-twin-outside-confirmation.md §4). With NEWWORLD=0 nothing changes.
#         CLEANWORLD (default 0; 1 adds the clean-world stage, 192 calls -> runs/cleanworld/<tag>.jsonl,
#                     docs/research/ccpp-twin-clean-collection.md). With CLEANWORLD=0 nothing changes.
#         CLEANCITE (default 0; 1 adds the evidence-citation exploration stage, 301 calls ->
#                    runs/cleancite/<tag>.jsonl, docs/research/ccpp-twin-explore-cite.md). Development models
#                    only (models.tsv). With CLEANCITE=0 nothing changes.
#         SPEC (default 0; 1 adds the indirect-specification exploration stage, 608 calls ->
#               runs/spec/<tag>.jsonl, docs/research/ccpp-twin-indirect-spec-design.md). Development models
#               only (models.tsv). With SPEC=0 nothing changes.
#         MODELS (default models.tsv; a file name in this directory, e.g. models_new.tsv).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="${TP:-4}"; PORT="${PORT:-8000}"; WORKERS="${WORKERS:-16}"; MAXTOK="${MAXTOK:-4096}"
STRAT_SAMPLES="${STRAT_SAMPLES:-10}"; ONLY="${ONLY:-}"; EXTRA_VARGS="${EXTRA_VARGS:-}"; ICE="${ICE:-0}"
SHAPE="${SHAPE:-0}"; SHAPE2="${SHAPE2:-0}"; NEWWORLD="${NEWWORLD:-0}"; CLEANWORLD="${CLEANWORLD:-0}"; CLEANCITE="${CLEANCITE:-0}"; SPEC="${SPEC:-0}"; MODELS="${MODELS:-models.tsv}"
BASE="http://localhost:$PORT/v1"
MAIN_N=1596
STRAT_N=$((2 * 2 * 5 * (STRAT_SAMPLES + 1)))
ICE_N=630
SHAPE_N=252
SHAPE2_N=210
NEWWORLD_N=128
CLEANWORLD_N=192
CLEANCITE_N=301
SPEC_N=608
mkdir -p "$HERE/runs/strategy" "$HERE/runs/env" "$HERE/runs/logs"
[ "$ICE" = 1 ] && mkdir -p "$HERE/runs/ice"
[ "$SHAPE" = 1 ] && mkdir -p "$HERE/runs/shape"
[ "$SHAPE2" = 1 ] && mkdir -p "$HERE/runs/shape2"
[ "$NEWWORLD" = 1 ] && mkdir -p "$HERE/runs/newworld"
[ "$CLEANWORLD" = 1 ] && mkdir -p "$HERE/runs/cleanworld"
if [ "$CLEANCITE" = 1 ] && [ "$MODELS" != "models.tsv" ]; then
  echo "CLEANCITE is an exploration on the development models only: use MODELS=models.tsv"; exit 1; fi
[ "$CLEANCITE" = 1 ] && mkdir -p "$HERE/runs/cleancite"
if [ "$SPEC" = 1 ] && [ "$MODELS" != "models.tsv" ]; then
  echo "SPEC is an exploration on the development models only: use MODELS=models.tsv"; exit 1; fi
[ "$SPEC" = 1 ] && mkdir -p "$HERE/runs/spec"

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
  SHAPE2F="$HERE/runs/shape2/$TAG.jsonl"
  shape2_done=1; [ "$SHAPE2" = 1 ] && [ "$(lines "$SHAPE2F")" -lt "$SHAPE2_N" ] && shape2_done=0
  NEWF="$HERE/runs/newworld/$TAG.jsonl"
  new_done=1; [ "$NEWWORLD" = 1 ] && [ "$(lines "$NEWF")" -lt "$NEWWORLD_N" ] && new_done=0
  CLEANF="$HERE/runs/cleanworld/$TAG.jsonl"
  clean_done=1; [ "$CLEANWORLD" = 1 ] && [ "$(lines "$CLEANF")" -lt "$CLEANWORLD_N" ] && clean_done=0
  CITEF="$HERE/runs/cleancite/$TAG.jsonl"
  cite_done=1; [ "$CLEANCITE" = 1 ] && [ "$(lines "$CITEF")" -lt "$CLEANCITE_N" ] && cite_done=0
  SPECF="$HERE/runs/spec/$TAG.jsonl"
  spec_done=1; [ "$SPEC" = 1 ] && [ "$(lines "$SPECF")" -lt "$SPEC_N" ] && spec_done=0
  if [ "$(lines "$MAIN")" -ge "$MAIN_N" ] && [ "$(lines "$STRAT")" -ge "$STRAT_N" ] && [ "$ice_done" = 1 ] \
     && [ "$shape_done" = 1 ] && [ "$shape2_done" = 1 ] && [ "$new_done" = 1 ] && [ "$clean_done" = 1 ] && [ "$cite_done" = 1 ] && [ "$spec_done" = 1 ]; then
    echo "== $TAG: already complete, skip"; summary+=("$TAG: complete (skipped)"); continue
  fi
  # Keep the original environment record when only the anti-icing stage is left to run.
  ENVF="$HERE/runs/env/$TAG.txt"
  if [ "$(lines "$MAIN")" -ge "$MAIN_N" ] && [ "$(lines "$STRAT")" -ge "$STRAT_N" ]; then ENVF="$HERE/runs/env/$TAG-ice.txt"; fi
  # Shape-only runs get their own record too, so the anti-icing record is not overwritten.
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$SHAPE" = 1 ] && [ "$ice_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-shape.txt"; fi
  # Shape stage 2 gets its own record (takes precedence when both shape stages are requested).
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$SHAPE2" = 1 ] && [ "$ice_done" = 1 ] && [ "$shape_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-shape2.txt"; fi
  # A new-world-only run (everything else already complete) gets its own record.
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$NEWWORLD" = 1 ] && [ "$ice_done" = 1 ] && [ "$shape_done" = 1 ] \
     && [ "$shape2_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-newworld.txt"; fi
  # A clean-world-only run (everything else already complete) gets its own record.
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$CLEANWORLD" = 1 ] && [ "$ice_done" = 1 ] && [ "$shape_done" = 1 ] \
     && [ "$shape2_done" = 1 ] && [ "$new_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-cleanworld.txt"; fi
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$CLEANCITE" = 1 ] && [ "$ice_done" = 1 ] && [ "$shape_done" = 1 ] \
     && [ "$shape2_done" = 1 ] && [ "$new_done" = 1 ] && [ "$clean_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-cleancite.txt"; fi
  # A specification-only run (everything else already complete) gets its own record.
  if [ "$ENVF" != "$HERE/runs/env/$TAG.txt" ] && [ "$SPEC" = 1 ] && [ "$ice_done" = 1 ] && [ "$shape_done" = 1 ] \
     && [ "$shape2_done" = 1 ] && [ "$new_done" = 1 ] && [ "$clean_done" = 1 ] && [ "$cite_done" = 1 ]; then
    ENVF="$HERE/runs/env/$TAG-spec.txt"; fi
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
  if [ "$SHAPE2" = 1 ] && [ "$(lines "$SHAPE2F")" -lt "$SHAPE2_N" ]; then
    echo "-- $TAG: shape stage 2 ($SHAPE2_N calls)"
    S2SMOKE="$HERE/runs/logs/smoke-shape2-$TAG.jsonl"; rm -f "$S2SMOKE"
    python3 "$HERE/collect.py" --stage shape2 --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$S2SMOKE" > /dev/null
    if python3 "$HERE/check_shape.py" "$S2SMOKE" --smoke; then
      python3 "$HERE/collect.py" --stage shape2 --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$SHAPE2F" | tail -1
      python3 "$HERE/check_shape.py" "$SHAPE2F"
    else
      summary+=("$TAG: shape2 smoke test FAILED (see output above)")
    fi
  fi
  if [ "$NEWWORLD" = 1 ] && [ "$(lines "$NEWF")" -lt "$NEWWORLD_N" ]; then
    echo "-- $TAG: new-world stage ($NEWWORLD_N calls)"
    NSMOKE="$HERE/runs/logs/smoke-newworld-$TAG.jsonl"; rm -f "$NSMOKE"
    python3 "$HERE/collect_newworld.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$NSMOKE" > /dev/null
    if python3 "$HERE/collect_newworld.py" --check "$NSMOKE" --smoke; then
      python3 "$HERE/collect_newworld.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$NEWF" | tail -1
      python3 "$HERE/collect_newworld.py" --check "$NEWF"
    else
      summary+=("$TAG: new-world smoke test FAILED (see output above)")
    fi
  fi
  if [ "$CLEANWORLD" = 1 ] && [ "$(lines "$CLEANF")" -lt "$CLEANWORLD_N" ]; then
    echo "-- $TAG: clean-world stage ($CLEANWORLD_N calls)"
    CSMOKE="$HERE/runs/logs/smoke-cleanworld-$TAG.jsonl"; rm -f "$CSMOKE"
    python3 "$HERE/collect_cleanworld.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$CSMOKE" > /dev/null
    if python3 "$HERE/collect_cleanworld.py" --check "$CSMOKE" --smoke; then
      python3 "$HERE/collect_cleanworld.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$CLEANF" | tail -1
      python3 "$HERE/collect_cleanworld.py" --check "$CLEANF"
    else
      summary+=("$TAG: clean-world smoke test FAILED (see output above)")
    fi
  fi
  if [ "$CLEANCITE" = 1 ] && [ "$(lines "$CITEF")" -lt "$CLEANCITE_N" ]; then
    echo "-- $TAG: evidence-citation exploration stage ($CLEANCITE_N calls)"
    XSMOKE="$HERE/runs/logs/smoke-cleancite-$TAG.jsonl"; rm -f "$XSMOKE"
    python3 "$HERE/collect_cleancite.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$XSMOKE" > /dev/null
    if python3 "$HERE/collect_cleancite.py" --check "$XSMOKE" --smoke; then
      python3 "$HERE/collect_cleancite.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$CITEF" | tail -1
      python3 "$HERE/collect_cleancite.py" --check "$CITEF"
    else
      summary+=("$TAG: cleancite smoke test FAILED (see output above)")
    fi
  fi
  if [ "$SPEC" = 1 ] && [ "$(lines "$SPECF")" -lt "$SPEC_N" ]; then
    echo "-- $TAG: indirect-specification exploration stage ($SPEC_N calls)"
    PSMOKE="$HERE/runs/logs/smoke-spec-$TAG.jsonl"; rm -f "$PSMOKE"
    python3 "$HERE/collect_spec.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
      --max-tokens "$MAXTOK" --samples 0 --out "$PSMOKE" > /dev/null
    if python3 "$HERE/collect_spec.py" --check "$PSMOKE" --smoke; then
      python3 "$HERE/collect_spec.py" --base-url "$BASE" --model "$MODEL" --tag "$TAG" --extra-body "$EXTRA" \
        --max-tokens "$MAXTOK" --workers "$WORKERS" --out "$SPECF" | tail -1
      python3 "$HERE/collect_spec.py" --check "$SPECF"
    else
      summary+=("$TAG: spec smoke test FAILED (see output above)")
    fi
  fi
  stop_server
  if [ "$SPEC" = 1 ]; then
    summary+=("$TAG: spec $(lines "$SPECF")/$SPEC_N")
  fi
  if [ "$CLEANCITE" = 1 ]; then
    summary+=("$TAG: cleancite $(lines "$CITEF")/$CLEANCITE_N")
  fi
  if [ "$CLEANWORLD" = 1 ]; then
    summary+=("$TAG: cleanworld $(lines "$CLEANF")/$CLEANWORLD_N")
  fi
  if [ "$NEWWORLD" = 1 ]; then
    summary+=("$TAG: newworld $(lines "$NEWF")/$NEWWORLD_N")
  fi
  if [ "$SHAPE2" = 1 ]; then
    summary+=("$TAG: shape2 $(lines "$SHAPE2F")/$SHAPE2_N")
  fi
  if [ "$SHAPE" = 1 ]; then
    summary+=("$TAG: shape $(lines "$SHAPEF")/$SHAPE_N")
  fi
  if [ "$ICE" = 1 ]; then
    summary+=("$TAG: main $(lines "$MAIN")/$MAIN_N, strategy $(lines "$STRAT")/$STRAT_N, ice $(lines "$ICEF")/$ICE_N")
  else
    summary+=("$TAG: main $(lines "$MAIN")/$MAIN_N, strategy $(lines "$STRAT")/$STRAT_N")
  fi
done 3< "$HERE/$MODELS"

echo; echo "===== summary ====="; printf '%s\n' "${summary[@]}"
echo "Next: git add analysis/llm_statements/runs && git commit -m 'LLM runs' && git push"
