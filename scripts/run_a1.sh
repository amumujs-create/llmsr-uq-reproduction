#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPRO_ROOT"
REPRO_PYTHON="${REPRO_PYTHON:-python}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
BACKEND_LABEL="${BACKEND_LABEL:-unverified-backend}"
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
common=(--model "$LLM_MODEL" --base-url "$LLM_BASE_URL" --backend-label "$BACKEND_LABEL")
case "$mode" in
  smoke)
    exec "$REPRO_PYTHON" paired_runner.py "${common[@]}" --iterations 2 --only lsr_synth_phys_osc_po10 "$@"
    ;;
  dev3)
    exec "$REPRO_PYTHON" paired_runner.py "${common[@]}" --iterations 125 \
      --only lsr_synth_phys_osc_po10,lsr_synth_phys_osc_po12,lsr_synth_phys_osc_po25 "$@"
    ;;
  *)
    echo 'Usage: bash scripts/run_a1.sh {smoke|dev3} [paired_runner options]'
    echo 'smoke: one dev problem, 2 rounds; dev3: three preselected dev problems, 125 rounds each.'
    echo 'Set LLM_MODEL, LLM_BASE_URL, BACKEND_LABEL as appropriate for your server.'
    [[ "$mode" == help || "$mode" == --help || "$mode" == -h ]] && exit 0
    exit 2
    ;;
esac
