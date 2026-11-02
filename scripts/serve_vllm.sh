#!/usr/bin/env bash
# Run from a separate vLLM environment, not the NumPy 1.x analysis environment.
set -euo pipefail
REPRO_MODEL="${REPRO_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
REPRO_PORT="${REPRO_PORT:-8000}"
REPRO_SERVER_PYTHON="${REPRO_SERVER_PYTHON:-python}"
exec "$REPRO_SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$REPRO_MODEL" \
  --served-model-name "$REPRO_MODEL" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port "$REPRO_PORT" "$@"
