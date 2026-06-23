#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=(
  "llama_original"
  "llama_alien"
  "qwen_original"
  "qwen_alien"
  "gemma_original"
  "gemma_alien"
)

for model_key in "${MODELS[@]}"; do
  echo "=== Running ${model_key} on GPU ${GPU_ID} ==="
  "${SCRIPT_DIR}/run_longbench_core.sh" "${model_key}" "${GPU_ID}"
done
