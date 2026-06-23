#!/usr/bin/env bash

set -euo pipefail

MODEL_KEY="${1:-}"
GPU_ID="${2:-4}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: $0 <model_key> [gpu_id]"
  echo "model_key: llama_original"
  exit 1
fi

ROOT="/workspace/codes/AlienLMv2"
VENV_PY="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/icml2026-rebuttal/long-generation/logs"
mkdir -p "${LOG_DIR}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/data2/jaehee/AlienLM/HF_DATASET}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/CACHE/MODELS}"
export HF_HOME="${HF_HOME:-/workspace/CACHE/MODELS}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/lm-evaluation-harness:${PYTHONPATH:-}"

MODEL_PATH=""
TOKENIZER_PATH=""
OUTPUT_PATH=""
RUN_NAME=""

case "${MODEL_KEY}" in
  llama_original)
    MODEL_PATH="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/meta-llama/Meta-Llama-3-8B-Instruct"
    RUN_NAME="llama_original_truthfulqa_gen"
    ;;
  *)
    echo "Unknown model_key: ${MODEL_KEY}"
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_PATH}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "Model: ${MODEL_PATH}" | tee "${LOG_FILE}"
echo "Tokenizer: ${TOKENIZER_PATH}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_PATH}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}" | tee -a "${LOG_FILE}"

"${VENV_PY}" -m lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=8192" \
  --tasks truthfulqa_gen \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}/truthfulqa_gen/0-shot" \
  --log_samples \
  2>&1 | tee -a "${LOG_FILE}"
