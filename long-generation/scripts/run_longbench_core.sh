#!/usr/bin/env bash

set -euo pipefail

MODEL_KEY="${1:-}"
GPU_ID="${2:-4}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: $0 <model_key> [gpu_id]" >&2
  echo "model_key: llama_original | llama_alien | qwen_original | qwen_alien | gemma_original | gemma_alien" >&2
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

TASKS="longbench_gov_report_e,longbench_qasper_e"
MODEL_PATH=""
TOKENIZER_PATH=""
OUTPUT_PATH=""
RUN_NAME=""
MAX_MODEL_LEN=""

case "${MODEL_KEY}" in
  llama_original)
    MODEL_PATH="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/llama_original"
    RUN_NAME="llama_original_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  llama_alien)
    MODEL_PATH="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/llama_alien"
    RUN_NAME="llama_alien_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  qwen_original)
    MODEL_PATH="/workspace/CACHE/MODELS/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/qwen_original"
    RUN_NAME="qwen_original_longbench_core"
    MAX_MODEL_LEN="32768"
    ;;
  qwen_alien)
    MODEL_PATH="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/qwen_alien"
    RUN_NAME="qwen_alien_longbench_core"
    MAX_MODEL_LEN="32768"
    ;;
  gemma_original)
    MODEL_PATH="/workspace/CACHE/MODELS/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/gemma_original"
    RUN_NAME="gemma_original_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  gemma_alien)
    MODEL_PATH="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen"
    TOKENIZER_PATH="${MODEL_PATH}"
    OUTPUT_PATH="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/gemma_alien"
    RUN_NAME="gemma_alien_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  *)
    echo "Unknown model_key: ${MODEL_KEY}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_PATH}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

{
  echo "Model key: ${MODEL_KEY}"
  echo "Model: ${MODEL_PATH}"
  echo "Tokenizer: ${TOKENIZER_PATH}"
  echo "Tasks: ${TASKS}"
  echo "Output: ${OUTPUT_PATH}"
  echo "GPU: ${GPU_ID}"
  echo "Max model len: ${MAX_MODEL_LEN}"
} | tee "${LOG_FILE}"

"${VENV_PY}" -m lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=${MAX_MODEL_LEN}" \
  --tasks "${TASKS}" \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}/longbench_core/0-shot" \
  --log_samples \
  2>&1 | tee -a "${LOG_FILE}"
