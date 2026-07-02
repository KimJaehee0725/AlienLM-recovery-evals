#!/usr/bin/env bash

set -euo pipefail

MODEL_KEY="${1:-}"
GPU_ID="${2:-0}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: $0 <model_key> [gpu_id]" >&2
  echo "model_key: llama_original | llama_alien | qwen_original | qwen_alien | gemma_original | gemma_alien" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ALIENLM_CODE_ROOT="${ALIENLM_CODE_ROOT:-}"
VENV_PY="${PYTHON_BIN:-${ALIENLM_CODE_ROOT:+$ALIENLM_CODE_ROOT/.venv/bin/python}}"
VENV_PY="${VENV_PY:-python}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
OUTPUT_ROOT="${LONG_GENERATION_OUTPUT_ROOT:-$EXP_DIR/outputs}"
mkdir -p "${LOG_DIR}"

DEFAULT_CACHE_ROOT="${HF_HOME:-$EXP_DIR/.cache}"
export HF_HOME="$DEFAULT_CACHE_ROOT"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DEFAULT_CACHE_ROOT/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$DEFAULT_CACHE_ROOT/hf_models}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
if [[ -n "$ALIENLM_CODE_ROOT" && -d "$ALIENLM_CODE_ROOT/lm-evaluation-harness" ]]; then
  export PYTHONPATH="$ALIENLM_CODE_ROOT/lm-evaluation-harness:${PYTHONPATH:-}"
fi

TASKS="longbench_gov_report_e,longbench_qasper_e"
MODEL_PATH=""
TOKENIZER_PATH=""
OUTPUT_PATH=""
RUN_NAME=""
MAX_MODEL_LEN=""

case "${MODEL_KEY}" in
  llama_original)
    MODEL_PATH="${LLAMA_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
    TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/llama_original"
    RUN_NAME="llama_original_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  llama_alien)
    MODEL_PATH="${LLAMA_ALIEN_MODEL_PATH:-dsba-lab/llama3-8b-instruct-alienlm-full}"
    TOKENIZER_PATH="${LLAMA_ALIEN_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/llama_alien"
    RUN_NAME="llama_alien_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  qwen_original)
    MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
    TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/qwen_original"
    RUN_NAME="qwen_original_longbench_core"
    MAX_MODEL_LEN="32768"
    ;;
  qwen_alien)
    MODEL_PATH="${QWEN_ALIEN_MODEL_PATH:?set QWEN_ALIEN_MODEL_PATH}"
    TOKENIZER_PATH="${QWEN_ALIEN_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/qwen_alien"
    RUN_NAME="qwen_alien_longbench_core"
    MAX_MODEL_LEN="32768"
    ;;
  gemma_original)
    MODEL_PATH="${GEMMA_MODEL_PATH:-google/gemma-2-9b-it}"
    TOKENIZER_PATH="${GEMMA_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/gemma_original"
    RUN_NAME="gemma_original_longbench_core"
    MAX_MODEL_LEN="8192"
    ;;
  gemma_alien)
    MODEL_PATH="${GEMMA_ALIEN_MODEL_PATH:?set GEMMA_ALIEN_MODEL_PATH}"
    TOKENIZER_PATH="${GEMMA_ALIEN_TOKENIZER_PATH:-${MODEL_PATH}}"
    OUTPUT_PATH="${OUTPUT_ROOT}/gemma_alien"
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
