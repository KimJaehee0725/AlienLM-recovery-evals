#!/bin/bash

set -u

# Experiment directory (parent of scripts/), and the local sweep driver.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SCRIPT="${EXP_DIR}/scripts/run_observer_tokenizer_sweep.py"

# External resources (override via environment variables).
VICTIM_TOKENIZER_PATH="${VICTIM_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer checkpoint}"
# Comma-separated name=path entries; paths may be hub ids or local checkpoints.
OBSERVER_TOKENIZERS="${OBSERVER_TOKENIZERS:-llama3=meta-llama/Meta-Llama-3-8B-Instruct,qwen25=Qwen/Qwen2.5-7B-Instruct,gemma2=google/gemma-2-9b-it,mistral=mistralai/Mistral-7B-Instruct-v0.3,phi3=microsoft/Phi-3-mini-4k-instruct}"
DATASET_CACHE="${DATASET_CACHE:-}"
HF_HOME="${HF_HOME:-}"

OUTPUT_ROOT="${EXP_DIR}/results/observer_tokenizer_sweep_n3_k10000_c0p01"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/observer_tokenizer_sweep_n3_k10000_c0p01.log"

REFERENCE_CORPUS="tulu3"
N_GRAM="3"
TRAIN_SIZE="10000"
TEST_SIZE="1000"
K_KNOWN_PAIRS="10000"
TOP_K_TOKENS="1000"
REFERENCE_SIZE="10000"
MIN_CONFIDENCE="0.01"
NUM_PROC="${NUM_PROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# Optional HuggingFace cache configuration (only exported when provided).
if [ -n "${HF_HOME}" ]; then
    export HF_HOME
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
fi
if [ -n "${DATASET_CACHE}" ]; then
    export HF_DATASETS_CACHE="${DATASET_CACHE}"
fi
export TOKENIZERS_PARALLELISM=false

CACHE_ARGS=()
if [ -n "${DATASET_CACHE}" ]; then
    CACHE_ARGS+=(--cache_dir "${DATASET_CACHE}")
fi

echo "============================================================"
echo "observer tokenizer sweep"
echo "log_file=${LOG_FILE}"
echo "output_dir=${OUTPUT_ROOT}"
echo "start_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

"${PYTHON_BIN}" -u "${SWEEP_SCRIPT}" \
    --victim_tokenizer_path "${VICTIM_TOKENIZER_PATH}" \
    --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}" \
    --observer_tokenizers "${OBSERVER_TOKENIZERS}" \
    --reference_corpus "${REFERENCE_CORPUS}" \
    --n "${N_GRAM}" \
    --train_size "${TRAIN_SIZE}" \
    --test_size "${TEST_SIZE}" \
    --k_known_pairs "${K_KNOWN_PAIRS}" \
    --top_k_tokens "${TOP_K_TOKENS}" \
    --reference_size "${REFERENCE_SIZE}" \
    --output_dir "${OUTPUT_ROOT}" \
    --num_proc "${NUM_PROC}" \
    --batch_size "${BATCH_SIZE}" \
    --encode_batch_size "${ENCODE_BATCH_SIZE}" \
    "${CACHE_ARGS[@]}" \
    --min_confidence "${MIN_CONFIDENCE}" \
    2>&1 | tee "${LOG_FILE}"

status=${PIPESTATUS[0]}
echo "end_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "exit_code=${status}"

exit "${status}"
