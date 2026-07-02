#!/bin/bash

set -u

# Experiment directory (parent of scripts/), and the local sweep driver.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SCRIPT="${EXP_DIR}/scripts/run_mixed_seed_pool_sweep.py"

# External resources (override via environment variables).
MODEL_PATH="${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
# Comma-separated list of alien tokenizer checkpoints, one per seed.
# Provide three distinct seeds for the mixed-seed comparison.
ALIEN_TOKENIZER_PATHS="${ALIEN_TOKENIZER_PATHS:?set ALIEN_TOKENIZER_PATHS to a comma-separated list of per-seed alien tokenizer checkpoints}"
DATASET_CACHE="${DATASET_CACHE:-}"
HF_HOME="${HF_HOME:-}"

OUTPUT_ROOT="${EXP_DIR}/results/seedpool_sweep_n3_c0p95_occ3"
LOG_DIR="${EXP_DIR}/logs"

REFERENCE_CORPUS="tulu3"
N_GRAM="3"
TRAIN_SIZE="10000"
TEST_SIZE="1000"
K_VALUES="1000,5000,10000"
TOP_K_TOKENS="1000"
REFERENCE_SIZE="10000"
MIN_CONFIDENCE="0.95"
MIN_OCCURRENCES="3"
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

for mode in shared mixed; do
    log_file="${LOG_DIR}/seedpool_${mode}_n${N_GRAM}_c0p95_occ3.log"
    output_dir="${OUTPUT_ROOT}/${mode}"

    echo "============================================================"
    echo "assignment_mode=${mode}"
    echo "log_file=${log_file}"
    echo "output_dir=${output_dir}"
    echo "start_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    "${PYTHON_BIN}" -u "${SWEEP_SCRIPT}" \
        --reference_corpus "${REFERENCE_CORPUS}" \
        --n "${N_GRAM}" \
        --train_size "${TRAIN_SIZE}" \
        --test_size "${TEST_SIZE}" \
        --k_values "${K_VALUES}" \
        --top_k_tokens "${TOP_K_TOKENS}" \
        --reference_size "${REFERENCE_SIZE}" \
        --output_dir "${output_dir}" \
        --num_proc "${NUM_PROC}" \
        --batch_size "${BATCH_SIZE}" \
        --encode_batch_size "${ENCODE_BATCH_SIZE}" \
        --org_tokenizer_path "${MODEL_PATH}" \
        --alien_tokenizer_paths "${ALIEN_TOKENIZER_PATHS}" \
        --assignment_mode "${mode}" \
        --assignment_seed "42" \
        "${CACHE_ARGS[@]}" \
        --min_confidence "${MIN_CONFIDENCE}" \
        --min_occurrences "${MIN_OCCURRENCES}" \
        2>&1 | tee "${log_file}"

    status=${PIPESTATUS[0]}
    echo "end_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "exit_code=${status}"

    if [ "${status}" -ne 0 ]; then
        echo "Run failed for assignment_mode=${mode}" >&2
        exit "${status}"
    fi
done

echo "Mixed-seed pool sweep completed successfully."
