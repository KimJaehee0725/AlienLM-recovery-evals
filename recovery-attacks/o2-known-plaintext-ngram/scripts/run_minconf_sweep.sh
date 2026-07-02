#!/bin/bash

set -u

# Experiment directory (parent of scripts/), and the locally copied attack code.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ATTACK_SCRIPT="${EXP_DIR}/run_evaluation.py"

# External resources (override via environment variables).
MODEL_PATH="${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer checkpoint}"
DATASET_CACHE="${DATASET_CACHE:-}"
HF_HOME="${HF_HOME:-}"

OUTPUT_ROOT="${EXP_DIR}/results/minconf_sweep_k10000_n3"
LOG_DIR="${EXP_DIR}/logs"

MIN_CONF_VALUES=("0.5" "0.05" "0.01")
REFERENCE_CORPUS="tulu3"
N_GRAM="3"
TRAIN_SIZE="10000"
TEST_SIZE="1000"
TOP_K_TOKENS="1000"
REFERENCE_SIZE="10000"
K_KNOWN_PAIRS="10000"
NUM_PROC="${NUM_PROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-1000}"

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

echo "Running min_confidence sweep with k_known_pairs=${K_KNOWN_PAIRS}, n=${N_GRAM}"
echo "Output root: ${OUTPUT_ROOT}"

for min_conf in "${MIN_CONF_VALUES[@]}"; do
    safe_conf="${min_conf//./p}"
    log_file="${LOG_DIR}/minconf_k${K_KNOWN_PAIRS}_n${N_GRAM}_c${safe_conf}.log"
    output_dir="${OUTPUT_ROOT}/c${safe_conf}"

    echo "============================================================"
    echo "min_confidence=${min_conf}"
    echo "log_file=${log_file}"
    echo "output_dir=${output_dir}"
    echo "start_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    CACHE_ARGS=()
    if [ -n "${DATASET_CACHE}" ]; then
        CACHE_ARGS+=(--cache_dir "${DATASET_CACHE}")
    fi

    "${PYTHON_BIN}" -u "${ATTACK_SCRIPT}" \
        --reference_corpus "${REFERENCE_CORPUS}" \
        --n "${N_GRAM}" \
        --train_size "${TRAIN_SIZE}" \
        --test_size "${TEST_SIZE}" \
        --top_k_tokens "${TOP_K_TOKENS}" \
        --reference_size "${REFERENCE_SIZE}" \
        --k_known_pairs "${K_KNOWN_PAIRS}" \
        --min_confidence "${min_conf}" \
        --output_dir "${output_dir}" \
        --num_proc "${NUM_PROC}" \
        --batch_size "${BATCH_SIZE}" \
        --org_tokenizer_path "${MODEL_PATH}" \
        --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}" \
        "${CACHE_ARGS[@]}" \
        2>&1 | tee "${log_file}"

    status=${PIPESTATUS[0]}
    echo "end_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "exit_code=${status}"

    if [ "${status}" -ne 0 ]; then
        echo "Run failed for min_confidence=${min_conf}" >&2
        exit "${status}"
    fi
done

echo "Sweep completed successfully."
