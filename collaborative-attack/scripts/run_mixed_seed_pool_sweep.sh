#!/bin/bash

set -u

ROOT_DIR="/workspace/codes/AlienLMv2"
REBUTTAL_DIR="${ROOT_DIR}/icml2026-rebuttal/collaborative-attack"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SWEEP_SCRIPT="${REBUTTAL_DIR}/scripts/run_mixed_seed_pool_sweep.py"

MODEL_PATH="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ALIEN_TOKENIZER_PATHS="/workspace/codes/AlienLMv2/alien_tokenizer/tokenizers/qwenv2_bucket_random_5_seed-42,/workspace/codes/AlienLMv2/alien_tokenizer/tokenizers/qwenv2_bucket_random_5_seed-43,/workspace/codes/AlienLMv2/alien_tokenizer/tokenizers/qwenv2_bucket_random_5_seed-44"
DATASET_CACHE="/workspace/data2/jaehee/AlienLM/HF_DATASET"
HF_HOME="/workspace/CACHE/MODELS"

OUTPUT_ROOT="${REBUTTAL_DIR}/results/seedpool_sweep_n3_c0p95_occ3"
LOG_DIR="${REBUTTAL_DIR}/logs"

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

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
export TOKENIZERS_PARALLELISM=false

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
        --cache_dir "${DATASET_CACHE}" \
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
