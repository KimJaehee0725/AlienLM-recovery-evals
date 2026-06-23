#!/bin/bash

set -u

ROOT_DIR="/workspace/codes/AlienLMv2"
REBUTTAL_DIR="${ROOT_DIR}/icml2026-rebuttal/collaborative-attack"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
ATTACK_SCRIPT="${ROOT_DIR}/iclr_review/attack_scenario/n-gram_w_known-plaintext/run_evaluation.py"

MODEL_PATH="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ALIEN_TOKENIZER_PATH="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-9306"
DATASET_CACHE="/workspace/data2/jaehee/AlienLM/HF_DATASET"
HF_HOME="/workspace/CACHE/MODELS"

OUTPUT_ROOT="${REBUTTAL_DIR}/results/minconf_sweep_k10000_n3"
LOG_DIR="${REBUTTAL_DIR}/logs"

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

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
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
        --cache_dir "${DATASET_CACHE}" \
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
