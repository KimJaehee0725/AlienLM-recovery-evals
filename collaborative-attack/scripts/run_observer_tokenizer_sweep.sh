#!/bin/bash

set -u

ROOT_DIR="/workspace/codes/AlienLMv2"
REBUTTAL_DIR="${ROOT_DIR}/icml2026-rebuttal/collaborative-attack"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SWEEP_SCRIPT="${REBUTTAL_DIR}/scripts/run_observer_tokenizer_sweep.py"

VICTIM_TOKENIZER_PATH="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ALIEN_TOKENIZER_PATH="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-9306"
OBSERVER_TOKENIZERS="llama3=/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,qwen25=/workspace/CACHE/MODELS/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28,gemma2=/workspace/CACHE/MODELS/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819,mistral=/workspace/CACHE/MODELS/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71,phi3=/workspace/CACHE/MODELS/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e"
DATASET_CACHE="/workspace/data2/jaehee/AlienLM/HF_DATASET"
HF_HOME="/workspace/CACHE/MODELS"

OUTPUT_ROOT="${REBUTTAL_DIR}/results/observer_tokenizer_sweep_n3_k10000_c0p01"
LOG_DIR="${REBUTTAL_DIR}/logs"
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

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
export TOKENIZERS_PARALLELISM=false

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
    --cache_dir "${DATASET_CACHE}" \
    --min_confidence "${MIN_CONFIDENCE}" \
    2>&1 | tee "${LOG_FILE}"

status=${PIPESTATUS[0]}
echo "end_time=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "exit_code=${status}"

exit "${status}"
