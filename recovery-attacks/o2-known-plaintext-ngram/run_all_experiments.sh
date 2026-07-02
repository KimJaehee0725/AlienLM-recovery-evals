#!/bin/bash

# N-gram Attack Evaluation - Run All Experiments
#
# Setup:
# - reference corpus: tulu3 (fixed)
# - train dataset size: 10,000
# - test dataset size: 1,000
# - top_k_tokens: 1,000
# - n: 2, 3, 4
# - k_known_pairs: swept over several values

# `set -e` is intentionally left off so that cleanup runs properly.
# set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_evaluation.py"
OUTPUT_DIR="${SCRIPT_DIR}/attack_results"
LOG_DIR="${SCRIPT_DIR}/logs"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_processes.sh"

# Required external resources (override via environment variables).
# ORG_TOKENIZER_PATH: victim/original tokenizer (hub id or local path).
# ALIEN_TOKENIZER_PATH: alien tokenizer checkpoint (hub id or local path).
# CACHE_DIR: HuggingFace datasets cache directory (optional).
ORG_TOKENIZER_PATH="${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer checkpoint}"
CACHE_DIR="${CACHE_DIR:-}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Clean up any hanging processes before running
echo "Checking for hanging processes..."
if [ -f "${CLEANUP_SCRIPT}" ]; then
    bash "${CLEANUP_SCRIPT}" 2>/dev/null || true
fi

# Check for existing processes
PIDS=$(ps aux | grep "run_evaluation.py" | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "Warning: Found existing run_evaluation.py processes: $PIDS"
    # Non-interactive mode (controlled by environment variable)
    if [ "${NON_INTERACTIVE:-0}" = "1" ]; then
        echo "Non-interactive mode: Killing existing processes automatically..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 2
        echo "Cleaned up existing processes."
    else
        read -p "Kill existing processes and continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$PIDS" | xargs kill -9 2>/dev/null || true
            sleep 2
            echo "Cleaned up existing processes."
        else
            echo "Exiting. Please clean up processes manually."
            exit 1
        fi
    fi
fi

# Experiment parameters
REFERENCE_CORPUS="tulu3"  # Reference corpus fixed
N_VALUES=(2 3 4)
K_KNOWN_PAIRS_VALUES=(10 50 100 500 1000)  # Number of known ciphertext-plaintext pairs (0 disables)
TRAIN_SIZE=10000
TEST_SIZE=1000
TOP_K_TOKENS=1000
REFERENCE_SIZE=10000  # Reference corpus size (None for all)
MIN_CONFIDENCE=0.5  # Minimum confidence for token mappings extracted from known pairs

# Run experiments
echo "=========================================="
echo "N-GRAM ATTACK EVALUATION - ALL EXPERIMENTS"
echo "=========================================="
echo "Reference Corpus: ${REFERENCE_CORPUS} (fixed)"
echo "N values: ${N_VALUES[@]}"
echo "Known plaintext pairs (k): ${K_KNOWN_PAIRS_VALUES[@]}"
echo "Train size: ${TRAIN_SIZE}"
echo "Test size: ${TEST_SIZE}"
echo "Top K tokens: ${TOP_K_TOKENS}"
echo "Reference size: ${REFERENCE_SIZE}"
echo "Min confidence: ${MIN_CONFIDENCE}"
echo "=========================================="
echo ""

total_experiments=$((${#N_VALUES[@]} * ${#K_KNOWN_PAIRS_VALUES[@]}))
current_experiment=0

for n in "${N_VALUES[@]}"; do
    for k_known_pairs in "${K_KNOWN_PAIRS_VALUES[@]}"; do
        current_experiment=$((current_experiment + 1))

        ref_size_str=""
        if [ -n "${REFERENCE_SIZE}" ]; then
            ref_size_str="_ref${REFERENCE_SIZE}"
        fi
        k_str=""
        if [ "${k_known_pairs}" -gt 0 ]; then
            k_str="_k${k_known_pairs}"
        fi
        experiment_name="${REFERENCE_CORPUS}_n${n}_train${TRAIN_SIZE}_test${TEST_SIZE}_topk${TOP_K_TOKENS}${ref_size_str}${k_str}"
        log_file="${LOG_DIR}/${experiment_name}.log"

        echo "[${current_experiment}/${total_experiments}] Running: ${experiment_name}"
        echo "  N-gram size: ${n}"
        echo "  Known pairs (k): ${k_known_pairs}"
        echo "  Log file: ${log_file}"
        echo "  Start time: $(date)"

        # Timeout (default 24 hours, adjustable)
        TIMEOUT_HOURS=${TIMEOUT_HOURS:-24}
        TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

        # Run the experiment (with timeout, in unbuffered mode)
        CMD_ARGS=(
            --reference_corpus "${REFERENCE_CORPUS}"
            --n "${n}"
            --train_size "${TRAIN_SIZE}"
            --test_size "${TEST_SIZE}"
            --top_k_tokens "${TOP_K_TOKENS}"
            --reference_size "${REFERENCE_SIZE}"
            --output_dir "${OUTPUT_DIR}"
            --num_proc 64
            --batch_size 1000
            --org_tokenizer_path "${ORG_TOKENIZER_PATH}"
            --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}"
        )

        # Pass the cache directory if set
        if [ -n "${CACHE_DIR}" ]; then
            CMD_ARGS+=(--cache_dir "${CACHE_DIR}")
        fi

        # Add known-plaintext parameters when configured
        if [ "${k_known_pairs}" -gt 0 ]; then
            CMD_ARGS+=(
                --k_known_pairs "${k_known_pairs}"
                --min_confidence "${MIN_CONFIDENCE}"
            )
        fi

        timeout "${TIMEOUT_SECONDS}" python -u "${PYTHON_SCRIPT}" \
            "${CMD_ARGS[@]}" \
            2>&1 | tee "${log_file}"

        exit_code=${PIPESTATUS[0]}

        echo "  End time: $(date)"

        if [ ${exit_code} -eq 0 ]; then
            echo "  Completed successfully"
        elif [ ${exit_code} -eq 124 ]; then
            echo "  Timeout (exceeded ${TIMEOUT_HOURS} hours)"
            echo "  Check log file: ${log_file}"
            echo "  Cleaning up hanging processes..."
            bash "${CLEANUP_SCRIPT}" 2>/dev/null || true
        else
            echo "  Failed with exit code ${exit_code}"
            echo "  Check log file: ${log_file}"
            echo "  Cleaning up hanging processes..."
            bash "${CLEANUP_SCRIPT}" 2>/dev/null || true
        fi

        echo ""
    done
done

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total experiments: ${total_experiments}"
echo "Results directory: ${OUTPUT_DIR}"
echo "Logs directory: ${LOG_DIR}"
echo "=========================================="
