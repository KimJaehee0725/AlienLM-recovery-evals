#!/bin/bash

# N-gram Attack Evaluation - Run All Experiments
#
# Settings:
# - reference corpus: tulu3, acereason, slimorca
# - train dataset size: 50,000
# - test dataset size: 50,000
# - top_k_tokens: 10,000
# - n: 2, 3, 4
#
# Required: set the ALIEN_TOKENIZER_PATH environment variable to the local
# path of the AlienLM tokenizer checkpoint before running.

# set -e is left commented out so cleanup always runs.
# set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_evaluation.py"
OUTPUT_DIR="${SCRIPT_DIR}/attack_results"
LOG_DIR="${SCRIPT_DIR}/logs"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_processes.sh"

# Create the log directory
mkdir -p "${LOG_DIR}"

# Clean up previous processes before running
echo "Checking for hanging processes..."
if [ -f "${CLEANUP_SCRIPT}" ]; then
    bash "${CLEANUP_SCRIPT}" 2>/dev/null || true
fi

# Check for previous processes
PIDS=$(ps aux | grep "run_evaluation.py" | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "Warning: Found existing run_evaluation.py processes: $PIDS"
    # Non-interactive mode (controlled via an environment variable)
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
REFERENCE_CORPORA=("tulu3" "acereason" "slimorca")
N_VALUES=(2 3 4)
TRAIN_SIZE=10000
TEST_SIZE=1000
TOP_K_TOKENS=1000
REFERENCE_SIZE=10000  # Reference corpus size (leave empty to use the full corpus)

# Run the experiments
echo "=========================================="
echo "N-GRAM ATTACK EVALUATION - ALL EXPERIMENTS"
echo "=========================================="
echo "Reference Corpora: ${REFERENCE_CORPORA[@]}"
echo "N values: ${N_VALUES[@]}"
echo "Train size: ${TRAIN_SIZE}"
echo "Test size: ${TEST_SIZE}"
echo "Top K tokens: ${TOP_K_TOKENS}"
echo "Reference size: ${REFERENCE_SIZE}"
echo "=========================================="
echo ""

total_experiments=$((${#REFERENCE_CORPORA[@]} * ${#N_VALUES[@]}))
current_experiment=0

for ref_corpus in "${REFERENCE_CORPORA[@]}"; do
    for n in "${N_VALUES[@]}"; do
        current_experiment=$((current_experiment + 1))

        ref_size_str=""
        if [ -n "${REFERENCE_SIZE}" ]; then
            ref_size_str="_ref${REFERENCE_SIZE}"
        fi
        experiment_name="${ref_corpus}_n${n}_train${TRAIN_SIZE}_test${TEST_SIZE}_topk${TOP_K_TOKENS}${ref_size_str}"
        log_file="${LOG_DIR}/${experiment_name}.log"

        echo "[${current_experiment}/${total_experiments}] Running: ${experiment_name}"
        echo "  Log file: ${log_file}"
        echo "  Start time: $(date)"

        # Timeout setting (default 24 hours, adjust as needed)
        TIMEOUT_HOURS=${TIMEOUT_HOURS:-24}
        TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

        # Run the experiment (with a timeout, in unbuffered mode)
        timeout "${TIMEOUT_SECONDS}" python -u "${PYTHON_SCRIPT}" \
            --reference_corpus "${ref_corpus}" \
            --n "${n}" \
            --train_size "${TRAIN_SIZE}" \
            --test_size "${TEST_SIZE}" \
            --top_k_tokens "${TOP_K_TOKENS}" \
            --reference_size "${REFERENCE_SIZE}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_proc 64 \
            --batch_size 1000 \
            2>&1 | tee "${log_file}"

        exit_code=${PIPESTATUS[0]}

        echo "  End time: $(date)"

        if [ ${exit_code} -eq 0 ]; then
            echo "  ✓ Completed successfully"
        elif [ ${exit_code} -eq 124 ]; then
            echo "  ✗ Timeout (exceeded ${TIMEOUT_HOURS} hours)"
            echo "  Check log file: ${log_file}"
            echo "  Cleaning up hanging processes..."
            bash "${CLEANUP_SCRIPT}" 2>/dev/null || true
        else
            echo "  ✗ Failed with exit code ${exit_code}"
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
