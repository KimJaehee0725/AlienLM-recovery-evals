#!/bin/bash

###############################################################################
# Progressive LLM Attack Experiment (Parallel Mode)
# Runs experiments sequentially over several few-shot sizes (0, 1, 3, 5, 10, 20).
# Reproduces Table 14 (parallel leakage, Appendix E.3.1).
###############################################################################

set -e  # Stop on error

# ============================================================================
# Configuration
# ============================================================================

# Default data path
DEFAULT_DATA_FILE="./data/test_data.jsonl"
DATA_FILE="$DEFAULT_DATA_FILE"

# Model settings
ATTACK_MODEL="gpt-4.1-2025-04-14"  # Empty string uses the config default
JUDGE_MODEL=""   # Empty string uses the config default

# Experiment settings
FEW_SHOT_SIZES="0 1 5 20"  # Few-shot sizes to test
N_TEST_SAMPLES=100          # Number of test samples
MAX_CONCURRENT_REQUESTS=32    # Limit on the number of concurrent requests
OUTPUT_DIR="./llm_attack_results/progressive_$(date +%Y%m%d_%H%M%S)"

# OpenAI settings (must already be exported in the environment)
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"

# Whether to use the LLM judge
USE_LLM_JUDGE=true  # true or false

# Non-parallel attack mode
NON_PARALLEL=false  # If true, use the non-parallel template

# ============================================================================
# Command-line argument parsing
# ============================================================================

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data_file FILE     Path to JSONL data file (default: $DEFAULT_DATA_FILE)"
    echo "  -f, --few_shots LIST     Few-shot sizes (default: \"0 1 5 20\")"
    echo "  -s, --samples N          Number of test samples (default: 100)"
    echo "  -c, --concurrent N        Max concurrent requests (default: 32)"
    echo "  -a, --attack_model MODEL Attack model (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini')"
    echo "  -j, --judge_model MODEL  Judge model (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini')"
    echo "  -o, --output_dir DIR      Output directory"
    echo "  --no_llm_judge            Disable LLM judge"
    echo "  --non_parallel            Use non-parallel attack template (indicates leaked encrypted data)"
    echo "  --list_data               List available data files"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Available data files:"
    if [ -d "./data" ]; then
        for file in ./data/test_data*.jsonl; do
            if [ -f "$file" ]; then
                echo "  - $(basename "$file")"
            fi
        done
    fi
    echo ""
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        -f|--few_shots)
            FEW_SHOT_SIZES="$2"
            shift 2
            ;;
        -s|--samples)
            N_TEST_SAMPLES="$2"
            shift 2
            ;;
        -c|--concurrent)
            MAX_CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        -a|--attack_model)
            ATTACK_MODEL="$2"
            shift 2
            ;;
        -j|--judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no_llm_judge)
            USE_LLM_JUDGE=false
            shift
            ;;
        --non_parallel)
            NON_PARALLEL=true
            shift
            ;;
        --list_data)
            echo "Available data files in ./data/:"
            if [ -d "./data" ]; then
                for file in ./data/test_data*.jsonl; do
                    if [ -f "$file" ]; then
                        lines=$(wc -l < "$file" 2>/dev/null || echo "?")
                        echo "  - $(basename "$file") ($lines lines)"
                    fi
                done
            else
                echo "  ./data/ directory not found"
            fi
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# Environment checks
# ============================================================================

echo "========================================================================"
echo "Progressive LLM Attack Experiment"
if [ "$NON_PARALLEL" = true ]; then
    echo "Mode: NON-PARALLEL ATTACK (using non-parallel template)"
fi
echo "========================================================================"
echo ""

# Check the API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Check the data file
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Please run prepare_data.py first to generate the data file."
    exit 1
fi

# Create the output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
if [ "$NON_PARALLEL" = true ]; then
    echo "  Mode:                   Non-parallel attack"
fi
echo "  Data file:              $DATA_FILE"
echo "  Few-shot sizes:         $FEW_SHOT_SIZES"
echo "  Test samples:           $N_TEST_SAMPLES"
echo "  Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
if [ -n "$ATTACK_MODEL" ]; then
    echo "  Attack model:           $ATTACK_MODEL"
else
    echo "  Attack model:           (default from config)"
fi
if [ -n "$JUDGE_MODEL" ]; then
    echo "  Judge model:            $JUDGE_MODEL"
else
    echo "  Judge model:            (default from config)"
fi
echo "  Output dir:             $OUTPUT_DIR"
echo "  LLM Judge:              $USE_LLM_JUDGE"
echo ""

# ============================================================================
# Run the experiment
# ============================================================================

# LLM judge option
LLM_JUDGE_OPT=""
if [ "$USE_LLM_JUDGE" = false ]; then
    LLM_JUDGE_OPT="--no_llm_judge"
fi

# Non-parallel option
NON_PARALLEL_OPT=""
if [ "$NON_PARALLEL" = true ]; then
    NON_PARALLEL_OPT="--non_parallel"
fi

# Record the start time
START_TIME=$(date +%s)

echo "========================================================================"
echo "Starting Progressive Experiment..."
echo "========================================================================"
echo ""

# Run the Python script
ATTACK_MODEL_OPT=""
JUDGE_MODEL_OPT=""

if [ -n "$ATTACK_MODEL" ]; then
    ATTACK_MODEL_OPT="--attack_model $ATTACK_MODEL"
fi

if [ -n "$JUDGE_MODEL" ]; then
    JUDGE_MODEL_OPT="--judge_model $JUDGE_MODEL"
fi

python run_llm_attack.py \
    --data_file "$DATA_FILE" \
    --mode progressive \
    --few_shot_sizes $FEW_SHOT_SIZES \
    --n_test_samples $N_TEST_SAMPLES \
    --max_concurrent_requests $MAX_CONCURRENT_REQUESTS \
    --output_dir "$OUTPUT_DIR" \
    $ATTACK_MODEL_OPT \
    $JUDGE_MODEL_OPT \
    $NON_PARALLEL_OPT \
    $LLM_JUDGE_OPT

# Compute the elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "Experiment Completed!"
echo "========================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# ============================================================================
# Print a result summary
# ============================================================================

# Find the progressive_experiment.json file
RESULTS_FILE="$OUTPUT_DIR/progressive_experiment.json"

if [ -f "$RESULTS_FILE" ]; then
    echo "========================================================================"
    echo "Quick Summary"
    echo "========================================================================"

    # Print a simple summary with Python
    python -c "
import json
import sys

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

results = data.get('results_by_shots', {})

print('\nResults by Few-shot Size:')
print('-' * 80)
print('N-shots'.ljust(10) + 'BLEU'.ljust(10) + 'ROUGE-1'.ljust(10) + 'ROUGE-L'.ljust(10) + 'LLM Judge'.ljust(10))
print('-' * 80)

for n_shots in sorted([int(k) for k in results.keys()]):
    metrics = results[str(n_shots)]
    bleu = metrics.get('bleu_mean', 0)
    rouge1 = metrics.get('rouge1_f_mean', 0)
    rougeL = metrics.get('rougeL_f_mean', 0)
    llm = metrics.get('llm_overall_mean', 0)

    print(str(n_shots).ljust(10) + str(round(bleu, 2)).ljust(10) + str(round(rouge1, 2)).ljust(10) + str(round(rougeL, 2)).ljust(10) + str(round(llm, 2)).ljust(10))

print('-' * 80)
    " 2>/dev/null || echo "Could not generate summary"
fi

echo ""
echo "Done!"
