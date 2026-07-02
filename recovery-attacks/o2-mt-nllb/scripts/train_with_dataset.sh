#!/bin/bash
# Example training script that uses a real dataset
# Usage: bash scripts/train_with_dataset.sh --dataset_name DATASET_NAME

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Optional Hugging Face cache locations (override via environment variables if desired)
[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${HF_DATASETS_CACHE:-}" ] && export HF_DATASETS_CACHE
export WANDB_PROJECT="${WANDB_PROJECT:-AlienLM-Translator}"
# WANDB_RUN_NAME is auto-generated below or can be set via an environment variable
set -e

# Resolve the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate your Python environment here if needed (e.g. conda activate / source venv).

# Default settings
MODEL_NAME="facebook/nllb-200-3.3B"
ORIGINAL_TOKENIZER_PATH="${ORIGINAL_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/translator/$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS=4

# Training hyperparameters
NUM_EPOCHS=2
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-5
MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=1024

# Dataset settings (required)
DATASET_NAME="slimorca"
DATASET_CONFIG_NAME=""
TEXT_COLUMN="text"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset_config_name)
            DATASET_CONFIG_NAME="$2"
            shift 2
            ;;
        --text_column)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check the dataset name
if [ -z "$DATASET_NAME" ]; then
    echo "Error: --dataset_name is required"
    echo ""
    echo "Examples:"
    echo "  # Use both Magpie datasets (Magpie-Pro-300K-Filtered + Magpie-Reasoning-V1-150K)"
    echo "  bash scripts/train_with_dataset.sh --dataset_name magpie"
    echo ""
    echo "  # Use another Hugging Face dataset"
    echo "  bash scripts/train_with_dataset.sh --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1"
    exit 1
fi

# Notice when using the Magpie datasets
if [ "$DATASET_NAME" = "magpie" ] || [ "$DATASET_NAME" = "Magpie" ] || [ "$DATASET_NAME" = "MAGPIE" ]; then
    echo "=========================================="
    echo "Using Magpie datasets:"
    echo "  - Magpie-Pro-300K-Filtered"
    echo "  - Magpie-Reasoning-V1-150K"
    echo "Note: Instruction and response are treated as separate samples"
    echo "=========================================="
    echo ""
fi

# Auto-generate WANDB_RUN_NAME if it is not set via an environment variable
if [ -z "$WANDB_RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    WANDB_RUN_NAME="translator_${DATASET_NAME}_${TIMESTAMP}"
    export WANDB_RUN_NAME
    echo "WANDB_RUN_NAME not set, using auto-generated name: $WANDB_RUN_NAME"
else
    echo "Using WANDB_RUN_NAME: $WANDB_RUN_NAME"
fi

# Call the main training script
bash scripts/train.sh \
    --model_name "$MODEL_NAME" \
    --original_tokenizer_path "$ORIGINAL_TOKENIZER_PATH" \
    --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus "$NUM_GPUS" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_source_length "$MAX_SOURCE_LENGTH" \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --dataset_name "$DATASET_NAME" \
    ${DATASET_CONFIG_NAME:+--dataset_config_name "$DATASET_CONFIG_NAME"} \
    --text_column "$TEXT_COLUMN" \
    --bf16 \
    --report_to wandb
