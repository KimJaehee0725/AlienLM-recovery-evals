#!/bin/bash
# Training script for translator model with 4 GPUs

set -e

# Default arguments
MODEL_NAME="facebook/nllb-200-3.3B"
ORIGINAL_TOKENIZER_PATH="${ORIGINAL_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer}"
OUTPUT_DIR="./outputs/translator_model"
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-5
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=512

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --original_tokenizer_path)
            ORIGINAL_TOKENIZER_PATH="$2"
            shift 2
            ;;
        --alien_tokenizer_path)
            ALIEN_TOKENIZER_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
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
        --max_source_length)
            MAX_SOURCE_LENGTH="$2"
            shift 2
            ;;
        --max_target_length)
            MAX_TARGET_LENGTH="$2"
            shift 2
            ;;
        --bf16)
            BF16_FLAG="--bf16"
            shift
            ;;
        --fp16)
            FP16_FLAG="--fp16"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate your Python environment here if needed (e.g. conda activate / source venv).

# Run training with torchrun for multi-GPU (DDP)
torchrun --nproc_per_node=4 train_translator.py \
    --model_name_or_path "$MODEL_NAME" \
    --original_tokenizer_path "$ORIGINAL_TOKENIZER_PATH" \
    --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --max_source_length "$MAX_SOURCE_LENGTH" \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --warmup_steps 500 \
    --save_steps 1000 \
    --logging_steps 1 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --report_to none \
    ${DATASET_NAME:+--dataset_name "$DATASET_NAME"} \
    ${DATASET_CONFIG_NAME:+--dataset_config_name "$DATASET_CONFIG_NAME"} \
    ${TEXT_COLUMN:+--text_column "$TEXT_COLUMN"} \
    ${BF16_FLAG:-} \
    ${FP16_FLAG:-}

echo "Training completed! Model saved to $OUTPUT_DIR"
