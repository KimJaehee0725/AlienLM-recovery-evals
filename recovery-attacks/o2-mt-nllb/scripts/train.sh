#!/bin/bash
# Script for full training
# Usage: bash scripts/train.sh [options]

set -e

# Default settings
MODEL_NAME="facebook/nllb-200-3.3B"
ORIGINAL_TOKENIZER_PATH="${ORIGINAL_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:?set ALIEN_TOKENIZER_PATH to the alien tokenizer}"
OUTPUT_DIR="./outputs/translator_model_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-5
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=512
WARMUP_STEPS=500
SAVE_STEPS=500
LOGGING_STEPS=1

# Dataset settings (default: magpie - uses Magpie-Pro and Magpie-Reasoning)
DATASET_NAME="magpie"
DATASET_CONFIG_NAME=""
TEXT_COLUMN="text"

# Other settings
USE_BF16=true
CACHE_DIR=""
NUM_SAMPLES=""
REPORT_TO="none"  # Options: "wandb", "tensorboard", "none"

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
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
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --report_to)
            REPORT_TO="$2"
            shift 2
            ;;
        --bf16)
            USE_BF16=true
            shift
            ;;
        --fp16)
            USE_BF16=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name MODEL_NAME                    Model name or path (default: facebook/nllb-200-3.3B)"
            echo "  --original_tokenizer_path PATH            Original tokenizer path"
            echo "  --alien_tokenizer_path PATH               Alien tokenizer path"
            echo "  --output_dir DIR                          Output directory"
            echo "  --num_gpus N                              Number of GPUs (default: 4)"
            echo "  --num_epochs N                            Number of epochs (default: 3)"
            echo "  --batch_size N                            Batch size per device (default: 4)"
            echo "  --gradient_accumulation N                 Gradient accumulation steps (default: 4)"
            echo "  --learning_rate LR                         Learning rate (default: 5e-5)"
            echo "  --dataset_name NAME                       Dataset name from HuggingFace"
            echo "  --dataset_config_name NAME                Dataset config name"
            echo "  --text_column COLUMN                      Text column name (default: text)"
            echo "  --max_source_length N                     Max source length (default: 512)"
            echo "  --max_target_length N                     Max target length (default: 512)"
            echo "  --warmup_steps N                          Warmup steps (default: 500)"
            echo "  --save_steps N                            Save steps (default: 500)"
            echo "  --logging_steps N                         Logging steps (default: 100)"
            echo "  --cache_dir DIR                           Cache directory"
            echo "  --num_samples N                           Number of samples (for testing)"
            echo "  --bf16                                    Use bf16 (default)"
            echo "  --fp16                                    Use fp16"
            echo "  --help                                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Resolve the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate your Python environment here if needed (e.g. conda activate / source venv).

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Check the CUDA setup
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
fi

# Check the GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $GPU_COUNT available."
        NUM_GPUS=$GPU_COUNT
    fi
fi

# Print the training configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Original Tokenizer: $ORIGINAL_TOKENIZER_PATH"
echo "Alien Tokenizer: $ALIEN_TOKENIZER_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size (per device): $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Source Length: $MAX_SOURCE_LENGTH"
echo "Max Target Length: $MAX_TARGET_LENGTH"
if [ -n "$DATASET_NAME" ]; then
    if [ "$DATASET_NAME" = "magpie" ]; then
        echo "Dataset: Magpie (Magpie-Pro-300K-Filtered + Magpie-Reasoning-V1-150K)"
        echo "Note: Instruction and response are treated as separate samples"
    else
        echo "Dataset: $DATASET_NAME"
        [ -n "$DATASET_CONFIG_NAME" ] && echo "Dataset Config: $DATASET_CONFIG_NAME"
        echo "Text Column: $TEXT_COLUMN"
    fi
else
    echo "Dataset: Dummy data (no dataset specified)"
fi
echo "Precision: $([ "$USE_BF16" = true ] && echo "bf16" || echo "fp16")"
echo "Log File: $LOG_FILE"
echo "=========================================="
echo ""

# Build the training command
TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS train_translator.py \
    --model_name_or_path \"$MODEL_NAME\" \
    --original_tokenizer_path \"$ORIGINAL_TOKENIZER_PATH\" \
    --alien_tokenizer_path \"$ALIEN_TOKENIZER_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --report_to \"$REPORT_TO\""

# Append optional flags
[ -n "$DATASET_NAME" ] && TRAIN_CMD="$TRAIN_CMD --dataset_name \"$DATASET_NAME\""
[ -n "$DATASET_CONFIG_NAME" ] && TRAIN_CMD="$TRAIN_CMD --dataset_config_name \"$DATASET_CONFIG_NAME\""
[ -n "$TEXT_COLUMN" ] && TRAIN_CMD="$TRAIN_CMD --text_column \"$TEXT_COLUMN\""
[ -n "$CACHE_DIR" ] && TRAIN_CMD="$TRAIN_CMD --cache_dir \"$CACHE_DIR\""
[ -n "$NUM_SAMPLES" ] && TRAIN_CMD="$TRAIN_CMD --num_samples $NUM_SAMPLES"

if [ "$USE_BF16" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --bf16"
else
    TRAIN_CMD="$TRAIN_CMD --fp16"
fi

# Run training (also tee to the log file)
echo "Starting training at $(date)"
echo "Command: $TRAIN_CMD"
echo ""

# Send both stdout and stderr to the log file and the screen
eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"

# Training finished
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
    echo "=========================================="
    exit $EXIT_CODE
fi
