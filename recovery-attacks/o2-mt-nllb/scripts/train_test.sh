#!/bin/bash
# Test training script (small data, quick validation)
# Usage: bash scripts/train_test.sh

set -e

# Resolve the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate your Python environment here if needed (e.g. conda activate / source venv).

# Test settings
NUM_SAMPLES=100  # number of samples for the test
NUM_EPOCHS=1     # short run
BATCH_SIZE=2     # small batch
OUTPUT_DIR="./outputs/translator_model_test_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Running TEST training (quick validation)"
echo "=========================================="
echo "Samples: $NUM_SAMPLES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Call the main training script
bash scripts/train.sh \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --save_steps 50 \
    --logging_steps 10 \
    --bf16

echo ""
echo "Test training completed! Check output: $OUTPUT_DIR"
