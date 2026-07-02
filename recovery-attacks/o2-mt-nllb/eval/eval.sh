#!/bin/bash
# Configure via environment variables
# Path to a trained translator checkpoint (set this before running).
export MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a trained translator checkpoint}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.1-2025-11-13}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-16}"

# Count the number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Run torchrun from this script's directory
cd "$(dirname "$0")"
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    evaluate_translator.py \
    --model_path "$MODEL_PATH" \
    --data_path data/test_data.jsonl \
    --max_samples 100 \
    --output_path results/evaluation_results.json \
    --api_key "$OPENAI_API_KEY" \
    --judge_model "$JUDGE_MODEL" \
    --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --use_llm_judge
