#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <full-0.5epoch|full-1epoch|full-2epoch|50k|150k|/abs/model/path> [--suite main|code|all] [--device 0,1] [--batch-size 8] [--tp 1] [--port 29610]" >&2
  exit 1
fi

MODEL_SPEC="$1"
shift || true

SUITE="all"
DEVICE="0,1"
BATCH_SIZE="8"
TP="1"
PORT="29610"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$DV_DIR/../.." && pwd)"
RESULTS_DIR="$DV_DIR/results"
ROOT_VENV_BIN="$REPO_DIR/.venv/bin"

if [[ -d "$ROOT_VENV_BIN" ]]; then
  export PATH="$ROOT_VENV_BIN:$PATH"
fi

resolve_model_path() {
  case "$1" in
    full-0.5epoch)
      echo "/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-2327"
      ;;
    full-1epoch)
      echo "/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-4654"
      ;;
    full-2epoch)
      echo "/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-9306"
      ;;
    50k)
      echo "/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/data-volume/Llama3-8B-Instruct-AlienLM-50k-stepmatch4654-qwenv2-seed42-ga2"
      ;;
    150k)
      echo "/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/data-volume/Llama3-8B-Instruct-AlienLM-150k-stepmatch4654-qwenv2-seed42-ga2"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

MODEL_PATH="$(resolve_model_path "$MODEL_SPEC")"
RUN_NAME="$(basename "$MODEL_PATH")"

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

DEFAULT_CACHE_ROOT="${HF_HOME:-$DV_DIR/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DEFAULT_CACHE_ROOT/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$DEFAULT_CACHE_ROOT/hf_models}"
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE_ROOT/hf_home}"
export CUDA_VISIBLE_DEVICES="$DEVICE"

mkdir -p "$RESULTS_DIR/$RUN_NAME"

echo "[evaluate_model]"
echo "model_spec=$MODEL_SPEC"
echo "model_path=$MODEL_PATH"
echo "suite=$SUITE"
echo "device=$DEVICE"
echo "batch_size=$BATCH_SIZE"
echo "tp=$TP"
echo "port=$PORT"
echo

run_main_suite() {
  local model_path="$1"
  local output_dir="$2"
  local device="$3"
  local batch_size="$4"
  local port="$5"

  local tasks=("mmlu" "arc_easy" "arc_challenge" "hellaswag" "winogrande" "truthfulqa_mc1" "gsm8k_cot")
  local few_shots=(5 25 25 10 5 0 5)
  local num_gpus
  num_gpus=$(echo "$device" | awk -F, '{print NF}')
  local model_args="pretrained=$model_path,trust_remote_code=True,dtype=bfloat16"

  for idx in "${!tasks[@]}"; do
    local task_name="${tasks[$idx]}"
    local num_fewshot="${few_shots[$idx]}"
    local task_output="$output_dir/$task_name/${num_fewshot}-shot"

    echo "========================================================================="
    echo "Starting evaluation for task: $task_name ($num_fewshot-shot)"
    echo "========================================================================="
    echo "Model: $model_path"
    echo "Device(s): $device ($num_gpus GPUs)"
    echo "Batch size: $batch_size"
    echo "Port: $port"
    echo "Output path: $task_output"
    echo "Using chat template: true"
    echo "-------------------------------------------------------------------------"

    accelerate launch \
      --main_process_port "$port" \
      --num_processes "$num_gpus" \
      -m lm_eval --model hf \
      --model_args "$model_args" \
      --tasks "$task_name" \
      --num_fewshot "$num_fewshot" \
      --batch_size "$batch_size" \
      --output_path "$task_output" \
      --log_samples \
      --apply_chat_template \
      --fewshot_as_multiturn

    echo "-------------------------------------------------------------------------"
    echo "Finished evaluation for task: $task_name"
    echo "Results saved in $task_output"
    echo "========================================================================="
    echo
  done
}

if [[ "$SUITE" == "main" || "$SUITE" == "all" ]]; then
  run_main_suite "$MODEL_PATH" "$RESULTS_DIR/$RUN_NAME/main" "$DEVICE" "$BATCH_SIZE" "$PORT"
fi

if [[ "$SUITE" == "code" || "$SUITE" == "all" ]]; then
  bash "$REPO_DIR/icml2026-submition/eval/scripts/utils/evaluate_code_evalplus.sh" \
    --model_path "$MODEL_PATH" \
    --output_dir "$RESULTS_DIR/$RUN_NAME/code" \
    --tensor_parallel_size "$TP"
fi
