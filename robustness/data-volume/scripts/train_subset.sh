#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <50k|150k> [--skip-preprocess]" >&2
  exit 1
fi

SIZE="$1"
shift || true

SKIP_PREPROCESS=false
for arg in "$@"; do
  if [[ "$arg" == "--skip-preprocess" ]]; then
    SKIP_PREPROCESS=true
  else
    echo "Unknown argument: $arg" >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ALIENLM_CODE_ROOT="${ALIENLM_CODE_ROOT:?set ALIENLM_CODE_ROOT to the local AlienLM repo used for Axolotl training}"
AXOLOTL_BIN="${AXOLOTL_BIN:-$ALIENLM_CODE_ROOT/.venv/bin/axolotl}"
export PATH="$ALIENLM_CODE_ROOT/.venv/bin:$PATH"

case "$SIZE" in
  50k)
    CONFIG_PATH="$DV_DIR/configs/llama3-8b-alienlm-qwenv2-50k-1epoch.yaml"
    DATA_PATH="$DV_DIR/data/subsets/magpie_mix_50k_seed42.jsonl"
    ;;
  150k)
    CONFIG_PATH="$DV_DIR/configs/llama3-8b-alienlm-qwenv2-150k-1epoch.yaml"
    DATA_PATH="$DV_DIR/data/subsets/magpie_mix_150k_seed42.jsonl"
    ;;
  *)
    echo "Unsupported size: $SIZE. Expected 50k or 150k." >&2
    exit 1
    ;;
esac

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing subset data: $DATA_PATH" >&2
  echo "Run build_data_subsets.py first." >&2
  exit 1
fi

DEFAULT_CACHE_ROOT="${HF_HOME:-$DV_DIR/.cache}"
export HF_HOME="$DEFAULT_CACHE_ROOT"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DEFAULT_CACHE_ROOT/hf_datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_PROJECT="${WANDB_PROJECT:-magpie-alienlmv2}"
export WANDB_DIR="${WANDB_DIR:-$DV_DIR/wandb}"
RUN_TAG="${RUN_TAG:-}"
LOG_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
  LOG_SUFFIX="_${RUN_TAG}"
fi

mkdir -p "$DV_DIR/logs"
mkdir -p "$WANDB_DIR"

echo "[train_subset]"
echo "size=$SIZE"
echo "config=$CONFIG_PATH"
echo "data=$DATA_PATH"
echo "alienlm_code_root=$ALIENLM_CODE_ROOT"
echo "axolotl_bin=$AXOLOTL_BIN"
echo "PATH=$PATH"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_DIR=$WANDB_DIR"
echo "RUN_TAG=${RUN_TAG:-<none>}"
echo

if [[ ! -x "$AXOLOTL_BIN" ]]; then
  echo "Missing axolotl executable: $AXOLOTL_BIN" >&2
  echo "Restore the root submission environment first." >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Missing WANDB_API_KEY. Export it before training." >&2
  exit 1
fi

if [[ "$SKIP_PREPROCESS" == false ]]; then
  echo "[preprocess]"
  "$AXOLOTL_BIN" preprocess "$CONFIG_PATH" | tee "$DV_DIR/logs/preprocess_${SIZE}${LOG_SUFFIX}.log"
  echo
fi

echo "[train]"
"$AXOLOTL_BIN" train "$CONFIG_PATH" | tee "$DV_DIR/logs/train_${SIZE}${LOG_SUFFIX}.log"
