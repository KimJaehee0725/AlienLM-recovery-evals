#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Missing WANDB_API_KEY. Export it before launching the paired run." >&2
  exit 1
fi

"$SCRIPT_DIR/train_subset.sh" 50k
"$SCRIPT_DIR/train_subset.sh" 150k
