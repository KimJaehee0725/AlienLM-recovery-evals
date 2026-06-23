#!/usr/bin/env bash

set -euo pipefail

OUT_ROOT="${1:-/workspace/data2/jaehee/AlienLM/outputs}"

if [[ ! -d "$OUT_ROOT" ]]; then
  echo "Output root not found: $OUT_ROOT" >&2
  exit 1
fi

echo "[output root]"
echo "$OUT_ROOT"
echo

echo "[top-level experiment dirs]"
find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
echo

echo "[checkpoint dirs up to depth 2]"
find "$OUT_ROOT" -mindepth 1 -maxdepth 2 -type d -name 'checkpoint-*' | sort
echo

echo "[latest checkpoint per immediate child dir]"
find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r exp_dir; do
  latest_ckpt="$(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  if [[ -n "$latest_ckpt" ]]; then
    printf "%s\t%s\n" "$(basename "$exp_dir")" "$latest_ckpt"
  fi
done
