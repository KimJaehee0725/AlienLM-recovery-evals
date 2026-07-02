#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# O3 weight-based token mapping attack (reconstructed from the paper).
# Paper reference: Table 3 (O3, model access); Appendix E.5 (weight-based
# mapping without the bijection seed).
#
# These commands are provided for reproduction. They are NOT executed here.
# -----------------------------------------------------------------------------
set -euo pipefail

# Root of the AlienLM code/checkpoints. Override with your own path.
ALIENLM_CODE_ROOT="${ALIENLM_CODE_ROOT:?set ALIENLM_CODE_ROOT to the AlienLM code/checkpoints root}"

# Adapted model (observed weights) and tokenizers.
MODEL_PATH="${MODEL_PATH:-${ALIENLM_CODE_ROOT}/outputs/Llama3-8B-Instruct-AlienLM/checkpoint}"
ALIEN_TOKENIZER="${ALIEN_TOKENIZER:-${MODEL_PATH}}"
BASE_TOKENIZER="${BASE_TOKENIZER:-meta-llama/Meta-Llama-3-8B-Instruct}"

# Attack both weight spaces.
for SPACE in embedding lm_head; do
    python weight_based_mapping.py \
        --model-path "${MODEL_PATH}" \
        --alien-tokenizer "${ALIEN_TOKENIZER}" \
        --base-tokenizer "${BASE_TOKENIZER}" \
        --space "${SPACE}"
done
