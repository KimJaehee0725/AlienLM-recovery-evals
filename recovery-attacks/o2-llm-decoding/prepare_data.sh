#!/bin/bash

###############################################################################
# LLM Attack용 데이터 준비 스크립트
# Magpie 데이터의 초기 1000개를 사용하여 암호화-평문 쌍 생성
###############################################################################

set -e  # 에러 발생 시 중단

# ============================================================================
# 설정
# ============================================================================

# Tokenizer paths. Override with environment variables when using local assets.
ORG_TOKENIZER_PATH="${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
ALIEN_TOKENIZER_PATH="${ALIEN_TOKENIZER_PATH:-dsba-lab/llama3-8b-instruct-alienlm-full}"

# 출력 디렉토리
OUTPUT_DIR="${OUTPUT_DIR:-./data}"

# 샘플 수
N_SAMPLES="${N_SAMPLES:-1000}"

# 캐시 디렉토리
CACHE_DIR="${CACHE_DIR:-${HF_DATASETS_CACHE:-}}"

# ============================================================================
# 환경 확인
# ============================================================================

echo "========================================================================"
echo "Preparing LLM Attack Data"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Original tokenizer: $ORG_TOKENIZER_PATH"
echo "  Alien tokenizer:    $ALIEN_TOKENIZER_PATH"
echo "  Number of samples:  $N_SAMPLES"
echo "  Output directory:   $OUTPUT_DIR"
echo ""

# ============================================================================
# 데이터 생성
# ============================================================================

CACHE_ARGS=()
if [[ -n "$CACHE_DIR" ]]; then
    CACHE_ARGS+=(--cache_dir "$CACHE_DIR")
fi

python prepare_data.py \
    --org_tokenizer_path "$ORG_TOKENIZER_PATH" \
    --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --batch_size 1024 \
    --num_proc 8 \
    "${CACHE_ARGS[@]}"

echo ""
echo "========================================================================"
echo "Data preparation completed!"
echo "========================================================================"
echo "Files saved to: $OUTPUT_DIR"
echo "  - test_alien.txt"
echo "  - test_original.txt"
echo "========================================================================"
