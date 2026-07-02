#!/bin/bash

###############################################################################
# Single LLM Attack Experiment
# 특정 few-shot 크기로 단일 실험 실행
###############################################################################

set -e  # 에러 발생 시 중단

# ============================================================================
# 설정
# ============================================================================

# 기본 데이터 경로
DEFAULT_DATA_FILE="./data/test_data.jsonl"
DATA_FILE="$DEFAULT_DATA_FILE"

# 모델 설정
ATTACK_MODEL=""  # 빈 문자열이면 config 기본값 사용
JUDGE_MODEL=""   # 빈 문자열이면 config 기본값 사용

# 실험 설정
N_SHOTS=5                       # Few-shot 크기 (0 = zero-shot)
N_TEST_SAMPLES=50               # 테스트 샘플 수
MAX_CONCURRENT_REQUESTS=32      # 동시 요청 수 제한
OUTPUT_DIR="./llm_attack_results/single_${N_SHOTS}shot_$(date +%Y%m%d_%H%M%S)"

# OpenAI 설정
# export OPENAI_API_KEY="your-api-key-here"  # 이미 환경변수로 설정되어 있어야 함

# LLM Judge 사용 여부
USE_LLM_JUDGE=true  # true 또는 false

# ============================================================================
# 명령줄 인자 파싱 (선택적)
# ============================================================================

# Usage 함수
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --n_shots N          Number of few-shot examples (default: 5)"
    echo "  -s, --samples N          Number of test samples (default: 50)"
    echo "  -c, --concurrent N       Max concurrent requests (default: 32)"
    echo "  -d, --data_file FILE     Path to JSONL data file (default: $DEFAULT_DATA_FILE)"
    echo "  -a, --attack_model MODEL Attack model (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini')"
    echo "  -j, --judge_model MODEL  Judge model (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini')"
    echo "  -o, --output_dir DIR     Output directory"
    echo "  --no_llm_judge           Disable LLM judge"
    echo "  --list_data              List available data files"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Available data files:"
    if [ -d "./data" ]; then
        for file in ./data/test_data*.jsonl; do
            if [ -f "$file" ]; then
                echo "  - $(basename "$file")"
            fi
        done
    fi
    echo ""
    exit 1
}

# 명령줄 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n_shots)
            N_SHOTS="$2"
            shift 2
            ;;
        -s|--samples)
            N_TEST_SAMPLES="$2"
            shift 2
            ;;
        -c|--concurrent)
            MAX_CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        -a|--attack_model)
            ATTACK_MODEL="$2"
            shift 2
            ;;
        -j|--judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        -d|--data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no_llm_judge)
            USE_LLM_JUDGE=false
            shift
            ;;
        --list_data)
            echo "Available data files in ./data/:"
            if [ -d "./data" ]; then
                for file in ./data/test_data*.jsonl; do
                    if [ -f "$file" ]; then
                        lines=$(wc -l < "$file" 2>/dev/null || echo "?")
                        echo "  - $(basename "$file") ($lines lines)"
                    fi
                done
            else
                echo "  ./data/ directory not found"
            fi
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# 환경 확인
# ============================================================================

echo "========================================================================"
echo "Single LLM Attack Experiment"
echo "========================================================================"
echo ""

# API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# 데이터 파일 확인
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Please run prepare_data.py first to generate the data file."
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Data file:              $DATA_FILE"
echo "  N-shots:                $N_SHOTS"
echo "  Test samples:           $N_TEST_SAMPLES"
echo "  Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
if [ -n "$ATTACK_MODEL" ]; then
    echo "  Attack model:           $ATTACK_MODEL"
else
    echo "  Attack model:           (default from config)"
fi
if [ -n "$JUDGE_MODEL" ]; then
    echo "  Judge model:            $JUDGE_MODEL"
else
    echo "  Judge model:            (default from config)"
fi
echo "  Output dir:             $OUTPUT_DIR"
echo "  LLM Judge:              $USE_LLM_JUDGE"
echo ""

# ============================================================================
# 실험 실행
# ============================================================================

# LLM judge 옵션
LLM_JUDGE_OPT=""
if [ "$USE_LLM_JUDGE" = false ]; then
    LLM_JUDGE_OPT="--no_llm_judge"
fi

# 시작 시간 기록
START_TIME=$(date +%s)

echo "========================================================================"
echo "Starting Single Experiment (${N_SHOTS}-shot)..."
echo "========================================================================"
echo ""

# Python 스크립트 실행
ATTACK_MODEL_OPT=""
JUDGE_MODEL_OPT=""

if [ -n "$ATTACK_MODEL" ]; then
    ATTACK_MODEL_OPT="--attack_model $ATTACK_MODEL"
fi

if [ -n "$JUDGE_MODEL" ]; then
    JUDGE_MODEL_OPT="--judge_model $JUDGE_MODEL"
fi

python run_llm_attack.py \
    --data_file "$DATA_FILE" \
    --mode single \
    --n_shots $N_SHOTS \
    --n_test_samples $N_TEST_SAMPLES \
    --max_concurrent_requests $MAX_CONCURRENT_REQUESTS \
    --output_dir "$OUTPUT_DIR" \
    $ATTACK_MODEL_OPT \
    $JUDGE_MODEL_OPT \
    $LLM_JUDGE_OPT

# 종료 시간 계산
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "Experiment Completed!"
echo "========================================================================"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# ============================================================================
# 결과 요약 출력
# ============================================================================

# 결과 파일 찾기
RESULTS_FILE="$OUTPUT_DIR/llm_attack_${N_SHOTS}shot.json"

if [ -f "$RESULTS_FILE" ]; then
    echo "========================================================================"
    echo "Quick Summary"
    echo "========================================================================"

    # Python으로 간단한 요약 출력
    python -c "
import json

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

metrics = data.get('aggregated_metrics', {})

print('\nMetrics:')
print('-' * 60)
print(f'  BLEU:           {metrics.get(\"bleu_mean\", 0):.4f} ± {metrics.get(\"bleu_std\", 0):.4f}')
print(f'  ROUGE-1 (F1):   {metrics.get(\"rouge1_f_mean\", 0):.4f} ± {metrics.get(\"rouge1_f_std\", 0):.4f}')
print(f'  ROUGE-2 (F1):   {metrics.get(\"rouge2_f_mean\", 0):.4f} ± {metrics.get(\"rouge2_f_std\", 0):.4f}')
print(f'  ROUGE-L (F1):   {metrics.get(\"rougeL_f_mean\", 0):.4f} ± {metrics.get(\"rougeL_f_std\", 0):.4f}')

if 'llm_overall_mean' in metrics:
    print(f'  LLM Judge:      {metrics.get(\"llm_overall_mean\", 0):.2f}/10 ± {metrics.get(\"llm_overall_std\", 0):.2f}')

print('-' * 60)
print(f'  Total samples:  {metrics.get(\"n_samples\", 0)}')
print(f'  Empty results:  {metrics.get(\"n_empty\", 0)}')
print('-' * 60)
    " 2>/dev/null || echo "Could not generate summary"
fi

echo ""
echo "Done!"
