#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_eval.sh <target> [target args...]

Targets:
  list                         Show this target list
  smoke                        Syntax/entrypoint smoke checks
  outputs [OUT_ROOT]           List checkpoints under an output root

  o1-frequency                 O1 passive frequency/ngram recovery
  o2-known-ngram               O2 known-plaintext ngram single run
  o2-known-sweep               O2 known-plaintext ngram sweep
  o2-known-summary             O2 known-plaintext result summaries
  o2-llm-single                O2 LLM decoding single run
  o2-llm-sweep                 O2 LLM decoding progressive sweep
  o2-mt-train                  O2 MT/NLLB translator training
  o3-weight                    O3 weight-space mapping recovery

  data-volume-build            Build 50k/150k Magpie subsets
  data-volume-train <50k|150k> Train a data-volume ablation model
  data-volume-eval [MODEL]     Evaluate a data-volume model
  data-volume-summary          Summarize data-volume eval runs
  token-length                 Original-tokenizer length summary
  token-length-original-alien  Original vs Alien tokenizer length summary
  vocab-overlap                Tokenizer vocab overlap summary
  failure-analysis             GSM8K/MBPP failure analysis
  long-truthfulqa <model> [gpu] TruthfulQA generation run
  long-longbench <model> [gpu] LongBench core run
  long-longbench-all [gpu]     LongBench core for all configured models
  long-summarize               LongBench result summary

Common environment:
  ALIEN_TOKENIZER_PATH         Alien tokenizer checkpoint or Hugging Face id
  ORG_TOKENIZER_PATH           Original tokenizer, defaults to Llama 3 Instruct
  ALIENLM_CODE_ROOT            Local AlienLM repo for Axolotl/lm_eval helpers
  OPENAI_API_KEY               Required only for O2 LLM decoding
  DRY_RUN=1                    Print commands without executing them
USAGE
}

need_env() {
  local name="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_in() {
  local dir="$1"
  shift
  echo "+ (cd ${dir#"$ROOT_DIR"/} && $*)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  (cd "$dir" && "$@")
}

target="${1:-list}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${target}" in
  list|help|-h|--help)
    usage
    ;;

  smoke)
    run "${PYTHON_BIN}" -m compileall -q "${ROOT_DIR}/recovery-attacks" "${ROOT_DIR}/robustness"
    run find "${ROOT_DIR}/scripts" "${ROOT_DIR}/recovery-attacks" "${ROOT_DIR}/robustness" -type f -name "*.sh" -exec bash -n {} ";"
    ;;

  outputs)
    run "${ROOT_DIR}/scripts/list_outputs.sh" "$@"
    ;;

  o1-frequency)
    need_env ALIEN_TOKENIZER_PATH
    run_in "${ROOT_DIR}/recovery-attacks/o1-frequency-ngram" \
      "${PYTHON_BIN}" run_evaluation.py \
      --reference_corpus "${REFERENCE_CORPUS:-tulu3}" \
      --n "${N_GRAM:-2}" \
      --train_size "${TRAIN_SIZE:-10000}" \
      --test_size "${TEST_SIZE:-1000}" \
      --top_k_tokens "${TOP_K_TOKENS:-1000}" \
      --reference_size "${REFERENCE_SIZE:-10000}" \
      --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}" \
      --org_tokenizer_path "${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}" \
      --output_dir "${OUTPUT_DIR:-./attack_results}" \
      --num_proc "${NUM_PROC:-8}" \
      --batch_size "${BATCH_SIZE:-1000}" \
      "$@"
    ;;

  o2-known-ngram)
    need_env ALIEN_TOKENIZER_PATH
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" \
      "${PYTHON_BIN}" run_evaluation.py \
      --reference_corpus "${REFERENCE_CORPUS:-tulu3}" \
      --n "${N_GRAM:-3}" \
      --train_size "${TRAIN_SIZE:-10000}" \
      --test_size "${TEST_SIZE:-1000}" \
      --top_k_tokens "${TOP_K_TOKENS:-1000}" \
      --reference_size "${REFERENCE_SIZE:-10000}" \
      --k_known_pairs "${K_KNOWN_PAIRS:-1000}" \
      --min_confidence "${MIN_CONFIDENCE:-0.5}" \
      --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}" \
      --org_tokenizer_path "${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}" \
      --output_dir "${OUTPUT_DIR:-./attack_results}" \
      --num_proc "${NUM_PROC:-8}" \
      --batch_size "${BATCH_SIZE:-1000}" \
      "$@"
    ;;

  o2-known-sweep)
    need_env ALIEN_TOKENIZER_PATH
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" \
      env NON_INTERACTIVE="${NON_INTERACTIVE:-1}" bash run_all_experiments.sh "$@"
    ;;

  o2-known-summary)
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" "${PYTHON_BIN}" scripts/summarize_known_pairs_results.py "$@"
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" "${PYTHON_BIN}" scripts/summarize_minconf_results.py "$@"
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" "${PYTHON_BIN}" scripts/summarize_observer_tokenizer_results.py "$@"
    run_in "${ROOT_DIR}/recovery-attacks/o2-known-plaintext-ngram" "${PYTHON_BIN}" scripts/summarize_seedpool_results.py "$@"
    ;;

  o2-llm-single)
    need_env OPENAI_API_KEY
    run_in "${ROOT_DIR}/recovery-attacks/o2-llm-decoding" \
      bash run_single_exp.sh \
      --n_shots "${N_SHOTS:-5}" \
      --samples "${SAMPLES:-50}" \
      --concurrent "${CONCURRENT:-16}" \
      --data_file "${DATA_FILE:-./data/test_data.jsonl}" \
      --output_dir "${OUTPUT_DIR:-./llm_attack_results/single_5shot}" \
      "$@"
    ;;

  o2-llm-sweep)
    need_env OPENAI_API_KEY
    run_in "${ROOT_DIR}/recovery-attacks/o2-llm-decoding" \
      bash run_full_exp_parallel.sh \
      --few_shots "${FEW_SHOTS:-0 1 5 20}" \
      --samples "${SAMPLES:-100}" \
      --concurrent "${CONCURRENT:-32}" \
      --data_file "${DATA_FILE:-./data/test_data.jsonl}" \
      "$@"
    ;;

  o2-mt-train)
    need_env ALIEN_TOKENIZER_PATH
    run_in "${ROOT_DIR}/recovery-attacks/o2-mt-nllb" \
      bash scripts/train.sh \
      --alien_tokenizer_path "${ALIEN_TOKENIZER_PATH}" \
      --original_tokenizer_path "${ORIGINAL_TOKENIZER_PATH:-${ORG_TOKENIZER_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}}" \
      --output_dir "${OUTPUT_DIR:-./outputs/translator_model}" \
      --num_gpus "${NUM_GPUS:-1}" \
      --num_samples "${NUM_SAMPLES:-10000}" \
      --report_to "${REPORT_TO:-none}" \
      "$@"
    ;;

  o3-weight)
    need_env ALIENLM_CODE_ROOT
    run_in "${ROOT_DIR}/recovery-attacks/o3-weight-based" bash run.sh "$@"
    ;;

  data-volume-build)
    run_in "${ROOT_DIR}/robustness/data-volume" "${PYTHON_BIN}" scripts/build_data_subsets.py "$@"
    ;;

  data-volume-train)
    size="${1:-}"
    if [[ -z "${size}" ]]; then
      echo "Usage: scripts/run_eval.sh data-volume-train <50k|150k> [args...]" >&2
      exit 1
    fi
    shift || true
    need_env ALIENLM_CODE_ROOT
    run_in "${ROOT_DIR}/robustness/data-volume" bash scripts/train_subset.sh "${size}" "$@"
    ;;

  data-volume-eval)
    model_spec="${1:-50k}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    run_in "${ROOT_DIR}/robustness/data-volume" bash scripts/evaluate_model.sh "${model_spec}" "$@"
    ;;

  data-volume-summary)
    run_in "${ROOT_DIR}/robustness/data-volume" "${PYTHON_BIN}" scripts/summarize_main_results.py "$@"
    ;;

  token-length)
    run_in "${ROOT_DIR}/robustness/token-length" "${PYTHON_BIN}" scripts/measure_token_lengths.py "$@"
    ;;

  token-length-original-alien)
    run_in "${ROOT_DIR}/robustness/token-length" "${PYTHON_BIN}" scripts/measure_original_vs_alien_lengths.py "$@"
    ;;

  vocab-overlap)
    run_in "${ROOT_DIR}/robustness/vocab-overlap" "${PYTHON_BIN}" scripts/measure_vocab_overlap.py "$@"
    ;;

  failure-analysis)
    run_in "${ROOT_DIR}/robustness/failure-analysis" "${PYTHON_BIN}" scripts/analyze_llama_gsm8k_mbpp.py "$@"
    ;;

  long-truthfulqa)
    run_in "${ROOT_DIR}/robustness/long-generation" bash scripts/run_truthfulqa_gen.sh "$@"
    ;;

  long-longbench)
    run_in "${ROOT_DIR}/robustness/long-generation" bash scripts/run_longbench_core.sh "$@"
    ;;

  long-longbench-all)
    run_in "${ROOT_DIR}/robustness/long-generation" bash scripts/run_longbench_core_all.sh "$@"
    ;;

  long-summarize)
    run_in "${ROOT_DIR}/robustness/long-generation" "${PYTHON_BIN}" scripts/summarize_longbench_core_results.py "$@"
    ;;

  *)
    echo "Unknown target: ${target}" >&2
    usage >&2
    exit 1
    ;;
esac
