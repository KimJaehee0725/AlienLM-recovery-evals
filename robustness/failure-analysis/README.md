# Robustness — Failure Analysis

Qualitative GSM8K / MBPP failure tagging for AlienLM evaluation outputs.

## Run

```bash
export EVAL_OUTPUT_ROOT=/path/to/eval/outputs
export LLAMA_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export LLAMA_ALIEN_TOKENIZER_PATH=dsba-lab/llama3-8b-instruct-alienlm-full

python scripts/analyze_llama_gsm8k_mbpp.py
```

If your outputs do not follow the default `llama_original/...` and `llama_alien/...` layout, set explicit sample paths:

```bash
export GSM8K_ORIG_PATH=/path/to/original/samples_gsm8k_cot.jsonl
export GSM8K_ALIEN_PATH=/path/to/alien/samples_gsm8k_cot.jsonl
export MBPP_ORIG_PATH=/path/to/original/samples_mbpp.jsonl
export MBPP_ALIEN_PATH=/path/to/alien/samples_mbpp.jsonl
```

## Outputs

The script writes compact summary and detailed Markdown/JSON files under `results/`. Raw evaluation logs and sample-level dumps should remain outside the repo.
