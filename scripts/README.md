# Scripts

This directory contains repo-level helpers for running or checking the recovery
and robustness evaluations from a fresh clone.

## Quick Checks

```bash
scripts/run_eval.sh list
DRY_RUN=1 scripts/run_eval.sh o2-known-ngram
scripts/run_eval.sh smoke
```

`DRY_RUN=1` prints the command without launching an experiment. Use it when
checking that paths and environment variables are wired correctly.

## Common Environment

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORG_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export ALIENLM_CODE_ROOT=/path/to/AlienLM
export HF_HOME=/path/to/hf/cache
export HF_DATASETS_CACHE=/path/to/hf/datasets
```

LLM-decoding tests additionally require:

```bash
export OPENAI_API_KEY=...
```

## Targets

| Target | Runs |
| --- | --- |
| `smoke` | Python compile check and shell syntax check |
| `outputs` | Lists checkpoint directories under an output root |
| `o1-frequency` | Passive frequency / n-gram recovery |
| `o2-known-ngram` | Known-plaintext n-gram single run |
| `o2-known-sweep` | Known-plaintext n-gram sweep |
| `o2-known-summary` | Known-plaintext summary scripts |
| `o2-llm-single` | LLM-decoding single run |
| `o2-llm-sweep` | LLM-decoding progressive sweep |
| `o2-mt-train` | MT/NLLB translator training |
| `o3-weight` | Weight-space token mapping |
| `data-volume-build` | Build 50k / 150k data subsets |
| `data-volume-train <50k\|150k>` | Train a data-volume ablation model |
| `data-volume-eval [MODEL]` | Evaluate a data-volume model |
| `data-volume-summary` | Summarize data-volume evaluation outputs |
| `token-length` | Token length comparison across original tokenizers |
| `token-length-original-alien` | Original vs Alien tokenizer length comparison |
| `vocab-overlap` | Exact vocab-string overlap |
| `failure-analysis` | GSM8K / MBPP failure analysis |
| `long-truthfulqa <model> [gpu]` | TruthfulQA generation run |
| `long-longbench <model> [gpu]` | LongBench core run |
| `long-longbench-all [gpu]` | LongBench core for all configured models |
| `long-summarize` | Summarize LongBench core outputs |

## Examples

```bash
DRY_RUN=1 ALIEN_TOKENIZER_PATH=/tmp/alien-tokenizer scripts/run_eval.sh o1-frequency

ALIEN_TOKENIZER_PATH=/tmp/alien-tokenizer \
  scripts/run_eval.sh o2-known-ngram

OPENAI_API_KEY=... \
  scripts/run_eval.sh o2-llm-single

ALIENLM_CODE_ROOT=/path/to/AlienLM \
  scripts/run_eval.sh data-volume-train 50k --skip-preprocess
```
