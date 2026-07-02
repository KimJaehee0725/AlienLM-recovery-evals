# AlienLM Recovery Evals

Compact recovery and robustness tests for AlienLM. This repository keeps runnable scripts, configs, and small summary artifacts only. Planning notes, paper-only notes, raw logs, checkpoints, caches, sample-level dumps, and large generated outputs are intentionally excluded.

Remote: `https://github.com/KimJaehee0725/AlienLM-recovery-evals`

## Repository Layout

| Area | Purpose | Entry point |
| --- | --- | --- |
| `recovery-attacks/o1-frequency-ngram` | Passive frequency / n-gram recovery with no aligned pairs | `run_evaluation.py`, `run_all_experiments.sh` |
| `recovery-attacks/o2-known-plaintext-ngram` | Known-plaintext n-gram recovery with aligned pairs | `run_evaluation.py`, `run_all_experiments.sh` |
| `recovery-attacks/o2-llm-decoding` | LLM-based plaintext recovery from alienized examples | `run_single_exp.sh`, `run_full_exp_parallel.sh` |
| `recovery-attacks/o2-mt-nllb` | Machine-translation style recovery baseline | `scripts/train.sh`, `train_translator.py` |
| `recovery-attacks/o3-weight-based` | Weight-space token mapping recovery | `weight_based_mapping.py`, `run.sh` |
| `robustness/data-volume` | 50k / 150k / full data-volume ablation | `scripts/build_data_subsets.py`, `scripts/train_subset.sh`, `scripts/evaluate_model.sh` |
| `robustness/token-length` | Original vs alien tokenizer length analysis | `scripts/measure_token_lengths.py` |
| `robustness/vocab-overlap` | Token string overlap across tokenizer vocabularies | `scripts/measure_vocab_overlap.py` |
| `robustness/failure-analysis` | GSM8K / MBPP qualitative failure tagging | `scripts/analyze_llama_gsm8k_mbpp.py` |
| `robustness/long-generation` | TruthfulQA generation and LongBench core checks | `scripts/run_truthfulqa_gen.sh`, `scripts/run_longbench_core.sh` |

## External Inputs

Most scripts were written against the paper experiment machine layout. Override these paths when running elsewhere.

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORG_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export ALIENLM_CODE_ROOT=/path/to/AlienLM
export HF_DATASETS_CACHE=/path/to/hf/datasets
export HF_HOME=/path/to/hf/cache
```

OpenAI-backed LLM decoding tests also require:

```bash
export OPENAI_API_KEY=...
```

## Quick Smoke Checks

These checks validate syntax and entrypoint wiring. They do not run paper-scale experiments.

```bash
python -m compileall -q recovery-attacks robustness
find recovery-attacks robustness -type f -name '*.sh' -print -exec bash -n {} \;
```

## Running Each Test

### O1: Frequency / N-gram Recovery

```bash
cd recovery-attacks/o1-frequency-ngram
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer

python run_evaluation.py   --reference_corpus tulu3   --n 2   --train_size 10000   --test_size 1000   --top_k_tokens 1000   --reference_size 10000   --output_dir ./attack_results   --num_proc 8   --batch_size 1000

NON_INTERACTIVE=1 bash run_all_experiments.sh
```

### O2: Known-Plaintext N-gram Recovery

```bash
cd recovery-attacks/o2-known-plaintext-ngram
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORG_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct

python run_evaluation.py   --reference_corpus tulu3   --n 3   --train_size 10000   --test_size 1000   --top_k_tokens 1000   --reference_size 10000   --k_known_pairs 1000   --min_confidence 0.5   --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH"   --output_dir ./attack_results

NON_INTERACTIVE=1 bash run_all_experiments.sh
```

### O2: LLM Decoding Recovery

```bash
cd recovery-attacks/o2-llm-decoding
export OPENAI_API_KEY=...

bash run_single_exp.sh --n_shots 5 --samples 50 --data_file ./data/test_data.jsonl
bash run_full_exp_parallel.sh --few_shots "0 1 5 20" --samples 100
```

### O2: MT / NLLB Translator Recovery

```bash
cd recovery-attacks/o2-mt-nllb
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORIGINAL_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct

bash scripts/train.sh   --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH"   --original_tokenizer_path "$ORIGINAL_TOKENIZER_PATH"   --output_dir ./outputs/translator_model   --num_gpus 1   --num_samples 10000
```

### O3: Weight-Based Mapping Recovery

```bash
cd recovery-attacks/o3-weight-based
export ALIENLM_CODE_ROOT=/path/to/AlienLM
export MODEL_PATH=/path/to/adapted/model
export ALIEN_TOKENIZER=/path/to/alien/tokenizer
export BASE_TOKENIZER=meta-llama/Meta-Llama-3-8B-Instruct

python weight_based_mapping.py   --model-path "$MODEL_PATH"   --alien-tokenizer "$ALIEN_TOKENIZER"   --base-tokenizer "$BASE_TOKENIZER"   --space embedding

bash run.sh
```

### Robustness: Data Volume

```bash
cd robustness/data-volume
python scripts/build_data_subsets.py --output-dir data/subsets
bash scripts/train_subset.sh 50k
bash scripts/train_subset.sh 150k
bash scripts/evaluate_model.sh 50k --suite all --device 0,1 --batch-size 8
python scripts/summarize_main_results.py
```

### Robustness: Token Length

```bash
cd robustness/token-length
python scripts/measure_token_lengths.py
python scripts/measure_original_vs_alien_lengths.py
```

### Robustness: Vocab Overlap

```bash
cd robustness/vocab-overlap
python scripts/measure_vocab_overlap.py
```

### Robustness: Failure Analysis

```bash
cd robustness/failure-analysis
python scripts/analyze_llama_gsm8k_mbpp.py
```

### Robustness: Long Generation

```bash
cd robustness/long-generation
bash scripts/run_truthfulqa_gen.sh llama_original 0
bash scripts/run_longbench_core.sh qwen_alien 0
python scripts/measure_truthfulqa_lengths.py
python scripts/summarize_longbench_core_results.py
```

## Output Policy

Write regenerated artifacts to ignored locations such as `attack_results/`, `logs/`, `outputs/`, `data/`, `data-prepared/`, `.cache/`, or external experiment roots. Do not commit private notes, raw logs, checkpoints, W&B runs, or large sample-level dumps.
