# Robustness — Token Length

Compares tokenized sequence lengths under original and alien tokenizers.

## Run

```bash
python scripts/measure_token_lengths.py
python scripts/measure_original_vs_alien_lengths.py
```

Useful overrides:

```bash
export SAMPLE_SIZE=10000
export HF_DATASETS_CACHE=/path/to/hf/datasets
export LLAMA_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export LLAMA_ALIEN_TOKENIZER_PATH=dsba-lab/llama3-8b-instruct-alienlm-full
export QWEN_ALIEN_TOKENIZER_PATH=/path/to/qwen/alien/tokenizer
export GEMMA_ALIEN_TOKENIZER_PATH=/path/to/gemma/alien/tokenizer
```

Set `LOCAL_FILES_ONLY=1` when all tokenizer assets are already cached locally.

## Outputs

The scripts write compact summaries under `results/`, including Markdown and JSON summaries. Large sample-level dumps should stay outside the repo or under ignored output paths.
