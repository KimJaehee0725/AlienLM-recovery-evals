# Robustness — Vocab Overlap

Measures exact token-string overlap between tokenizer vocabularies.

## Run

```bash
python scripts/measure_vocab_overlap.py
```

Useful overrides:

```bash
export LLAMA_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export QWEN_TOKENIZER_PATH=Qwen/Qwen2.5-7B-Instruct
export GEMMA_TOKENIZER_PATH=google/gemma-2-9b-it
export OUTPUT_DIR=results
```

Set `LOCAL_FILES_ONLY=1` when all tokenizer assets are already cached locally.

## Outputs

- `results/vocab_overlap_summary.md`
- `results/vocab_overlap_summary.json`

The metric is exact token-string overlap over `tokenizer.get_vocab().keys()`, with special-token-inclusive and special-token-excluded variants where supported.
