# O2 — Known-Plaintext N-gram Recovery

Known-plaintext recovery test that combines a small set of aligned plaintext/alien examples with n-gram statistics from a public reference corpus.

## Inputs

- `ALIEN_TOKENIZER_PATH`: alien tokenizer checkpoint or HF id.
- `ORG_TOKENIZER_PATH`: original tokenizer, default `meta-llama/Meta-Llama-3-8B-Instruct`.
- `CACHE_DIR`: optional Hugging Face dataset cache.

## Single Run

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORG_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct

python run_evaluation.py   --reference_corpus tulu3   --n 3   --train_size 10000   --test_size 1000   --top_k_tokens 1000   --reference_size 10000   --k_known_pairs 1000   --min_confidence 0.5   --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH"   --org_tokenizer_path "$ORG_TOKENIZER_PATH"   --output_dir ./attack_results
```

## Sweep

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
NON_INTERACTIVE=1 bash run_all_experiments.sh
```

## Summaries

```bash
python scripts/summarize_known_pairs_results.py
python scripts/summarize_minconf_results.py
python scripts/summarize_observer_tokenizer_results.py
python scripts/summarize_seedpool_results.py
```

Outputs go to ignored `attack_results/` and `logs/` directories.
