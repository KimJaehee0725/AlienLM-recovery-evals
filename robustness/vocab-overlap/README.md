# Robustness — Vocab Overlap

Measures exact token-string overlap between tokenizer vocabularies.

## Run

```bash
python scripts/measure_vocab_overlap.py
```

## Outputs

- `results/vocab_overlap_summary.md`
- `results/vocab_overlap_summary.json`

The metric is exact token-string overlap over `tokenizer.get_vocab().keys()`, with special-token-inclusive and special-token-excluded variants where supported.
