# Vocab Overlap

This directory contains rebuttal-only scripts and results for measuring
token-string overlap across Llama, Qwen, and Gemma tokenizers.

Definition:

- Base unit: exact token string from `tokenizer.get_vocab().keys()`
- Main report: overlap excluding special tokens
- Also reported: overlap including special tokens

Metrics:

- `intersection`: `|A ∩ B|`
- `overlap_vs_a`: `|A ∩ B| / |A|`
- `overlap_vs_b`: `|A ∩ B| / |B|`
- `jaccard`: `|A ∩ B| / |A ∪ B|`

Run:

```bash
/workspace/codes/AlienLMv2/.venv/bin/python \
  /workspace/codes/AlienLMv2/icml2026-rebuttal/vocab-overlap/scripts/measure_vocab_overlap.py
```

Reports:

- Summary: `results/vocab_overlap_summary.md`
- JSON: `results/vocab_overlap_summary.json`
