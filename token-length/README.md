# Token Length Comparison

This directory contains rebuttal-only scripts and results for comparing
per-sample token lengths on the 450k Magpie training pool.

Current measurement setup:

- Data pool:
  - `Magpie-Align/Magpie-Pro-300K-Filtered`
  - `Magpie-Align/Magpie-Reasoning-V1-150K`
- Sampling:
  - `10,000` samples
  - uniform random sample over the combined 450k pool
  - fixed `seed=42`
- Tokenizers:
  - Llama 3 8B Instruct
  - Qwen 2.5 7B Instruct
  - Gemma 2 9B IT
- Length definition:
  - tokenized length of the same plain text per sample
  - text is built from the sample conversation contents joined as raw text
  - `add_special_tokens=False`

Run:

```bash
/workspace/codes/AlienLMv2/.venv/bin/python \
  /workspace/codes/AlienLMv2/icml2026-rebuttal/token-length/scripts/measure_token_lengths.py
```

Reports:

- Summary: `results/token_length_summary.md`
- Detailed report: `results/token_length_detailed_report.md`
