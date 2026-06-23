# Long Generation Rebuttal

This directory contains scripts and notes for rebuttal experiments that address the reviewer request for free-form and long-context generation.

## Current plan

- Recover existing `truthfulqa_gen` results from prior runs.
- Run missing `truthfulqa_gen` for Llama original.
- Measure per-model input/output length statistics from `log_samples`.
- If the results look usable, proceed to a small LongBench subset.

## GPU note

- `truthfulqa_gen` is lightweight and should run comfortably on a single A100 80GB.
- For LongBench, memory is not the main bottleneck for 8B models on a single A100. The stricter limit is context length:
  - Llama 3 8B: `8192`
  - Gemma 2 9B: `8192`
  - Qwen2.5 7B: `32768`
- Because AlienLM can increase prompt token length, Qwen is the safest first choice for LongBench.
