# Robustness — Long Generation

TruthfulQA generation-length checks and a small LongBench core subset.

## TruthfulQA Generation

```bash
bash scripts/run_truthfulqa_gen.sh llama_original 0
python scripts/measure_truthfulqa_lengths.py
```

## LongBench Core

```bash
bash scripts/run_longbench_core.sh llama_original 0
bash scripts/run_longbench_core.sh llama_alien 0
bash scripts/run_longbench_core.sh qwen_original 0
bash scripts/run_longbench_core.sh qwen_alien 0
bash scripts/run_longbench_core.sh gemma_original 0
bash scripts/run_longbench_core.sh gemma_alien 0
python scripts/summarize_longbench_core_results.py
```

Scripts default to the original paper machine paths. Override `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, model paths inside the script, or run from an equivalent environment.

Outputs go to ignored `logs/` and external evaluation output roots.
