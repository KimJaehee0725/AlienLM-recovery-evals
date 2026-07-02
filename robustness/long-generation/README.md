# Robustness — Long Generation

TruthfulQA generation-length checks and a small LongBench core subset.

## Common Setup

```bash
export ALIENLM_CODE_ROOT=/path/to/AlienLM
export LONG_GENERATION_OUTPUT_ROOT=/path/to/long-generation/outputs
export HF_HOME=/path/to/hf/cache
export HF_DATASETS_CACHE=/path/to/hf/datasets

export LLAMA_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export LLAMA_ALIEN_MODEL_PATH=dsba-lab/llama3-8b-instruct-alienlm-full
export QWEN_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export QWEN_ALIEN_MODEL_PATH=/path/to/qwen/alien/model
export GEMMA_MODEL_PATH=google/gemma-2-9b-it
export GEMMA_ALIEN_MODEL_PATH=/path/to/gemma/alien/model
```

## TruthfulQA Generation

```bash
bash scripts/run_truthfulqa_gen.sh llama_original 0
bash scripts/run_truthfulqa_gen.sh llama_alien 0
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

The Llama original and Llama Alien defaults use Hugging Face ids. Qwen/Gemma Alien paths are intentionally supplied through environment variables because those checkpoints may live in local experiment storage.

Outputs go to ignored `logs/`, `outputs/`, and summary files under `results/`.
