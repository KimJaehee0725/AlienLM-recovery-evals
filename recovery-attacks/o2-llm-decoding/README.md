# O2 — LLM Decoding Recovery

LLM-based recovery test where an external model attempts to decode alienized text from zero-shot or few-shot examples.

## Inputs

- `OPENAI_API_KEY`: required.
- `./data/test_data.jsonl`: JSONL input file with alien/plain examples.
- Optional model overrides: `--attack_model`, `--judge_model`.

## Single Run

```bash
export OPENAI_API_KEY=...

bash run_single_exp.sh   --n_shots 5   --samples 50   --concurrent 16   --data_file ./data/test_data.jsonl   --output_dir ./llm_attack_results/single_5shot
```

## Progressive Sweep

```bash
export OPENAI_API_KEY=...

bash run_full_exp_parallel.sh   --few_shots "0 1 5 20"   --samples 100   --concurrent 32   --data_file ./data/test_data.jsonl
```

Use `--non_parallel` for the non-parallel leakage template and `--no_llm_judge` to skip judge calls.

Outputs go to ignored `llm_attack_results/`.
