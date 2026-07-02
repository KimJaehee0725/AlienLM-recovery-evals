# Robustness — Data Volume

Data-volume ablation for AlienLM training. Builds 50k / 150k subsets from the Magpie pool, trains step-matched models, and evaluates them against full-data checkpoints.

## Build Subsets

```bash
python scripts/build_data_subsets.py --output-dir data/subsets
```

## Train

```bash
export WANDB_API_KEY=...
bash scripts/train_subset.sh 50k
bash scripts/train_subset.sh 150k
```

Use `--skip-preprocess` to reuse an existing Axolotl prepared dataset:

```bash
bash scripts/train_subset.sh 50k --skip-preprocess
```

## Evaluate

```bash
bash scripts/evaluate_model.sh full-1epoch --suite all --device 0,1 --batch-size 8
bash scripts/evaluate_model.sh 50k --suite all --device 0,1 --batch-size 8
bash scripts/evaluate_model.sh 150k --suite all --device 0,1 --batch-size 8
python scripts/summarize_main_results.py
```

Outputs go to ignored `logs/`, `data/`, `data-prepared/`, `wandb/`, and external model output roots.
