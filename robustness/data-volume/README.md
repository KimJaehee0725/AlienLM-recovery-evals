# Robustness — Data Volume

Data-volume ablation for AlienLM training. Builds 50k / 150k subsets from the Magpie pool, trains step-matched models, and evaluates them against full-data checkpoints.

## Build Subsets

```bash
python scripts/build_data_subsets.py --output-dir data/subsets
```

## Train

```bash
export ALIENLM_CODE_ROOT=/path/to/AlienLM
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
export ALIENLM_CODE_ROOT=/path/to/AlienLM
export FULL_ONE_EPOCH_MODEL_PATH=/path/to/full/checkpoint
export DATA_VOLUME_50K_MODEL_PATH=/path/to/50k/model
export DATA_VOLUME_150K_MODEL_PATH=/path/to/150k/model

bash scripts/evaluate_model.sh full-1epoch --suite all --device 0,1 --batch-size 8
bash scripts/evaluate_model.sh 50k --suite all --device 0,1 --batch-size 8
bash scripts/evaluate_model.sh 150k --suite all --device 0,1 --batch-size 8
python scripts/summarize_main_results.py \
  --run full=results/$(basename "$FULL_ONE_EPOCH_MODEL_PATH") \
  --run 50k=results/$(basename "$DATA_VOLUME_50K_MODEL_PATH") \
  --run 150k=results/$(basename "$DATA_VOLUME_150K_MODEL_PATH") \
  --md-out results/data_volume_main_comparison.md \
  --json-out results/data_volume_main_comparison.json
```

You may also pass an explicit model path instead of a named shortcut:

```bash
bash scripts/evaluate_model.sh /path/to/model --suite main --device 0 --batch-size 4
```

Outputs go to ignored `logs/`, `data/`, `data-prepared/`, `wandb/`, and external model output roots.
