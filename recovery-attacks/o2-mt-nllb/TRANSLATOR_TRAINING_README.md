# Translator Model Training

Code for training a translation model that takes alien (encoded) text as input and recovers the
original text.

## Overview

- **Model**: `facebook/nllb-200-3.3B`
- **Training**: 4-GPU training with DDP (Distributed Data Parallel)
- **Data generation**: For each batch, original-to-alien pairs are generated on the fly using the
  alien tokenizer and the original tokenizer.
- **Training data**: Magpie-Pro-300K-Filtered and Magpie-Reasoning-V1-150K (default).
  - Instruction and response are treated as separate samples.

## Environment setup

```bash
cd ${ALIENLM_CODE_ROOT}/recovery-attacks/o2-mt-nllb

# Create and activate a Python environment (example with venv)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# or
pip install torch transformers datasets accelerate sentencepiece protobuf
```

## How to use

### Run with the helper scripts (recommended)

#### 1. Default training (uses the Magpie datasets — default)
```bash
# Set the alien tokenizer location first
export ALIEN_TOKENIZER_PATH=/path/to/alien_tokenizer

# Start training with default settings
# (Magpie-Pro-300K-Filtered + Magpie-Reasoning-V1-150K)
# Instruction and response are treated as separate samples.
bash scripts/train.sh

# Train with custom settings
bash scripts/train.sh \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir ./outputs/my_model

# Limit the number of samples (for testing)
bash scripts/train.sh --num_samples 10000
```

#### 2. Use a real dataset
```bash
# Use a Hugging Face dataset
bash scripts/train_with_dataset.sh \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --text_column text

# Or call train.sh directly
bash scripts/train.sh \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --text_column text
```

#### 3. Test run (quick validation)
```bash
# Quick test with a small amount of data
bash scripts/train_test.sh
```

#### 4. Run directly (Python)
```bash
# Direct run with torchrun
export ALIEN_TOKENIZER_PATH=/path/to/alien_tokenizer
torchrun --nproc_per_node=4 train_translator.py \
    --model_name_or_path facebook/nllb-200-3.3B \
    --original_tokenizer_path meta-llama/Meta-Llama-3-8B-Instruct \
    --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH" \
    --output_dir ./outputs/translator_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --bf16
```

### Key arguments

- `--model_name_or_path`: Model to use (default: `facebook/nllb-200-3.3B`)
- `--original_tokenizer_path`: Path to the original tokenizer
- `--alien_tokenizer_path`: Path to the alien tokenizer (or set `ALIEN_TOKENIZER_PATH`)
- `--output_dir`: Where to save the model
- `--dataset_name`: Hugging Face dataset name (default: `magpie`)
  - `magpie`: uses both Magpie-Pro-300K-Filtered and Magpie-Reasoning-V1-150K
  - Instruction and response are treated as separate samples
  - Pass any other dataset name to use that dataset
- `--dataset_config_name`: Dataset config name (optional; not used for Magpie)
- `--text_column`: Name of the dataset text column (default: `text`; not used for Magpie)
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--bf16`: Use BF16 precision (recommended)
- `--fp16`: Use FP16 precision

### Script options

The `train.sh` script supports many options:

```bash
bash scripts/train.sh --help  # show all options
```

Key options:
- `--num_gpus N`: Number of GPUs to use (default: 4)
- `--num_epochs N`: Number of training epochs
- `--batch_size N`: Batch size per device
- `--gradient_accumulation N`: Gradient accumulation steps
- `--learning_rate LR`: Learning rate
- `--dataset_name NAME`: Hugging Face dataset name
- `--dataset_config_name NAME`: Dataset config name
- `--text_column COLUMN`: Text column name
- `--num_samples N`: Limit the number of samples (for testing)
- `--bf16` / `--fp16`: Precision

### Example: using a real dataset

```bash
# Use the wikitext dataset
bash scripts/train.sh \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --text_column text \
    --num_epochs 5 \
    --batch_size 8

# Or use train_with_dataset.sh
bash scripts/train_with_dataset.sh \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1
```

## Code layout

```
o2-mt-nllb/
├── translator/
│   ├── __init__.py
│   └── build_translator.py        # Translator class definition
├── train_translator.py            # Main training script
├── scripts/
│   ├── train.sh                   # Main training script (recommended)
│   ├── train_with_dataset.sh      # Dataset-based training script
│   ├── train_test.sh              # Test training script
│   └── train_translator.sh        # Legacy run script
├── eval/                          # Evaluation script and its README
├── pyproject.toml                 # Project config
├── requirements.txt               # Dependency list
└── TRANSLATOR_TRAINING_README.md  # This file
```

## How it works

1. **Data generation**: `TranslationDataset` takes each original text and uses the translator to
   generate the alien (encoded) text per batch.
2. **Training**: A Seq2Seq model is trained with the alien text as input and the original text as
   the target.
3. **DDP**: `torchrun` distributes the data across 4 GPUs for training.

## Notes

- If you run out of GPU memory, reduce `--batch_size` or increase `--gradient_accumulation`.
- NLLB is a large model and needs sufficient GPU memory (at least 8 GB per GPU recommended).
- The first run may take time to download the model.
- Training logs are written to the `logs/` directory automatically.
- Model checkpoints are saved under `outputs/`, in a timestamped directory created automatically.

## Troubleshooting

### CUDA out of memory
- Reduce the batch size or increase gradient accumulation.
- Use `--bf16` or `--fp16` to lower memory usage.

### DDP errors
- Make sure PyTorch and transformers are recent versions.
- Make sure you launch with `torchrun`.
- Make sure CUDA is set up correctly (check GPUs with `nvidia-smi`).
