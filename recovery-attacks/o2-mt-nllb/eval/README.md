# Translator Model Evaluation

Script for decoding evaluation data with a trained translator model and scoring the results.

## How to use

### Method 1: Use the eval.sh script (recommended)

`eval.sh` lets you configure everything through environment variables or in-script variables.

#### Basic usage

```bash
# Set the required environment variables, then run
export MODEL_PATH="./outputs/translator_model"   # trained checkpoint
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
bash eval.sh
```

#### Configure via environment variables

```bash
export MODEL_PATH="./outputs/translator_model"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
export JUDGE_MODEL="gpt-4o"
export MAX_CONCURRENT_REQUESTS=32
export CUDA_VISIBLE_DEVICES=0
export DEVICE="cuda"
export MAX_LENGTH=1024
export BATCH_SIZE=16

# Run
bash eval.sh
```

#### Editing in-script variables

You can open `eval.sh` and edit the variables near the top directly:

```bash
# Model path (required)
MODEL_PATH="./outputs/translator_model"

# OpenAI API settings
OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"  # always read from the environment
JUDGE_MODEL="gpt-5.1"  # e.g. gpt-5.1, gpt-4o, gpt-4o-mini
MAX_CONCURRENT_REQUESTS=32

# GPU settings
DEVICE="cuda"  # cuda or cpu
CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU IDs to use (e.g. 0, 1, 0,1)

# Generation settings
MAX_LENGTH=1024
BATCH_SIZE=8

# Evaluation settings
USE_LLM_JUDGE=true  # true or false
```

### Method 2: Run the Python script directly

```bash
python evaluate_translator.py \
    --model_path /path/to/trained/model \
    --data_path data/test_data.jsonl \
    --output_path results/evaluation_results.json \
    --judge_model gpt-5.1 \
    --max_concurrent_requests 10 \
    --device cuda \
    --max_length 512 \
    --batch_size 8 \
    --use_llm_judge
```

The OpenAI API key is read from the `OPENAI_API_KEY` environment variable. You may also pass
`--api_key "$OPENAI_API_KEY"`, but never hardcode a literal key.

### Key options

#### eval.sh variables

- `MODEL_PATH`: Path to the trained translator model (required)
- `DATA_PATH`: Path to the evaluation data file (default: `data/test_data.jsonl`)
- `OUTPUT_PATH`: Path to save evaluation results (default: `results/evaluation_results.json`)
- `OPENAI_API_KEY`: OpenAI API key (set as an environment variable)
- `JUDGE_MODEL`: LLM judge model name (default: `gpt-5.1`)
- `MAX_CONCURRENT_REQUESTS`: Number of concurrent LLM judge requests (default: 10)
- `DEVICE`: Device to use (default: `cuda`)
- `CUDA_VISIBLE_DEVICES`: GPU IDs to use (default: `0`)
- `MAX_LENGTH`: Maximum generation length (default: 512)
- `BATCH_SIZE`: Decoding batch size (default: 8)
- `USE_LLM_JUDGE`: Whether to use LLM-as-a-judge evaluation (default: `true`)

#### Python script options

- `--model_path`: Path to the trained translator model (required)
- `--data_path`: Path to the evaluation data file (default: `data/test_data.jsonl`)
- `--output_path`: Path to save evaluation results (default: `results/evaluation_results.json`)
- `--api_key`: OpenAI API key (optional; the `OPENAI_API_KEY` environment variable is used otherwise)
- `--judge_model`: LLM judge model name (default: `gpt-5.1`)
- `--max_concurrent_requests`: Number of concurrent LLM judge requests (default: 10)
- `--device`: Device to use (default: `cuda`)
- `--max_length`: Maximum generation length (default: 512)
- `--batch_size`: Decoding batch size (default: 8)
- `--use_llm_judge`: Whether to use LLM-as-a-judge evaluation (default: True)

### Examples

```bash
# Example 1: evaluate with default settings
export MODEL_PATH="./outputs/translator_model"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
bash eval.sh

# Example 2: custom settings via environment variables
export MODEL_PATH="./outputs/translator_model"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
export JUDGE_MODEL="gpt-4o"
export MAX_CONCURRENT_REQUESTS=20
export CUDA_VISIBLE_DEVICES=1
bash eval.sh

# Example 3: run the Python script directly
python evaluate_translator.py \
    --model_path ./outputs/translator_model \
    --data_path data/test_data.jsonl \
    --output_path results/my_evaluation.json \
    --judge_model gpt-4o \
    --max_concurrent_requests 20 \
    --device cuda \
    --batch_size 16 \
    --max_length 1024
```

## Evaluation pipeline

1. **Load the model**: Load the trained translator model and tokenizer.
2. **Load the data**: Load the evaluation data (JSONL format, requires `original` and `alien` fields).
3. **Decode**: Decode the encoded text (`alien`) back to English with the model.
4. **Score**: Compare the decoded text against the reference (`original`).
   - Automatic metrics: BLEU, ROUGE, METEOR
   - LLM-as-a-judge: semantic similarity, structural similarity, overall quality
5. **Save results**: Write the evaluation results and aggregated metrics to a JSON file.

## Evaluation data format

The evaluation data must be in JSONL format. Each line looks like:

```json
{"original": "original text", "alien": "encoded text"}
```

## Result format

Evaluation results are saved in the following format:

```json
{
  "results": [
    {
      "original": "original text",
      "alien": "encoded text",
      "decrypted": "decoded text",
      "evaluation": {
        "bleu": 0.85,
        "rouge1_f": 0.90,
        "rouge2_f": 0.88,
        "rougeL_f": 0.89,
        "meteor": 0.87,
        "llm_semantic": 3,
        "llm_structural": 3,
        "llm_overall": 3
      }
    }
  ],
  "metrics": {
    "bleu_mean": 0.85,
    "rouge1_f_mean": 0.90,
    "llm_overall_mean": 3.0,
    ...
  }
}
```

## Environment variables

- `OPENAI_API_KEY`: Required when using the LLM judge. (Or pass it via `--api_key`.)
- `CUDA_VISIBLE_DEVICES`: Specifies which GPU IDs to use. (e.g. `0`, `1`, `0,1`)

## Notes

- `MODEL_PATH` is required. When using `eval.sh`, set it in the script or pass it as an environment variable.
- `OPENAI_API_KEY` is required when using the LLM judge. Set it as an environment variable or pass it with `--api_key`. Never commit a literal API key.
- When using GPUs, you can choose which GPUs to use with the `CUDA_VISIBLE_DEVICES` environment variable.
