# O2 — MT / NLLB Translator Recovery

Machine-translation style recovery baseline that trains an NLLB translator from alienized text back to natural text.

## Inputs

- `ALIEN_TOKENIZER_PATH`: alien tokenizer checkpoint or HF id.
- `ORIGINAL_TOKENIZER_PATH`: original tokenizer, default `meta-llama/Meta-Llama-3-8B-Instruct`.
- Optional Hugging Face cache: `HF_HOME`, `HF_DATASETS_CACHE`.

## Train

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer
export ORIGINAL_TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct

bash scripts/train.sh   --alien_tokenizer_path "$ALIEN_TOKENIZER_PATH"   --original_tokenizer_path "$ORIGINAL_TOKENIZER_PATH"   --model_name facebook/nllb-200-3.3B   --output_dir ./outputs/translator_model   --num_gpus 1   --num_samples 10000   --report_to none
```

## Evaluate

```bash
python eval/evaluate_translator.py --help
```

Outputs go to ignored `outputs/` and `logs/` directories.
