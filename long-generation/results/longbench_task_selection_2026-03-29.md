# LongBench Task Selection Note

## Goal

We need two longer-generation tasks for rebuttal that are compatible with the available backbones:

- `Llama3-8B-Instruct` and its AlienLM variant (`8k` context)
- `Qwen2.5-7B-Instruct` and its AlienLM variant (`32k` context)
- `Gemma2-9B-it` and its AlienLM variant (`8k` context)

## Prompt-Length Check

Prompt lengths were measured with each backbone tokenizer on the task-visible `doc_to_text` prompt.

### qmsum

| Backbone tokenizer | Mean prompt tokens | Max prompt tokens | Over model context |
| --- | ---: | ---: | ---: |
| Llama | 13915.6 | 30437 | 167 / 200 (`83.5%`) over `8192` |
| Qwen | 13946.7 | 30437 | 0 / 200 (`0.0%`) over `32768` |
| Gemma | 14404.6 | 31423 | 172 / 200 (`86.0%`) over `8192` |

### gov_report_e

| Backbone tokenizer | Mean prompt tokens | Max prompt tokens | Over model context |
| --- | ---: | ---: | ---: |
| Llama | 8193.6 | 27720 | 113 / 300 (`37.7%`) over `8192` |
| Qwen | 8440.3 | 28609 | 0 / 300 (`0.0%`) over `32768` |
| Gemma | 8396.4 | 28630 | 119 / 300 (`39.7%`) over `8192` |

### qasper_e

| Backbone tokenizer | Mean prompt tokens | Max prompt tokens | Over model context |
| --- | ---: | ---: | ---: |
| Llama | 6252.2 | 21156 | 32 / 224 (`14.3%`) over `8192` |
| Qwen | 6387.2 | 21926 | 0 / 224 (`0.0%`) over `32768` |
| Gemma | 6357.0 | 21464 | 32 / 224 (`14.3%`) over `8192` |

## Decision

`qmsum` is a poor fit for the `8k` backbones because more than `80%` of prompts exceed the context limit for both `Llama` and `Gemma`. That would make a six-model comparison difficult to interpret.

`gov_report_e` is a long summarization task and `qasper_e` is a long-context free-form QA task. Together they provide a better tradeoff between task diversity and context compatibility for the `8k` models than `qmsum` or `multi_news_e`.

## Recommended Evaluation Set

- `longbench_gov_report_e`
- `longbench_qasper_e`

These are run as the `longbench_core` subset in the scripts under `/workspace/codes/AlienLMv2/icml2026-rebuttal/long-generation/scripts/`.
