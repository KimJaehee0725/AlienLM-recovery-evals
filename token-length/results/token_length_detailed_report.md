# Token Length Detailed Report

## Setup

- Data pool:
  - `Magpie-Align/Magpie-Pro-300K-Filtered`
  - `Magpie-Align/Magpie-Reasoning-V1-150K`
- Pool size: `450,000`
- Sample size: `10,000`
- Sampling: uniform random over the combined pool, `seed=42`
- Sample composition:
  - `magpie_pro`: `6,594`
  - `magpie_reasoning`: `3,406`
- Length definition:
  - identical plain text per sample
  - no chat template
  - `add_special_tokens=False`

## Overall Statistics

| tokenizer | avg tokens/sample | median | stdev | p90 | p95 | p99 | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` | 617.8083 | 627.0 | 247.5365 | 822 | 920 | 1285 | 38 | 4087 | 6,178,083 |
| `qwen25_7b_instruct` | 626.7329 | 632.0 | 282.1945 | 830 | 932 | 1312 | 39 | 11809 | 6,267,329 |
| `gemma2_9b_it` | 644.6784 | 645.0 | 290.1384 | 853 | 962 | 1392 | 40 | 11796 | 6,446,784 |

## Pairwise Comparison

Baseline shortest tokenizer on this 10k sample is `llama3_8b_instruct`.

| comparison | avg token difference | relative difference | total token difference on 10k |
| --- | ---: | ---: | ---: |
| `qwen25_7b_instruct` vs `llama3_8b_instruct` | +8.9246 | +1.44% | +89,246 |
| `gemma2_9b_it` vs `llama3_8b_instruct` | +26.8701 | +4.35% | +268,701 |
| `gemma2_9b_it` vs `qwen25_7b_instruct` | +17.9455 | +2.86% | +179,455 |

Interpretation:

- `Llama` and `Qwen` are very close in token count on the same 10k sample.
- `Gemma` is consistently longer, but the gap is still single-digit percent relative to `Llama`.

## By Source Dataset

### Magpie-Pro

| tokenizer | avg tokens/sample | median | stdev | p90 | p95 | p99 | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` | 675.9792 | 652.5 | 118.5260 | 810 | 884 | 1080 | 411 | 2065 | 4,457,407 |
| `qwen25_7b_instruct` | 681.4630 | 657.0 | 124.2987 | 817 | 889 | 1106 | 411 | 2259 | 4,493,567 |
| `gemma2_9b_it` | 692.7348 | 669.0 | 128.0487 | 828 | 910 | 1132 | 477 | 2653 | 4,567,893 |

Pairwise deltas on `magpie_pro`:

- `qwen25_7b_instruct` vs `llama3_8b_instruct`: `+5.4838` tokens/sample, `+0.81%`
- `gemma2_9b_it` vs `llama3_8b_instruct`: `+16.7555` tokens/sample, `+2.48%`
- `gemma2_9b_it` vs `qwen25_7b_instruct`: `+11.2718` tokens/sample, `+1.65%`

### Magpie-Reasoning

| tokenizer | avg tokens/sample | median | stdev | p90 | p95 | p99 | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` | 505.1897 | 448.0 | 365.3346 | 859 | 1034 | 1507 | 38 | 4087 | 1,720,676 |
| `qwen25_7b_instruct` | 520.7757 | 456.0 | 432.2811 | 874 | 1050 | 1562 | 39 | 11809 | 1,773,762 |
| `gemma2_9b_it` | 551.6415 | 486.0 | 449.7583 | 928 | 1129 | 1751 | 40 | 11796 | 1,878,891 |

Pairwise deltas on `magpie_reasoning`:

- `qwen25_7b_instruct` vs `llama3_8b_instruct`: `+15.5860` tokens/sample, `+3.09%`
- `gemma2_9b_it` vs `llama3_8b_instruct`: `+46.4518` tokens/sample, `+9.19%`
- `gemma2_9b_it` vs `qwen25_7b_instruct`: `+30.8658` tokens/sample, `+5.93%`

## Main Takeaway

On the same 10k sample from the 450k Magpie training pool, tokenizer length differences are present but modest overall:

- `Llama` is shortest.
- `Qwen` is only about `1.44%` longer than `Llama`.
- `Gemma` is about `4.35%` longer than `Llama`.

This supports the rebuttal point that modern instruction-tuned LLM tokenizers produce broadly comparable token counts on the same training data, especially for `Llama` and `Qwen`.
