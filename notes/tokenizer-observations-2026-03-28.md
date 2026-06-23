# Tokenizer Observations for Rebuttal

Date: `2026-03-28`

This note summarizes two tokenizer-side observations used for the ICML 2026 rebuttal:

1. token count differences on the same Magpie data
2. vocabulary overlap across Llama, Qwen, and Gemma tokenizers

Detailed raw outputs are stored in:

- token-length summary: `/workspace/codes/AlienLMv2/icml2026-rebuttal/token-length/results/token_length_summary.md`
- token-length detailed report: `/workspace/codes/AlienLMv2/icml2026-rebuttal/token-length/results/token_length_detailed_report.md`
- vocab overlap summary: `/workspace/codes/AlienLMv2/icml2026-rebuttal/vocab-overlap/results/vocab_overlap_summary.md`

## 1. Token Count Comparison on the Same 10k Sample

Setup:

- Data pool: `Magpie-Pro-300K-Filtered + Magpie-Reasoning-V1-150K`
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

Overall results:

| tokenizer | avg tokens/sample | median | stdev | total tokens |
| --- | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` | `617.8083` | `627.0` | `247.5365` | `6,178,083` |
| `qwen25_7b_instruct` | `626.7329` | `632.0` | `282.1945` | `6,267,329` |
| `gemma2_9b_it` | `644.6784` | `645.0` | `290.1384` | `6,446,784` |

Pairwise differences:

- `Qwen vs Llama`: `+8.9246` tokens/sample, `+1.44%`
- `Gemma vs Llama`: `+26.8701` tokens/sample, `+4.35%`
- `Gemma vs Qwen`: `+17.9455` tokens/sample, `+2.86%`

By source:

| tokenizer | source | avg tokens/sample |
| --- | --- | ---: |
| `llama3_8b_instruct` | `magpie_pro` | `675.9792` |
| `qwen25_7b_instruct` | `magpie_pro` | `681.4630` |
| `gemma2_9b_it` | `magpie_pro` | `692.7348` |
| `llama3_8b_instruct` | `magpie_reasoning` | `505.1897` |
| `qwen25_7b_instruct` | `magpie_reasoning` | `520.7757` |
| `gemma2_9b_it` | `magpie_reasoning` | `551.6415` |

Short interpretation:

- `Llama` and `Qwen` produce very similar token counts on the same 10k Magpie sample.
- Even `Gemma`, which is the longest among the three, is only `4.35%` longer than `Llama` on average.
- This supports the claim that modern instruction-tuned tokenizers yield broadly comparable token counts on the same data, especially for `Llama` and `Qwen`.

## 2. Vocabulary Overlap

Definition:

- Base unit: exact token string identity from `tokenizer.get_vocab().keys()`
- Main comparison: excluding special tokens

Tokenizer sizes:

| tokenizer | vocab size incl. special | special token count | vocab size excl. special |
| --- | ---: | ---: | ---: |
| `llama3_8b_instruct` | `128,256` | `2` | `128,254` |
| `qwen25_7b_instruct` | `151,665` | `14` | `151,651` |
| `gemma2_9b_it` | `256,000` | `6` | `255,994` |

Pairwise overlap excluding special tokens:

| pair | intersection | overlap vs A | overlap vs B | jaccard |
| --- | ---: | ---: | ---: | ---: |
| `Llama vs Qwen` | `109,566` | `85.43%` of Llama | `72.25%` of Qwen | `64.32%` |
| `Llama vs Gemma` | `25,632` | `19.99%` of Llama | `10.01%` of Gemma | `7.15%` |
| `Qwen vs Gemma` | `25,138` | `16.58%` of Qwen | `9.82%` of Gemma | `6.57%` |

Three-way overlap excluding special tokens:

- common token strings across all three tokenizers: `25,119`
- overlap vs Llama: `19.59%`
- overlap vs Qwen: `16.56%`
- overlap vs Gemma: `9.81%`

Short interpretation:

- `Llama` and `Qwen` share a large fraction of their vocabularies by exact token string identity.
- `Gemma` is clearly more distinct, but there is still a non-trivial shared core of common token strings.
- Together with the token-count measurement above, this supports the rebuttal argument that `Llama` and `Qwen` are tokenizer-wise fairly close in practice.

## Rebuttal-Oriented Takeaway

A concise way to use these observations in the rebuttal is:

> We measured tokenizer behavior on a 10k uniform sample from the 450k Magpie training pool. On identical plain text inputs, Qwen is only 1.44% longer than Llama on average, while Gemma is 4.35% longer. In addition, Llama and Qwen share 109,566 exact token strings, corresponding to 85.43% of Llama's vocabulary and 72.25% of Qwen's vocabulary, which helps explain why their effective token counts are also very similar in practice.
