# Vocab Overlap Summary

Definition:
- Base unit is exact token string identity from `tokenizer.get_vocab().keys()`.
- Main comparison excludes special tokens.
- Metrics reported: `intersection`, `overlap_vs_a`, `overlap_vs_b`, and `jaccard`.

## Tokenizer Sizes

| tokenizer | vocab size incl. special | special token count | vocab size excl. special |
| --- | ---: | ---: | ---: |
| `llama3_8b_instruct` | 128256 | 2 | 128254 |
| `qwen25_7b_instruct` | 151665 | 14 | 151651 |
| `gemma2_9b_it` | 256000 | 6 | 255994 |

## Pairwise Overlap Excluding Special Tokens

| pair | intersection | overlap vs A | overlap vs B | jaccard |
| --- | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` vs `qwen25_7b_instruct` | 109566 | 85.43% | 72.25% | 64.32% |
| `llama3_8b_instruct` vs `gemma2_9b_it` | 25632 | 19.99% | 10.01% | 7.15% |
| `qwen25_7b_instruct` vs `gemma2_9b_it` | 25138 | 16.58% | 9.82% | 6.57% |

## Pairwise Overlap Including Special Tokens

| pair | intersection | overlap vs A | overlap vs B | jaccard |
| --- | ---: | ---: | ---: | ---: |
| `llama3_8b_instruct` vs `qwen25_7b_instruct` | 109566 | 85.43% | 72.24% | 64.32% |
| `llama3_8b_instruct` vs `gemma2_9b_it` | 25632 | 19.99% | 10.01% | 7.15% |
| `qwen25_7b_instruct` vs `gemma2_9b_it` | 25139 | 16.58% | 9.82% | 6.57% |

## Three-Way Overlap

### Excluding Special Tokens

- Intersection size: `25119`
- Union size: `400682`
- Jaccard: `6.27%`
- Overlap vs Llama: `19.59%`
- Overlap vs Qwen: `16.56%`
- Overlap vs Gemma: `9.81%`

### Including Special Tokens

- Intersection size: `25119`
- Union size: `400703`
- Jaccard: `6.27%`
- Overlap vs Llama: `19.59%`
- Overlap vs Qwen: `16.56%`
- Overlap vs Gemma: `9.81%`
