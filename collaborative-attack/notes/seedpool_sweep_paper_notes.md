# Mixed-Seed Collaborative Attack: Paper Notes

## Setup

- Goal: test whether pooled plaintext leakage across multiple users remains equally effective when users share the same alien seed versus when they use different seeds.
- Corpus: Magpie reasoning subset, `train=10k`, `test=1k`, `reference=tulu3 10k`, `n=3`, `top_k=1000`.
- Attack:
  - attacker receives `k` plaintext alien-text pairs
  - pairs are tokenized with the original tokenizer on both sides
  - only high-consensus token correspondences are kept (`min_confidence=0.95`, `min_occurrences=3`)
  - remaining tokens are recovered with the same `n-gram + Hungarian matching` solver
- Compared settings:
  - `shared seed`: all users use the same alien tokenizer seed
  - `mixed seeds`: users are pooled across three different seeds (`42/43/44`)

## Main Numbers

| k known pairs | shared token acc | mixed token acc | delta |
| --- | ---: | ---: | ---: |
| 1000 | 6.05% | 4.40% | -1.65pp |
| 5000 | 4.77% | 3.19% | -1.58pp |
| 10000 | 2.15% | 2.56% | +0.42pp |

Known-only coverage shows the same pattern at moderate leakage budgets:

| k known pairs | shared coverage | mixed coverage | delta |
| --- | ---: | ---: | ---: |
| 1000 | 35.89% | 26.10% | -9.79pp |
| 5000 | 29.31% | 21.25% | -8.06pp |
| 10000 | 13.75% | 18.91% | +5.16pp |

## Safe Interpretation

- At `k=1000` and `k=5000`, collaborative leakage is clearly stronger when users share the same seed.
- When users are pooled across different seeds, both direct recovered coverage and final token reconstruction accuracy drop.
- At `k=10000`, the gap largely closes and slightly reverses, so the result should be framed conservatively: distinct seeds reduce collusive leakage at low-to-moderate leakage budgets, but they do not eliminate the risk under very large leakage.

## Suggested Rebuttal Wording

We additionally simulated a collaborative known-plaintext attack in which the adversary pools leaked pairs either from users sharing the same alien seed or from users using three different seeds. Using the same `n`-gram plus Hungarian recovery pipeline, mixed-seed pooling was consistently weaker at moderate leakage budgets: token reconstruction dropped from `6.05%` to `4.40%` at `k=1k` and from `4.77%` to `3.19%` at `k=5k`. At a very large leakage budget (`k=10k`), the gap largely disappeared, so our conclusion is not that multi-tenant seeds eliminate collusive risk, but that they materially damp it unless leakage becomes very large.
