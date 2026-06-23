# Mixed-Seed Collaborative Attack Sweep

Comparison between pooled leakage under a shared seed and pooled leakage across three mixed seeds.

| k known pairs | setting | aligned pairs | known mappings | known-only coverage | known-only accuracy | full attack token acc | recovered mappings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | shared | 34/1000 | 371 | 35.89% | 6.04% | 6.05% | 1164 |
| 1000 | mixed | 41/1000 | 531 | 26.10% | 4.39% | 4.40% | 1279 |
| 5000 | shared | 183/5000 | 1107 | 29.31% | 4.75% | 4.77% | 1783 |
| 5000 | mixed | 190/5000 | 1218 | 21.25% | 3.11% | 3.19% | 1941 |
| 10000 | shared | 368/10000 | 1525 | 13.75% | 2.11% | 2.15% | 2253 |
| 10000 | mixed | 360/10000 | 1769 | 18.91% | 2.56% | 2.56% | 2526 |

## Delta: Shared vs Mixed

| k known pairs | metric | shared | mixed | delta (mixed - shared) |
| --- | --- | ---: | ---: | ---: |
| 1000 | known mappings | 371 | 531 | +160 |
| 1000 | known-only coverage | 35.89% | 26.10% | -9.79pp |
| 1000 | known-only accuracy | 6.04% | 4.39% | -1.65pp |
| 1000 | full attack token acc | 6.05% | 4.40% | -1.65pp |
| 5000 | known mappings | 1107 | 1218 | +111 |
| 5000 | known-only coverage | 29.31% | 21.25% | -8.06pp |
| 5000 | known-only accuracy | 4.75% | 3.11% | -1.64pp |
| 5000 | full attack token acc | 4.77% | 3.19% | -1.58pp |
| 10000 | known mappings | 1525 | 1769 | +244 |
| 10000 | known-only coverage | 13.75% | 18.91% | +5.16pp |
| 10000 | known-only accuracy | 2.11% | 2.56% | +0.44pp |
| 10000 | full attack token acc | 2.15% | 2.56% | +0.42pp |

## Takeaway

Within the same leakage budget, compare shared and mixed row pairs horizontally rather than comparing different k values vertically, because the held-out split changes with k.
At k=1000 and k=5000, mixed-seed pooling is consistently weaker than shared-seed pooling in both known-only coverage and full attack token accuracy.
At k=10000, the gap largely closes and slightly reverses, so the cautious conclusion is that distinct tenant seeds damp collaborative leakage at low-to-moderate budgets, but they do not eliminate risk once leakage becomes very large.
