# Data-Volume Main-Suite Comparison

Scores are reported as percentages. Metric choice follows the paper's main classification/math setup:
- `MMLU`, `WinoGrande`, `TruthfulQA MC1`: `acc`
- `ARC-Easy`, `ARC-Challenge`, `HellaSwag`: `acc_norm`
- `GSM8K CoT`: strict-match exact match

| Task | full-0.5epoch | full-1epoch | 50k-ga2 | 150k-ga2 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 35.97 | 39.20 | 35.76 | 39.56 |
| ARC-Easy | 62.54 | 64.23 | 59.72 | 65.03 |
| ARC-Challenge | 41.81 | 45.48 | 39.76 | 44.80 |
| HellaSwag | 59.51 | 60.89 | 52.78 | 59.60 |
| WinoGrande | 59.59 | 62.04 | 56.91 | 60.06 |
| TruthfulQA MC1 | 35.50 | 35.13 | 32.56 | 35.13 |
| GSM8K CoT | 45.11 | 55.65 | 40.56 | 50.34 |

| Macro Average | 48.58 | 51.80 | 45.43 | 50.64 |

## Notes

- `full-0.5epoch`: `/workspace/codes/AlienLMv2/icml2026-rebuttal/data-volume/results/checkpoint-2327`
- `full-1epoch`: `/workspace/codes/AlienLMv2/icml2026-rebuttal/data-volume/results/checkpoint-4654`
- `50k-ga2`: `/workspace/codes/AlienLMv2/icml2026-rebuttal/data-volume/results/Llama3-8B-Instruct-AlienLM-50k-stepmatch4654-qwenv2-seed42-ga2`
- `150k-ga2`: `/workspace/codes/AlienLMv2/icml2026-rebuttal/data-volume/results/Llama3-8B-Instruct-AlienLM-150k-stepmatch4654-qwenv2-seed42-ga2`
