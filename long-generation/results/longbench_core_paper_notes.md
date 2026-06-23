# LongBench Core: Paper Notes

## Setup

- Tasks: `gov_report_e` and `qasper_e` from LongBench.
- Models: Llama, Qwen, Gemma original checkpoints and their AlienLM-adapted counterparts.
- Evaluation: `vLLM`, `0-shot`, single-GPU inference.

## Main Numbers

| Backbone | GovReport original | GovReport AlienLM | Qasper original | Qasper AlienLM |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B-Instruct | 0.2811 | 0.2282 | 0.1275 | 0.2482 |
| Qwen2.5-7B-Instruct | 0.3170 | 0.1556 | 0.1077 | 0.0611 |
| Gemma-2-9b-it | 0.2561 | 0.1882 | 0.1689 | 0.0795 |

## Delta Summary

- Llama-3-8B-Instruct: GovReport -0.0529 (-18.82%), Qasper +0.1207 (+94.61%).
- Qwen2.5-7B-Instruct: GovReport -0.1613 (-50.90%), Qasper -0.0466 (-43.25%).
- Gemma-2-9b-it: GovReport -0.0680 (-26.54%), Qasper -0.0894 (-52.92%).

## Suggested Wording

We additionally evaluated two LongBench generation tasks, `gov_report_e` and `qasper_e`, across Llama, Qwen, and Gemma backbones. `gov_report_e` degraded consistently for all three models after AlienLM adaptation, while `qasper_e` degraded for Qwen and Gemma but improved for Llama. This indicates that the performance gap is not confined to multiple-choice benchmarks, but also that free-form generation under AlienLM remains task-dependent rather than uniformly degraded.
