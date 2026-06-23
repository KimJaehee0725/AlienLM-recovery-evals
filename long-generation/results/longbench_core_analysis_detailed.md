# LongBench Core Analysis

Tasks: `longbench_gov_report_e` and `longbench_qasper_e`.
Metric types differ across tasks (`ROUGE` vs `QA F1`), so the safest comparison is original vs AlienLM within each task and backbone.

## Raw Scores

| Backbone | Variant | GovReport ROUGE | Qasper F1 | Result File |
| --- | --- | ---: | ---: | --- |
| Llama-3-8B-Instruct | original | 0.2811 ± 0.0035 | 0.1275 ± 0.0113 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/llama_original/longbench_core/0-shot/__workspace__CACHE__MODELS__models--meta-llama--Meta-Llama-3-8B-Instruct__snapshots__8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/results_2026-03-29T10-15-47.104387.json |
| Llama-3-8B-Instruct | alien | 0.2282 ± 0.0049 | 0.2482 ± 0.0183 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/llama_alien/longbench_core/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/results_2026-03-29T10-23-28.258763.json |
| Qwen2.5-7B-Instruct | original | 0.3170 ± 0.0029 | 0.1077 ± 0.0083 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/qwen_original/longbench_core/0-shot/__workspace__CACHE__MODELS__models--Qwen--Qwen2.5-7B-Instruct__snapshots__a09a35458c702b33eeacc393d103063234e8bc28/results_2026-03-29T10-33-17.027994.json |
| Qwen2.5-7B-Instruct | alien | 0.1556 ± 0.0037 | 0.0611 ± 0.0043 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/qwen_alien/longbench_core/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/results_2026-03-29T10-43-11.945820.json |
| Gemma-2-9b-it | original | 0.2561 ± 0.0032 | 0.1689 ± 0.0167 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/gemma_original/longbench_core/0-shot/__workspace__CACHE__MODELS__models--google--gemma-2-9b-it__snapshots__11c9b309abf73637e4b6f9a3fa1e92e615547819/results_2026-03-29T10-56-01.532129.json |
| Gemma-2-9b-it | alien | 0.1882 ± 0.0049 | 0.0795 ± 0.0090 | /workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/gemma_alien/longbench_core/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen/results_2026-03-29T11-10-34.914111.json |

## Original vs AlienLM Delta

| Backbone | GovReport delta | GovReport relative | Qasper delta | Qasper relative |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B-Instruct | -0.0529 | -18.82% | +0.1207 | +94.61% |
| Qwen2.5-7B-Instruct | -0.1613 | -50.90% | -0.0466 | -43.25% |
| Gemma-2-9b-it | -0.0680 | -26.54% | -0.0894 | -52.92% |

## Interpretation

- `gov_report_e` drops for all three backbones after AlienLM adaptation.
- `qasper_e` drops for Qwen and Gemma, but Llama improves noticeably on this task.
- So the long-generation story is not simply that AlienLM always fails on free-form tasks. A safer statement is that performance remains task-dependent, but the degradation clearly extends beyond MCQA-only settings.
