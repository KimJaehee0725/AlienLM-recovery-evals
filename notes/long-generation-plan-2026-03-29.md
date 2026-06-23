# Long Generation Evaluation Plan

## Why this experiment

- Reviewer concern: current evaluation is dominated by MCQA, with limited evidence on free-form or long generation.
- Existing default evaluation script only runs `mmlu`, `arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `truthfulqa_mc1`, and `gsm8k_cot`.
- The local `lm-evaluation-harness` already supports both `truthfulqa_gen` and `LongBench`.

## Existing assets

- Existing free-form generation results already exist under `/workspace/data2/jaehee/AlienLM/outputs/*/truthfulqa_gen/...`.
- This means we can immediately add one non-MCQA generation benchmark without rerunning training.
- For new long-generation evidence, the repo already contains LongBench task configs under `lm-evaluation-harness/lm_eval/tasks/longbench/`.

## Recommended protocol

### Stage 1: Recover existing free-form generation evidence

- Use `truthfulqa_gen` as the first rebuttal result.
- Compare original vs AlienLM checkpoints for the same backbone when available.
- If one original baseline is missing, rerun only that missing `truthfulqa_gen` job rather than the full benchmark suite.

### Stage 2: Add one small LongBench subset

- Do not run the full 21-task LongBench suite.
- Prefer generation-centric tasks that directly answer the reviewer concern:
  - `longbench_qmsum`
  - `longbench_gov_report_e`
- Optional third task if time permits:
  - `longbench_multi_news_e`
  - or `longbench_repobench-p_e` if we want a code-oriented long generation result

## Why these tasks

- `qmsum`: query-based meeting summarization, clearly generative.
- `gov_report_e`: long-document summarization, clearly generative, and the `_e` variant reports by length bucket.
- `multi_news_e`: multi-document summarization, also clearly generative.
- `repobench-p_e`: code completion over long context, useful only if we specifically want a code-generation angle.

## Why use LongBench-E

- LongBench-E reports separate scores for `0-4k`, `4-8k`, and `8k+` input ranges.
- This is useful for AlienLM because alienization can expand tokenized prompt length and change the effective context budget.
- If long-generation performance drops, the length buckets help distinguish general degradation from context-expansion effects.

## Minimal rebuttal package

- Existing `truthfulqa_gen`
- New `longbench_qmsum`
- New `longbench_gov_report_e`

This is the shortest defensible package.

## Model priority

- First priority: `Llama3-8B-Instruct` original vs `Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2`
- Second priority if time remains: `Qwen2.5-7B-Instruct` original vs AlienLM
- Third priority: `Gemma2-9b-it`

## Recommended analysis to include

- Main table:
  - backbone
  - task
  - original score
  - AlienLM score
  - relative retention
- Length analysis:
  - original vs alien tokenized prompt length on the exact evaluation prompts
  - proportion of prompts crossing `4k` or `8k` thresholds after alienization
- Short qualitative appendix:
  - 2 to 4 example summaries or long-form outputs
  - one success case and one failure case

## What not to do

- Do not use retrieval-heavy LongBench tasks as the main rebuttal evidence for "long generation".
- Do not run the entire LongBench suite unless compute is unexpectedly free.
- Do not mix many backbones and many tasks at once before confirming one backbone runs correctly.

## Practical recommendation

If compute is tight, start with:

- `truthfulqa_gen` from existing logs
- new `longbench_qmsum`
- new `longbench_gov_report_e`
- Llama only

If these run cleanly and the result looks usable, expand to Qwen next.
