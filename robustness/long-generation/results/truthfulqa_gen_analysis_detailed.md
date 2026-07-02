# TruthfulQA Generation Analysis

## Scope

This note summarizes a free-form generation check on `truthfulqa_gen` for three backbones:

- `Llama3-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Gemma2-9B-it`

Each backbone is compared against its corresponding `AlienLM` variant. The newly added `Llama3-8B-Instruct` original baseline was evaluated on March 29, 2026 with `vllm` on a single `A100 80GB`, and its raw result file is stored under `/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/...`.

Primary source tables:

- [truthfulqa_gen_length_summary.md](/workspace/codes/AlienLMv2/icml2026-rebuttal/long-generation/results/truthfulqa_gen_length_summary.md)
- [truthfulqa_gen_length_summary.json](/workspace/codes/AlienLMv2/icml2026-rebuttal/long-generation/results/truthfulqa_gen_length_summary.json)

## Important Caveat

The `log_samples` files used here store the task-visible prompts and outputs seen by `lm_eval`. They do not directly expose the client-side alienized wire-format strings.

This matters for interpretation:

- `input length` here should be read as the length of the evaluation prompt under each model tokenizer, not as the transmitted encrypted prompt length.
- `output length` here should be read as the task-visible generated answer length, not necessarily the raw encrypted response length before client-side decoding.
- For API cost or wire-format token inflation, the tokenizer-length experiments in `icml2026-rebuttal/token-length/` remain the correct reference.

## Headline Result

TruthfulQA generation provides a non-MCQA, free-form generation view, and all three backbones degrade under AlienLM.

Average absolute drop across the three backbones:

- `BLEU Acc`: `-0.0889`
- `ROUGE-L Acc`: `-0.0849`

Per-backbone results are below.

## Backbone-Level Comparison

| Backbone | BLEU Acc Orig | BLEU Acc Alien | Delta | Rel. Delta | ROUGE-L Acc Orig | ROUGE-L Acc Alien | Delta | Rel. Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3-8B-Instruct | 0.4627 | 0.3929 | -0.0698 | -15.08% | 0.4749 | 0.4039 | -0.0710 | -14.95% |
| Qwen2.5-7B-Instruct | 0.5055 | 0.3819 | -0.1236 | -24.46% | 0.5018 | 0.3929 | -0.1089 | -21.71% |
| Gemma2-9B-it | 0.4896 | 0.4162 | -0.0734 | -15.00% | 0.5153 | 0.4406 | -0.0747 | -14.49% |

## Length Behavior

### Input Length

Input token and character statistics are identical within each backbone:

- `Llama3-8B-Instruct`: `157.81` mean input tokens
- `Qwen2.5-7B-Instruct`: `166.89` mean input tokens
- `Gemma2-9B-it`: `177.89` mean input tokens

This is consistent with the caveat above: the saved sample prompts are task-visible prompts, so this table should not be used to argue about client-side encrypted prompt inflation.

### Output Length

Output lengths do change, but not in a uniform direction.

| Backbone | Mean Output Tok Orig | Mean Output Tok Alien | Delta | Rel. Delta | Output Tok Var Orig | Output Tok Var Alien | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3-8B-Instruct | 21.54 | 22.33 | +0.79 | +3.67% | 290.81 | 358.18 | +67.37 |
| Qwen2.5-7B-Instruct | 77.82 | 33.72 | -44.10 | -56.67% | 2235.34 | 2662.09 | +426.75 |
| Gemma2-9B-it | 22.83 | 26.51 | +3.69 | +16.15% | 134.27 | 247.99 | +113.72 |

Character-level output length shows the same pattern:

- `Llama`: `89.17 -> 94.22` chars, `+5.67%`
- `Qwen`: `379.97 -> 131.39` chars, `-65.42%`
- `Gemma`: `100.38 -> 112.21` chars, `+11.79%`

## Interpretation

### 1. AlienLM degrades free-form generation, not only MCQA

This experiment is useful precisely because `truthfulqa_gen` is not a multiple-choice benchmark. The degradation appears across all three backbones on both `BLEU Acc` and `ROUGE-L Acc`, so the performance drop is not confined to answer selection tasks.

### 2. The degradation is not explained by a single length effect

If the main failure mode were simply "AlienLM makes generations longer" or "AlienLM truncates generations," we would expect a consistent directional change across models. That is not what we observe:

- `Llama` and `Gemma` produce slightly longer outputs under AlienLM.
- `Qwen` produces much shorter outputs under AlienLM.
- All three still lose accuracy.

This pattern suggests that the main issue is not a universal output-length inflation effect, but a broader degradation in free-form answer quality and stability.

### 3. Qwen shows the strongest generation-style shift

`Qwen2.5-7B-Instruct` has the largest score drop and the largest shortening of generated answers:

- `BLEU Acc`: `0.5055 -> 0.3819`
- `ROUGE-L Acc`: `0.5018 -> 0.3929`
- Mean output tokens: `77.82 -> 33.72`

This indicates that, at least for Qwen, AlienLM is associated with a more conservative or compressed generation style on TruthfulQA.

### 4. Llama and Gemma lose quality without obvious output collapse

For `Llama` and `Gemma`, AlienLM outputs are slightly longer on average and more variable, yet accuracy still drops by about `15%` relative.

That is a useful rebuttal point because it shows the quality drop is not reducible to a trivial "the model just stopped answering" story. Instead, the model is still producing normal-length answers, but they less often align with the truthful reference set.

## Suggested Rebuttal Reading

The safest claim supported by this table is:

- AlienLM does not only affect multiple-choice tasks; it also degrades free-form generation quality on TruthfulQA generation across three backbones.
- The effect is not explained by a single uniform change in answer length.
- The observed generation behavior is backbone-dependent, with Qwen showing a stronger shortening effect, while Llama and Gemma show quality degradation despite similar or slightly longer answer lengths.

What this table does **not** support on its own:

- a statement about encrypted wire-format prompt or response length
- a statement that length inflation is the dominant cause of quality loss
- a claim about long-context behavior beyond this benchmark
