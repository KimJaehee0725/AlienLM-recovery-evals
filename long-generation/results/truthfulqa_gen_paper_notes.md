# TruthfulQA Generation Notes For Rebuttal

`TruthfulQA generation` gives a non-MCQA check of free-form generation quality. Across all three backbones, `AlienLM` degrades generation performance relative to the original models: `Llama` drops from `0.4627` to `0.3929` in `BLEU Acc` and from `0.4749` to `0.4039` in `ROUGE-L Acc`, `Qwen` drops from `0.5055` to `0.3819` and from `0.5018` to `0.3929`, and `Gemma` drops from `0.4896` to `0.4162` and from `0.5153` to `0.4406`. This shows that the performance gap is not limited to multiple-choice evaluation.

The output-length patterns are not uniform across models, which argues against a single trivial length-based explanation. Under `AlienLM`, `Qwen` produces much shorter answers on average (`77.82 -> 33.72` tokens), whereas `Llama` and `Gemma` produce slightly longer answers (`21.54 -> 22.33` and `22.83 -> 26.51`). Since all three models still lose accuracy, the degradation is better characterized as a broader reduction in free-form answer quality and stability rather than a simple increase or decrease in output length.

Interpretation caveat: the prompt/output lengths reported in this table come from `lm_eval` sample logs, so they reflect task-visible prompts and decoded generations, not the raw encrypted wire-format strings used by the client-side AlienLM translation layer.
