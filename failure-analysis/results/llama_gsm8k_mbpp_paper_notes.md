# Paper Notes: Llama Failure Analysis on GSM8K and MBPP

## Main Claim

For Llama, the large drop on math and code is better explained by exact symbolic surface-form sensitivity than by prompt-length growth.

## Evidence

- GSM8K drops from 73.5% to 50.8% (-22.7pp).
- MBPP drops from 56.8% to 23.0% (-33.8pp).
- In both GSM8K and MBPP, prompt token counts are preserved exactly between the original prompt and the alien prompt when each is measured in its own tokenizer space.
- GSM8K has 371 `original-correct -> alien-wrong` cases; these cases are longer and numerically denser than the `both-correct` subset.
- MBPP has 181 `original-pass -> alien-fail` cases; among them, syntax errors appear in 54 cases, undefined identifiers in 53 cases, and function-name mismatch in 20 cases.

## Interpretation

- On GSM8K, AlienLM often preserves the broad plan but introduces a small semantic drift in the final computation: wrong aggregation, extra normalization, or slightly incorrect equation setup.
- On MBPP, the same kind of surface-form instability is much more damaging because code execution depends on exact syntax, exact identifiers, and exact API names.
- This suggests that AlienLM is relatively compatible with tasks that tolerate paraphrase, but substantially weaker on tasks that require exact symbolic fidelity.

## Candidate Rebuttal Wording

Our qualitative analysis suggests that the degradation on GSM8K and MBPP is not primarily due to longer prompts: for Llama, the original and alien prompts have exactly matched token counts under their respective tokenizers. Instead, the main issue is exact symbolic fidelity. On GSM8K, AlienLM failures typically arise from small semantic drifts in multi-step arithmetic, such as incorrect aggregation or an unnecessary normalization step. On MBPP, failures are more severe because code is brittle to syntax, identifier names, and boundary conditions; many alien outputs become uncompilable or violate the required function signature despite remaining close in high-level intent.
