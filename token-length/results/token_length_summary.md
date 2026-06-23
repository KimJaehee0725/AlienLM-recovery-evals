# Token Length Summary

- Sample size: `10000`
- Seed: `42`
- Sample source counts: `{'magpie_pro': 6594, 'magpie_reasoning': 3406}`
- Length definition: identical plain text per sample with `add_special_tokens=False`

| tokenizer | avg tokens/sample | median | stdev | min | max | total tokens | delta vs min avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma2_9b_it | 644.6784 | 645.0 | 290.1384 | 40 | 11796 | 6446784 | +4.35% |
| llama3_8b_instruct | 617.8083 | 627.0 | 247.5365 | 38 | 4087 | 6178083 | +0.00% |
| qwen25_7b_instruct | 626.7329 | 632.0 | 282.1945 | 39 | 11809 | 6267329 | +1.44% |

Baseline shortest average tokenizer: `llama3_8b_instruct` (617.8083 tokens/sample)

## By Source

| tokenizer | source | avg tokens/sample | total tokens |
| --- | --- | ---: | ---: |
| gemma2_9b_it | magpie_pro | 692.7348 | 4567893 |
| gemma2_9b_it | magpie_reasoning | 551.6415 | 1878891 |
| llama3_8b_instruct | magpie_pro | 675.9792 | 4457407 |
| llama3_8b_instruct | magpie_reasoning | 505.1897 | 1720676 |
| qwen25_7b_instruct | magpie_pro | 681.4630 | 4493567 |
| qwen25_7b_instruct | magpie_reasoning | 520.7757 | 1773762 |
