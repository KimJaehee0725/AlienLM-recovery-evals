# Observer Tokenizer Sweep

Llama AlienLM attack performance when the attacker tokenizes the same plaintext/alien-text pairs with different off-the-shelf model tokenizers.

| observer tokenizer | avg plain len | avg alien len | aligned pairs | known mappings | known-only coverage | known-only accuracy | full attack token acc | recovered mappings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma2 | 545.94 | 813.54 | 0/10000 | 0 | 0.00% | 0.00% | 0.08% | 1000 |
| llama3 | 500.51 | 534.47 | 81/10000 | 5 | 9.29% | 1.14% | 1.19% | 1000 |
| mistral | 588.87 | 1284.85 | 0/10000 | 0 | 0.00% | 0.00% | 0.07% | 1000 |
| phi3 | 587.93 | 1288.99 | 0/10000 | 0 | 0.00% | 0.00% | 0.11% | 1000 |
| qwen25 | 515.05 | 637.10 | 5/10000 | 0 | 0.00% | 0.00% | 0.05% | 1000 |

## Interpretation

These numbers are tokenizer-specific token reconstruction rates, so they should be read comparatively rather than as directly identical units across tokenizers.
The main question is whether the attack remains strong when the observer does not know the victim tokenizer and instead tokenizes both sides with another common LLM tokenizer.
