# Original vs Alien Token Length Summary

Definition:
- `original_plain`: original plain text tokenized by the original tokenizer.
- `alien_on_plain`: the same original plain text tokenized directly by the alien tokenizer.
- `alien_on_alienized`: the alienized text obtained by `alien_tokenizer.decode(original_ids)` and then tokenized by the alien tokenizer.

Interpretation:
- `alien_on_alienized` measures the alien tokenizer on alienized text directly.
- `original_on_alienized` follows the actual single-alien wrapper path used in Axolotl: `plain -> alien_text -> original/base tokenizer encode`.
- `alien_on_plain` is included only as a tokenizer-drift reference on identical raw text.

- Sample size: `10000`

## llama3_8b_instruct

- Original tokenizer: `/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- Alien tokenizer: `/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2`
- Note: Alien tokenizer built from the Qwen-family vocabulary permutation.
- Alien-tokenizer roundtrip length match rate: `0.49%`
- Alien-tokenizer roundtrip exact ID match rate: `0.00%`
- Wrapper-path length match rate: `0.49%`
- Wrapper-path exact ID match rate: `0.00%`

| measure | avg tokens/sample | median | stdev | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original_plain` | 617.8083 | 627.0 | 247.5365 | 38 | 4087 | 6178083 |
| `alien_on_plain` | 617.8083 | 627.0 | 247.5365 | 38 | 4087 | 6178083 |
| `alien_on_alienized` | 645.5669 | 649.0 | 264.3197 | 42 | 6407 | 6455669 |
| `original_on_alienized` | 645.5669 | 649.0 | 264.3197 | 42 | 6407 | 6455669 |

Deltas:
- `alien_on_plain` vs `original_plain`: avg diff `+0.0000`, relative diff `+0.00%`, total diff `+0`
- `alien_on_alienized` vs `original_plain`: avg diff `+27.7586`, relative diff `+4.49%`, total diff `+277586`
- `original_on_alienized` vs `original_plain`: avg diff `+27.7586`, relative diff `+4.49%`, total diff `+277586`

## qwen25_7b_instruct

- Original tokenizer: `/workspace/CACHE/MODELS/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`
- Alien tokenizer: `/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama`
- Note: Alien tokenizer built from the Llama-family vocabulary permutation.
- Alien-tokenizer roundtrip length match rate: `8.02%`
- Alien-tokenizer roundtrip exact ID match rate: `0.75%`
- Wrapper-path length match rate: `8.02%`
- Wrapper-path exact ID match rate: `0.00%`

| measure | avg tokens/sample | median | stdev | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original_plain` | 626.7329 | 632.0 | 282.1945 | 39 | 11809 | 6267329 |
| `alien_on_plain` | 626.7329 | 632.0 | 282.1945 | 39 | 11809 | 6267329 |
| `alien_on_alienized` | 631.7696 | 638.0 | 282.9941 | 39 | 11810 | 6317696 |
| `original_on_alienized` | 631.7696 | 638.0 | 282.9941 | 39 | 11810 | 6317696 |

Deltas:
- `alien_on_plain` vs `original_plain`: avg diff `+0.0000`, relative diff `+0.00%`, total diff `+0`
- `alien_on_alienized` vs `original_plain`: avg diff `+5.0367`, relative diff `+0.80%`, total diff `+50367`
- `original_on_alienized` vs `original_plain`: avg diff `+5.0367`, relative diff `+0.80%`, total diff `+50367`

## gemma2_9b_it

- Original tokenizer: `/workspace/CACHE/MODELS/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819`
- Alien tokenizer: `/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen`
- Note: Alien tokenizer built from the Qwen-family vocabulary permutation.
- Alien-tokenizer roundtrip length match rate: `0.73%`
- Alien-tokenizer roundtrip exact ID match rate: `0.00%`
- Wrapper-path length match rate: `0.73%`
- Wrapper-path exact ID match rate: `0.00%`

| measure | avg tokens/sample | median | stdev | min | max | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original_plain` | 644.6784 | 645.0 | 290.1384 | 40 | 11796 | 6446784 |
| `alien_on_plain` | 644.6784 | 645.0 | 290.1384 | 40 | 11796 | 6446784 |
| `alien_on_alienized` | 667.2477 | 668.0 | 295.2898 | 41 | 11789 | 6672477 |
| `original_on_alienized` | 667.2477 | 668.0 | 295.2898 | 41 | 11789 | 6672477 |

Deltas:
- `alien_on_plain` vs `original_plain`: avg diff `+0.0000`, relative diff `+0.00%`, total diff `+0`
- `alien_on_alienized` vs `original_plain`: avg diff `+22.5693`, relative diff `+3.50%`, total diff `+225693`
- `original_on_alienized` vs `original_plain`: avg diff `+22.5693`, relative diff `+3.50%`, total diff `+225693`
