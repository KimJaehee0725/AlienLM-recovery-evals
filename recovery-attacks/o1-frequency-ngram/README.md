# O1 — Frequency / N-gram Token-Recovery Attack

Passive observation attack that tries to recover the AlienLM token-permutation (bijection) purely from token-frequency and n-gram statistics of an observed alien corpus, with **no aligned (alien, original) pairs**.

**Paper reference**: Table 3 (O1, passive observation); Appendix E.2 (frequency matching).

**What it measures**: token-recovery rate obtained by matching alien token-frequency / n-gram statistics against public-corpus statistics, with no aligned pairs available to the attacker. It reports both token-level accuracy (fraction of recovered test tokens that match the ground-truth original tokens) and bijection recovery rate (fraction of recovered token-to-token mappings that match the true bijection).

## How it works

1. An observed alien corpus is built by re-encoding public instruction data (Magpie) through the original tokenizer and decoding through the alien tokenizer.
2. Unigram frequencies and n-gram context statistics are collected for both the alien corpus and an independent public reference corpus (Tulu3 / AceReason / SlimOrca).
3. A similarity matrix between the top-k most frequent alien tokens and the top-k most frequent reference tokens is built from a blend of frequency similarity and n-gram context overlap.
4. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) solves for the optimal token-to-token assignment.
5. The recovered mapping is applied to held-out alien test sequences and scored against the ground-truth original sequences. The true bijection (derived by matching identical token strings across the two tokenizer vocabularies) is used only for scoring, never by the attack itself.

## Files

- `ngram_datasets.py` — core module: dataset loaders, bijection extractor, n-gram attack, and evaluation.
- `run_evaluation.py` — runs a single experiment for one reference corpus and one n value.
- `run_all_experiments.sh` — sweeps all reference-corpus × n combinations sequentially.
- `cleanup_processes.sh` — kills hanging `run_evaluation.py` worker processes.

## Inputs

Environment variables:

- `ALIEN_TOKENIZER_PATH` (required) — local path to the AlienLM tokenizer checkpoint. Required by `run_evaluation.py` (also passable via `--alien_tokenizer_path`).
- `HF_DATASETS_CACHE` (optional) — directory used to cache Hugging Face datasets. Falls back to the Hugging Face default when unset.
- `OPENAI_API_KEY` — not used by this experiment.

External resources (downloaded on demand from the Hugging Face Hub):

- Original tokenizer: `meta-llama/Meta-Llama-3-8B-Instruct` (default; overridable via `--org_tokenizer_path`).
- Observed-corpus source: `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered` and `Magpie-Align/Magpie-Reasoning-V1-150K`.
- Reference corpora: `allenai/tulu-3-sft-olmo-2-mixture` (tulu3), `nvidia/AceReason-1.1-SFT` (acereason), `Open-Orca/SlimOrca` (slimorca).

## Run

These commands are for reproduction; do not execute them as part of vendoring.

Single experiment:

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer/checkpoint
export HF_DATASETS_CACHE=/path/to/hf/cache   # optional

python run_evaluation.py \
    --reference_corpus tulu3 \
    --n 2 \
    --train_size 50000 \
    --test_size 50000 \
    --top_k_tokens 10000 \
    --reference_size 50000 \
    --output_dir ./attack_results \
    --num_proc 8 \
    --batch_size 1000
```

Full sweep (all 3 reference corpora × n ∈ {2, 3, 4} = 9 runs):

```bash
export ALIEN_TOKENIZER_PATH=/path/to/alien/tokenizer/checkpoint
NON_INTERACTIVE=1 bash run_all_experiments.sh
```

Adjust the per-run timeout (default 24 hours):

```bash
TIMEOUT_HOURS=48 NON_INTERACTIVE=1 bash run_all_experiments.sh
```

Clean up hanging worker processes:

```bash
bash cleanup_processes.sh            # kill run_evaluation.py workers
bash cleanup_processes.sh --aggressive   # also prompt for datasets/transformers workers
```

## Outputs

Results are written under `attack_results/` (a git-ignored directory). Each run creates a subdirectory named
`{reference_corpus}_n{n}_train{train_size}_test{test_size}_topk{top_k_tokens}[_ref{reference_size}]/` containing:

- `summary.json` — token accuracy, bijection recovery rate, and corpus/mapping sizes.
- `recovered_mapping.json` — the recovered alien-to-original token mapping.

`run_all_experiments.sh` additionally writes a per-run log to `logs/` (also git-ignored).

## Notes

- Dependencies: `transformers`, `datasets`, `numpy`, `scipy`.
- No GPU is required; the attack is CPU- and memory-bound. The bash sweep defaults to `--num_proc 64`; reduce it if memory is limited.
- `OPENAI_API_KEY` is not needed for this experiment.
- Large reference corpora can consume substantial memory; cap them with `--reference_size` and lower `--num_proc` / `--batch_size` if needed.
- The true bijection is used only to score recovery; the attack itself observes no aligned pairs.
