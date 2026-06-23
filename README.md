# AlienLM Attack Evals

This repository collects the compact attack and evaluation experiments that are
kept outside the official `AlienLM` code branches.

Included material is limited to scripts, configs, summaries, and paper notes.
Raw logs, sample-level JSONL dumps, per-task evaluation dumps, checkpoints, and
caches are intentionally excluded.

## External Inputs

Most scripts expect the original experiment machine layout unless overridden:

- official code repo: `../AlienLMv2` or `ALIENLM_CODE_ROOT`
- model cache: `/workspace/CACHE/MODELS`
- dataset cache: `/workspace/data2/jaehee/AlienLM/HF_DATASET`
- model/eval output root: `/workspace/data2/jaehee/AlienLM/outputs`

Treat those paths as defaults captured from the paper/rebuttal environment.
Before rerunning an experiment elsewhere, edit the relevant script/config or set
the corresponding environment variables documented in that script.

## Experiments

| Experiment | Purpose | Run entrypoint | Summary entrypoint | Existing summaries |
| --- | --- | --- | --- | --- |
| `collaborative-attack` | Known-plaintext and observer-tokenizer recovery sweeps | `collaborative-attack/scripts/run_known_pairs_sweep.sh`, `run_minconf_sweep.sh`, `run_observer_tokenizer_sweep.sh`, `run_mixed_seed_pool_sweep.sh` | `collaborative-attack/scripts/summarize_*_results.py` | `collaborative-attack/results/**/summary.json`, `collaborative-attack/notes/*.md` |
| `data-volume` | 50k/150k/full data-volume ablation | `data-volume/scripts/build_data_subsets.py`, `train_subset.sh`, `evaluate_model.sh` | `data-volume/scripts/summarize_main_results.py` | `data-volume/results/data_volume*.md`, `data_volume*.json` |
| `token-length` | Plain text token-length comparison across original/alien tokenizers | `token-length/scripts/measure_token_lengths.py`, `measure_original_vs_alien_lengths.py` | scripts write summaries directly | `token-length/results/*.md`, `*.json` |
| `vocab-overlap` | Exact token-string overlap across tokenizer vocabularies | `vocab-overlap/scripts/measure_vocab_overlap.py` | script writes summary directly | `vocab-overlap/results/vocab_overlap_summary.*` |
| `failure-analysis` | Sample-level GSM8K/MBPP qualitative alignment and tags | `failure-analysis/scripts/analyze_llama_gsm8k_mbpp.py` | script writes summary artifacts | `failure-analysis/results/*`, `status-2026-03-29.md` |
| `long-generation` | TruthfulQA generation length and LongBench core checks | `long-generation/scripts/run_truthfulqa_gen.sh`, `run_longbench_core.sh` | `measure_truthfulqa_lengths.py`, `summarize_longbench_core_results.py` | `long-generation/results/*.md`, `*.json` |

## Notes

- `notes/` contains planning and context notes from the rebuttal workspace.
- `scripts/list_outputs.sh` is a helper for locating external output trees.
- Raw result regeneration should write to ignored `logs/`, `outputs/`, `data/`,
  `data-prepared/`, or external output roots.

## Quick Checks

```bash
python -m compileall -q collaborative-attack data-volume failure-analysis long-generation token-length vocab-overlap
find . -type f -name '*.sh' -print -exec bash -n {} \;
```
