# Collaborative Known-Plaintext Attack

This directory contains rebuttal-only wrappers for the known-plaintext + n-gram
attack under larger paired-data budgets. Existing research code under
`iclr_review/` is left unchanged.

Current sweep:

- Attack code: `iclr_review/attack_scenario/n-gram_w_known-plaintext/run_evaluation.py`
- Fixed setting: `reference_corpus=tulu3`, `n=3`, `train_size=10000`,
  `test_size=1000`, `top_k_tokens=1000`, `reference_size=10000`
- Larger paired-data budget: `k_known_pairs=10000`
- `min_confidence` sweep: `0.5`, `0.05`, `0.01`

Current finding from that sweep:

- `min_confidence=0.5`: `token acc 0.1489%`, `known mappings 3`
- `min_confidence=0.05`: `token acc 0.2496%`, `known mappings 137`
- `min_confidence=0.01`: `token acc 0.3846%`, `known mappings 4288`

This indicates that the collaborative attack is bottlenecked less by the raw
number of paired samples alone and more by how aggressively the extracted token
correspondences are admitted. For reviewer-facing scaling, we therefore run the
known-pairs sweep under the strongest observed setting, `min_confidence=0.01`.

Known-pairs scaling sweep:

- Fixed setting: `reference_corpus=tulu3`, `n=3`, `train_size=10000`,
  `test_size=1000`, `top_k_tokens=1000`, `reference_size=10000`,
  `min_confidence=0.01`
- `k_known_pairs`: `1000`, `5000`, `10000`

Why `n=3` only:

- Prior runs in `iclr_review` showed nearly identical behavior for `n=2,3,4`.
- The objective here is to isolate how collaborative paired-data aggregation
  changes recovery as `min_confidence` is relaxed.

Run:

```bash
bash /workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack/scripts/run_minconf_sweep.sh
```

Collect summary:

```bash
/workspace/codes/AlienLMv2/.venv/bin/python \
  /workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack/scripts/summarize_minconf_results.py
```

Run known-pairs scaling:

```bash
bash /workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack/scripts/run_known_pairs_sweep.sh
```

Collect known-pairs scaling summary:

```bash
/workspace/codes/AlienLMv2/.venv/bin/python \
  /workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack/scripts/summarize_known_pairs_results.py
```
