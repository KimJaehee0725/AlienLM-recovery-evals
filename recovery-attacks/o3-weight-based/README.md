# O3 Weight-Based Token Mapping Attack

Recover the secret alien-to-original token bijection of an adapted model purely from its learned token vectors, without access to the bijection seed.

**Paper reference**: Table 3 (O3, model access); Appendix E.5 (weight-based mapping without the bijection seed).

**What it measures**: an observer who holds the adapted model weights performs nearest-neighbor matching of alien-token vectors against original-token vectors in embedding space or LM-head space. Similarity is cosine on L2-normalized vectors; for each alien token the top-1 (argmax) original token is selected, and the script reports top-1 mapping accuracy against the ground-truth bijection (recovered by matching token surface strings between the alien and base tokenizers).

> **Reconstruction note**: The original repository did **not** contain a standalone implementation of this attack. The only related artifact (the n-gram attack notebook) implements a statistical n-gram bijection-recovery attack instead. `weight_based_mapping.py` is therefore a faithful minimal reconstruction of the method described in Appendix E.5; see the header comment in that file for the exact procedure.

**Inputs** (described, never hardcoded):
- `ALIENLM_CODE_ROOT` — environment variable pointing at the AlienLM code/checkpoint root.
- `MODEL_PATH` — path or Hugging Face id of the adapted model whose weights are observed (defaults under `ALIENLM_CODE_ROOT`).
- `ALIEN_TOKENIZER` — path or HF id of the alien (adapted) tokenizer (defaults to `MODEL_PATH`).
- `BASE_TOKENIZER` — path or HF id of the original/base tokenizer (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`). Access to gated base models on the Hugging Face Hub may require authentication.

**Run** (for reproduction; do not execute as part of vendoring):

```bash
export ALIENLM_CODE_ROOT=/path/to/alienlm

# Single space
python weight_based_mapping.py \
    --model-path "$ALIENLM_CODE_ROOT/outputs/Llama3-8B-Instruct-AlienLM/checkpoint" \
    --alien-tokenizer "$ALIENLM_CODE_ROOT/outputs/Llama3-8B-Instruct-AlienLM/checkpoint" \
    --base-tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --space embedding

# Both spaces via the helper script
bash run.sh
```

CLI flags: `--model-path`, `--alien-tokenizer`, `--base-tokenizer`, `--space {embedding,lm_head}`, plus optional `--device` and `--batch-size`.

**Outputs**: the script prints the top-1 mapping accuracy (and counts of scored / correct tokens) to stdout. It does not write result files. If you redirect output, send it to an ignored directory such as `./attack_results/` (git-ignored), which is not part of the public repo.

**Notes**:
- Dependencies: `torch`, `transformers` (and their dependencies).
- A GPU is recommended for the 8B-scale embedding / LM-head matrices but the script falls back to CPU automatically (`--device cpu`).
- No OpenAI API key is required for this attack.
- LM-head space (`--space lm_head`) requires a model with untied output embeddings; for tied-weight models use `--space embedding`.
