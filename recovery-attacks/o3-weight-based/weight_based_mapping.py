#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# RECONSTRUCTED IMPLEMENTATION (not vendored from the original source).
#
# The original repository did NOT contain a standalone implementation of this
# attack. The only related notebook (attack_scenario/n-gram/notebookv2.ipynb)
# implements a *statistical n-gram* bijection-recovery attack, not the
# weight-based embedding / LM-head cosine mapping described in the paper.
#
# This file therefore faithfully reconstructs the O3 "weight-based mapping"
# threat model from the paper:
#   - Paper reference: Table 3 (O3, model access);
#     Appendix E.5 (weight-based mapping without the bijection seed).
#
# Threat model (O3): an observer has the *adapted* model weights but NOT the
# secret bijection seed that defines the alien<->original token mapping. The
# observer attempts to recover the mapping purely from the geometry of the
# learned token vectors.
#
# Method:
#   1. Load the adapted model's token vectors in the chosen weight space:
#        - "embedding": model.get_input_embeddings().weight   (embed_tokens)
#        - "lm_head"  : model.get_output_embeddings().weight   (lm_head)
#   2. The adapted vocabulary contains both "original" token rows (tokens that
#      kept their original surface form) and "alien" token rows (tokens whose
#      surface form was remapped by the secret bijection). We identify the two
#      groups by comparing the alien tokenizer's vocab against the base
#      tokenizer's vocab.
#   3. L2-normalize every row, then for each alien token row compute cosine
#      similarity against all candidate original token rows and take the
#      argmax (top-1 nearest neighbor).
#   4. The ground-truth pairing is recovered by matching token *surface
#      strings*: the alien tokenizer maps each alien id to a string; the same
#      string exists in the base tokenizer with its own id. A predicted match
#      is correct when the argmax-selected original row corresponds to the
#      base id for that same surface string.
#   5. Report top-1 mapping accuracy = (# correct top-1 matches) / (# alien
#      tokens evaluated).
# -----------------------------------------------------------------------------

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def l2_normalize(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise L2 normalization."""
    norms = matrix.norm(dim=1, keepdim=True).clamp_min(eps)
    return matrix / norms


def get_weight_space(model, space: str) -> torch.Tensor:
    """Return the token-vector matrix for the requested weight space.

    "embedding" -> input embeddings (embed_tokens)
    "lm_head"   -> output embeddings (lm_head / unembedding)
    """
    if space == "embedding":
        weight = model.get_input_embeddings().weight
    elif space == "lm_head":
        out = model.get_output_embeddings()
        if out is None:
            raise ValueError(
                "Model has no separate output embeddings (tied weights). "
                "Use --space embedding instead."
            )
        weight = out.weight
    else:
        raise ValueError(f"Unknown space: {space!r}")
    return weight.detach().to(torch.float32)


def build_token_groups(alien_tok, base_tok):
    """Partition the adapted vocabulary into alien vs. original token ids.

    Returns:
        alien_ids:        list[int] adapted-vocab ids whose surface string
                          differs from the base tokenizer's string for the
                          same id (the remapped / "alien" tokens).
        gt_base_id:       dict[int -> int] mapping an alien adapted-vocab id to
                          the *base* tokenizer id that owns the same surface
                          string (the ground-truth original token).
        candidate_ids:    list[int] adapted-vocab ids treated as candidate
                          "original" token rows (surface string unchanged).
    """
    alien_vocab = alien_tok.get_vocab()      # str -> id (adapted vocab)
    base_vocab = base_tok.get_vocab()        # str -> id (base vocab)

    # id -> surface string for the adapted vocab.
    alien_id_to_str = {idx: tok for tok, idx in alien_vocab.items()}
    base_id_to_str = {idx: tok for tok, idx in base_vocab.items()}

    alien_ids = []
    candidate_ids = []
    gt_base_id = {}

    for idx, surface in alien_id_to_str.items():
        base_surface = base_id_to_str.get(idx)
        if base_surface is not None and base_surface == surface:
            # Surface form unchanged -> treat as an original / candidate row.
            candidate_ids.append(idx)
            continue
        # Surface form changed -> this is an alien (remapped) token. Its
        # ground-truth original token is whichever base id owns this surface.
        base_id = base_vocab.get(surface)
        if base_id is None:
            # Surface string not present in the base vocab; cannot score.
            continue
        alien_ids.append(idx)
        gt_base_id[idx] = base_id

    return alien_ids, gt_base_id, candidate_ids


def run_attack(model_path, alien_tokenizer, base_tokenizer, space, device, batch_size):
    print(f"[load] base tokenizer: {base_tokenizer}", flush=True)
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer)
    print(f"[load] alien tokenizer: {alien_tokenizer}", flush=True)
    alien_tok = AutoTokenizer.from_pretrained(alien_tokenizer)

    print(f"[load] model: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    model.eval()

    weight = get_weight_space(model, space)  # [vocab, dim]
    weight = weight.to(device)
    weight = l2_normalize(weight)
    print(f"[info] weight space '{space}' shape: {tuple(weight.shape)}", flush=True)

    alien_ids, gt_base_id, candidate_ids = build_token_groups(alien_tok, base_tok)
    print(
        f"[info] alien tokens to map: {len(alien_ids)} | "
        f"candidate original rows: {len(candidate_ids)}",
        flush=True,
    )

    if not alien_ids or not candidate_ids:
        raise RuntimeError(
            "Could not identify alien/original token groups from the tokenizers. "
            "Check that --alien-tokenizer and --base-tokenizer are correct."
        )

    # Candidate matrix: rows for original tokens (in adapted-vocab index space).
    candidate_idx_tensor = torch.tensor(candidate_ids, device=device)
    candidate_matrix = weight.index_select(0, candidate_idx_tensor)  # [C, dim]

    # For ground-truth scoring we need: which candidate row corresponds to the
    # base id that owns the alien token's surface string. The candidate rows are
    # adapted-vocab ids; an unchanged-surface candidate id equals its base id by
    # construction (surface matched). So the ground-truth candidate position is
    # the index of gt_base_id within candidate_ids.
    candidate_pos = {idx: pos for pos, idx in enumerate(candidate_ids)}

    correct = 0
    scored = 0
    missing_gt = 0

    alien_idx_tensor = torch.tensor(alien_ids, device=device)
    for start in range(0, len(alien_ids), batch_size):
        end = min(start + batch_size, len(alien_ids))
        batch_alien_ids = alien_ids[start:end]
        batch_rows = weight.index_select(
            0, alien_idx_tensor[start:end]
        )  # [B, dim]

        # Cosine similarity (rows already L2-normalized) -> [B, C].
        sims = batch_rows @ candidate_matrix.T
        top1 = sims.argmax(dim=1)  # candidate-position index per alien token

        for i, alien_id in enumerate(batch_alien_ids):
            gt_id = gt_base_id[alien_id]
            gt_pos = candidate_pos.get(gt_id)
            if gt_pos is None:
                # Ground-truth original token is not among candidate rows
                # (e.g. its surface was itself remapped). Skip it.
                missing_gt += 1
                continue
            scored += 1
            if int(top1[i].item()) == gt_pos:
                correct += 1

        print(
            f"[progress] {end}/{len(alien_ids)} alien tokens scored",
            flush=True,
        )

    accuracy = correct / scored if scored else 0.0
    print("\n========== RESULTS ==========", flush=True)
    print(f"space:                {space}", flush=True)
    print(f"alien tokens scored:  {scored}", flush=True)
    print(f"skipped (no gt row):  {missing_gt}", flush=True)
    print(f"correct top-1:        {correct}", flush=True)
    print(f"top-1 accuracy:       {accuracy:.6f}", flush=True)
    print("=============================", flush=True)
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "O3 weight-based token mapping attack (reconstructed from the paper, "
            "Appendix E.5). Nearest-neighbor matching of alien-token vs "
            "original-token vectors in embedding / LM-head space using cosine "
            "similarity on L2-normalized vectors; reports top-1 mapping accuracy."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or HF id of the adapted model whose weights are observed.",
    )
    parser.add_argument(
        "--alien-tokenizer",
        required=True,
        help="Path or HF id of the alien (adapted) tokenizer.",
    )
    parser.add_argument(
        "--base-tokenizer",
        required=True,
        help="Path or HF id of the original / base tokenizer.",
    )
    parser.add_argument(
        "--space",
        choices=["embedding", "lm_head"],
        default="embedding",
        help="Weight space to attack: input embeddings or LM head.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of alien token rows scored per batch.",
    )
    args = parser.parse_args()

    run_attack(
        model_path=args.model_path,
        alien_tokenizer=args.alien_tokenizer,
        base_tokenizer=args.base_tokenizer,
        space=args.space,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    sys.exit(main())
