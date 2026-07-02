#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer


# The base attack code (ngram_datasets.py) lives in the experiment directory,
# which is the parent of this scripts/ directory.
ATTACK_DIR = Path(__file__).resolve().parent.parent
if str(ATTACK_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACK_DIR))

from ngram_datasets import NGramAttack, load_reference_corpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuttal-only mixed-seed collaborative attack sweep"
    )
    parser.add_argument("--reference_corpus", type=str, default="tulu3")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--k_values", type=str, default="1000,5000,10000")
    parser.add_argument("--top_k_tokens", type=int, default=1000)
    parser.add_argument("--reference_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--org_tokenizer_path", type=str, required=True)
    parser.add_argument("--alien_tokenizer_paths", type=str, required=True)
    parser.add_argument("--assignment_mode", type=str, choices=["shared", "mixed"], required=True)
    parser.add_argument("--assignment_seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.95)
    parser.add_argument("--min_occurrences", type=int, default=3)
    return parser.parse_args()


def batch_encode_texts(tokenizer, texts: Sequence[str], batch_size: int) -> List[List[int]]:
    encoded: List[List[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = tokenizer.batch_encode_plus(
            list(batch),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        encoded.extend(result["input_ids"])
    return encoded


def load_magpie_subset(total_needed: int, cache_dir: str) -> tuple[List[str], Dict[str, int]]:
    dataset_specs = [
        ("reasoning", "Magpie-Align/Magpie-Reasoning-V1-150K"),
        ("pro", "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered"),
    ]
    texts: List[str] = []
    counts = {name: 0 for name, _ in dataset_specs}

    for short_name, dataset_name in dataset_specs:
        if len(texts) >= total_needed:
            break
        split = load_dataset(dataset_name, cache_dir=cache_dir, split="train")
        take = min(total_needed - len(texts), len(split))
        for row in split.select(range(take)):
            texts.append(row["instruction"] + row["response"])
            counts[short_name] += 1
            if len(texts) >= total_needed:
                break

    if len(texts) < total_needed:
        raise RuntimeError(f"Requested {total_needed} samples but only loaded {len(texts)}")
    return texts, counts


def build_assignments(
    total_needed: int, mode: str, num_tokenizers: int, seed: int
) -> List[int]:
    if mode == "shared":
        return [0] * total_needed
    rng = random.Random(seed)
    return [rng.randrange(num_tokenizers) for _ in range(total_needed)]


def build_alien_texts(
    plain_ids: Sequence[Sequence[int]],
    alien_tokenizers: Sequence,
    assignments: Sequence[int],
) -> List[str]:
    alien_texts: List[str] = []
    for ids, tokenizer_idx in zip(plain_ids, assignments):
        alien_texts.append(alien_tokenizers[tokenizer_idx].decode(ids))
    return alien_texts


def extract_known_mappings(
    known_alien_ids: Sequence[Sequence[int]],
    known_plain_ids: Sequence[Sequence[int]],
    min_confidence: float,
    min_occurrences: int,
) -> tuple[Dict[int, int], Dict[str, float]]:
    mapping_counts: Dict[int, Counter] = defaultdict(Counter)
    token_totals: Counter = Counter()
    aligned_pairs = 0
    aligned_tokens = 0

    for alien_ids, plain_ids in zip(known_alien_ids, known_plain_ids):
        if len(alien_ids) != len(plain_ids):
            continue
        aligned_pairs += 1
        aligned_tokens += len(alien_ids)
        for alien_id, plain_id in zip(alien_ids, plain_ids):
            mapping_counts[alien_id][plain_id] += 1
            token_totals[alien_id] += 1

    known_mappings: Dict[int, int] = {}
    conflicting_tokens = 0
    for alien_id, counts in mapping_counts.items():
        top_plain_id, top_count = counts.most_common(1)[0]
        consensus = top_count / token_totals[alien_id]
        if len(counts) > 1:
            conflicting_tokens += 1
        if token_totals[alien_id] >= min_occurrences and consensus >= min_confidence:
            known_mappings[alien_id] = top_plain_id

    stats = {
        "aligned_pairs": aligned_pairs,
        "total_pairs": len(known_alien_ids),
        "aligned_pair_rate": aligned_pairs / len(known_alien_ids) if known_alien_ids else 0.0,
        "aligned_tokens": aligned_tokens,
        "candidate_cipher_tokens": len(mapping_counts),
        "conflicting_cipher_tokens": conflicting_tokens,
        "known_mappings_size": len(known_mappings),
    }
    return known_mappings, stats


def evaluate_known_only(
    test_alien_ids: Sequence[Sequence[int]],
    test_plain_ids: Sequence[Sequence[int]],
    known_mappings: Dict[int, int],
) -> Dict[str, float]:
    total_tokens = 0
    covered_tokens = 0
    correct_tokens = 0

    for alien_ids, plain_ids in zip(test_alien_ids, test_plain_ids):
        min_len = min(len(alien_ids), len(plain_ids))
        for alien_id, plain_id in zip(alien_ids[:min_len], plain_ids[:min_len]):
            total_tokens += 1
            if alien_id in known_mappings:
                covered_tokens += 1
                if known_mappings[alien_id] == plain_id:
                    correct_tokens += 1

    return {
        "known_only_total_tokens": total_tokens,
        "known_only_covered_tokens": covered_tokens,
        "known_only_correct_tokens": correct_tokens,
        "known_only_coverage": covered_tokens / total_tokens if total_tokens else 0.0,
        "known_only_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "known_only_precision": correct_tokens / covered_tokens if covered_tokens else 0.0,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    alien_tokenizer_paths = [path.strip() for path in args.alien_tokenizer_paths.split(",") if path.strip()]
    if not alien_tokenizer_paths:
        raise ValueError("No alien tokenizer paths provided")
    if args.assignment_mode == "shared":
        alien_tokenizer_paths = [alien_tokenizer_paths[0]]

    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    max_k = max(k_values)
    total_needed = args.train_size + max_k + args.test_size

    print("=" * 80, flush=True)
    print("MIXED-SEED COLLABORATIVE ATTACK SWEEP", flush=True)
    print("=" * 80, flush=True)
    print(f"assignment_mode={args.assignment_mode}", flush=True)
    print(f"k_values={k_values}", flush=True)
    print(f"min_confidence={args.min_confidence}", flush=True)
    print(f"min_occurrences={args.min_occurrences}", flush=True)
    print("=" * 80, flush=True)

    print("\n[Step 1] Loading tokenizers...", flush=True)
    original_tokenizer = AutoTokenizer.from_pretrained(args.org_tokenizer_path)
    alien_tokenizers = [AutoTokenizer.from_pretrained(path) for path in alien_tokenizer_paths]
    print(f"✓ Loaded {len(alien_tokenizers)} alien tokenizer(s)", flush=True)

    print("\n[Step 2] Loading Magpie subset...", flush=True)
    original_texts, dataset_counts = load_magpie_subset(total_needed, args.cache_dir)
    print(f"✓ Loaded {len(original_texts)} Magpie samples", flush=True)
    print(f"  Dataset counts: {dataset_counts}", flush=True)

    print("\n[Step 3] Encoding original texts...", flush=True)
    plain_ids = batch_encode_texts(original_tokenizer, original_texts, args.encode_batch_size)
    print("✓ Original texts encoded", flush=True)

    print("\n[Step 4] Building alien texts...", flush=True)
    assignments = build_assignments(
        total_needed=total_needed,
        mode=args.assignment_mode,
        num_tokenizers=len(alien_tokenizers),
        seed=args.assignment_seed,
    )
    assignment_counts = Counter(assignments)
    alien_texts = build_alien_texts(plain_ids, alien_tokenizers, assignments)
    print(f"✓ Alien texts built with assignment counts: {dict(assignment_counts)}", flush=True)

    print("\n[Step 5] Re-encoding alien texts under original tokenizer...", flush=True)
    alien_surface_ids = batch_encode_texts(original_tokenizer, alien_texts, args.encode_batch_size)
    print("✓ Alien texts re-encoded", flush=True)

    print("\n[Step 6] Loading reference corpus...", flush=True)
    reference_corpus = load_reference_corpus(
        dataset_name=args.reference_corpus,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        max_size=args.reference_size,
    )
    print(f"✓ Loaded {len(reference_corpus)} reference texts", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        output_root / "run_config.json",
        {
            "assignment_mode": args.assignment_mode,
            "assignment_seed": args.assignment_seed,
            "assignment_counts": dict(assignment_counts),
            "alien_tokenizer_paths": alien_tokenizer_paths,
            "org_tokenizer_path": args.org_tokenizer_path,
            "dataset_counts": dataset_counts,
            "k_values": k_values,
            "n": args.n,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "top_k_tokens": args.top_k_tokens,
            "reference_corpus": args.reference_corpus,
            "reference_size": args.reference_size,
            "min_confidence": args.min_confidence,
            "min_occurrences": args.min_occurrences,
        },
    )

    for k_known_pairs in k_values:
        print("\n" + "-" * 80, flush=True)
        print(f"[Sweep] k_known_pairs={k_known_pairs}", flush=True)
        print("-" * 80, flush=True)

        known_start = args.train_size
        known_end = known_start + k_known_pairs
        test_end = known_end + args.test_size

        train_alien = alien_texts[: args.train_size]
        known_alien_ids = alien_surface_ids[known_start:known_end]
        known_plain_ids = plain_ids[known_start:known_end]
        test_alien = alien_texts[known_end:test_end]
        test_alien_ids = alien_surface_ids[known_end:test_end]
        test_plain_ids = plain_ids[known_end:test_end]
        test_plain_texts = original_texts[known_end:test_end]

        known_mappings, known_stats = extract_known_mappings(
            known_alien_ids=known_alien_ids,
            known_plain_ids=known_plain_ids,
            min_confidence=args.min_confidence,
            min_occurrences=args.min_occurrences,
        )
        print(
            f"✓ Extracted {len(known_mappings)} known mappings from "
            f"{known_stats['aligned_pairs']}/{known_stats['total_pairs']} aligned pairs",
            flush=True,
        )

        known_only_metrics = evaluate_known_only(
            test_alien_ids=test_alien_ids,
            test_plain_ids=test_plain_ids,
            known_mappings=known_mappings,
        )
        print(
            "✓ Known-only coverage="
            f"{known_only_metrics['known_only_coverage'] * 100:.2f}% "
            "accuracy="
            f"{known_only_metrics['known_only_accuracy'] * 100:.2f}%",
            flush=True,
        )

        attack = NGramAttack(
            n=args.n,
            top_k_tokens=args.top_k_tokens,
            known_mappings=known_mappings,
        )
        attack_metrics, recovered_mapping = attack.attack(
            alien_texts=train_alien,
            reference_texts=reference_corpus,
            test_alien_texts=test_alien,
            test_ground_truth_texts=test_plain_texts,
            tokenizer=original_tokenizer,
            true_bijection=None,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            top_k_tokens=args.top_k_tokens,
        )

        summary = {
            "assignment_mode": args.assignment_mode,
            "assignment_seed": args.assignment_seed,
            "assignment_counts": dict(assignment_counts),
            "reference_corpus": args.reference_corpus,
            "reference_size": args.reference_size,
            "n": args.n,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "top_k_tokens": args.top_k_tokens,
            "k_known_pairs": k_known_pairs,
            "min_confidence": args.min_confidence,
            "min_occurrences": args.min_occurrences,
            **known_stats,
            **known_only_metrics,
            "token_accuracy": attack_metrics["token_accuracy"],
            "correct_tokens": attack_metrics["correct_tokens"],
            "total_tokens": attack_metrics["total_tokens"],
            "recovered_mapping_size": len(recovered_mapping),
        }

        run_dir = output_root / f"k{k_known_pairs}"
        save_json(run_dir / "summary.json", summary)
        save_json(
            run_dir / "known_mappings.json",
            {str(key): value for key, value in known_mappings.items()},
        )
        save_json(
            run_dir / "recovered_mapping.json",
            {str(key): value for key, value in recovered_mapping.items()},
        )
        print(f"✓ Saved results to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
