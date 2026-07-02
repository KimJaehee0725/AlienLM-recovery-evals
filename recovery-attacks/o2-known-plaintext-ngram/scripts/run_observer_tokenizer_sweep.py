#!/usr/bin/env python3

import argparse
import json
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
        description="Observer-tokenizer sensitivity sweep for Llama AlienLM attacks"
    )
    parser.add_argument("--victim_tokenizer_path", type=str, required=True)
    parser.add_argument("--alien_tokenizer_path", type=str, required=True)
    parser.add_argument("--observer_tokenizers", type=str, required=True)
    parser.add_argument("--reference_corpus", type=str, default="tulu3")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--k_known_pairs", type=int, default=10000)
    parser.add_argument("--top_k_tokens", type=int, default=1000)
    parser.add_argument("--reference_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.01)
    return parser.parse_args()


def batch_encode_texts(tokenizer, texts: Sequence[str], batch_size: int) -> List[List[int]]:
    encoded: List[List[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        result = tokenizer.batch_encode_plus(
            batch,
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


def build_alien_texts(
    original_texts: Sequence[str],
    victim_tokenizer,
    alien_tokenizer,
    encode_batch_size: int,
) -> List[str]:
    plain_ids = batch_encode_texts(victim_tokenizer, original_texts, encode_batch_size)
    return [alien_tokenizer.decode(ids) for ids in plain_ids]


def extract_known_mappings_global(
    known_alien_ids: Sequence[Sequence[int]],
    known_plain_ids: Sequence[Sequence[int]],
    min_confidence: float,
) -> tuple[Dict[int, int], Dict[str, float]]:
    mapping_counts: Dict[int, Counter] = defaultdict(Counter)
    aligned_pairs = 0
    aligned_tokens = 0

    total_pairs = len(known_alien_ids)
    for alien_ids, plain_ids in zip(known_alien_ids, known_plain_ids):
        if len(alien_ids) != len(plain_ids):
            continue
        aligned_pairs += 1
        aligned_tokens += len(alien_ids)
        for alien_id, plain_id in zip(alien_ids, plain_ids):
            mapping_counts[alien_id][plain_id] += 1

    known_mappings: Dict[int, int] = {}
    for alien_id, counts in mapping_counts.items():
        top_plain_id, top_count = counts.most_common(1)[0]
        confidence = top_count / total_pairs if total_pairs else 0.0
        if confidence >= min_confidence:
            known_mappings[alien_id] = top_plain_id

    stats = {
        "aligned_pairs": aligned_pairs,
        "total_pairs": total_pairs,
        "aligned_pair_rate": aligned_pairs / total_pairs if total_pairs else 0.0,
        "aligned_tokens": aligned_tokens,
        "candidate_cipher_tokens": len(mapping_counts),
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


def parse_observer_specs(specs: str) -> List[tuple[str, str]]:
    parsed: List[tuple[str, str]] = []
    for item in specs.split(","):
        if not item.strip():
            continue
        name, path = item.split("=", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def compute_length_stats(tokenized_texts: Sequence[Sequence[int]]) -> Dict[str, float]:
    lengths = [len(ids) for ids in tokenized_texts]
    return {
        "avg_tokens": sum(lengths) / len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "total_tokens": sum(lengths),
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    observer_specs = parse_observer_specs(args.observer_tokenizers)
    total_needed = args.train_size + args.k_known_pairs + args.test_size

    print("=" * 80, flush=True)
    print("OBSERVER TOKENIZER SENSITIVITY SWEEP", flush=True)
    print("=" * 80, flush=True)
    print(f"observer_tokenizers={[name for name, _ in observer_specs]}", flush=True)
    print(f"k_known_pairs={args.k_known_pairs}", flush=True)
    print(f"min_confidence={args.min_confidence}", flush=True)
    print("=" * 80, flush=True)

    print("\n[Step 1] Loading victim tokenizers...", flush=True)
    victim_tokenizer = AutoTokenizer.from_pretrained(args.victim_tokenizer_path)
    alien_tokenizer = AutoTokenizer.from_pretrained(args.alien_tokenizer_path)
    print("✓ Victim tokenizer pair loaded", flush=True)

    print("\n[Step 2] Loading Magpie subset...", flush=True)
    original_texts, dataset_counts = load_magpie_subset(total_needed, args.cache_dir)
    print(f"✓ Loaded {len(original_texts)} Magpie samples", flush=True)
    print(f"  Dataset counts: {dataset_counts}", flush=True)

    print("\n[Step 3] Building fixed Llama AlienLM texts...", flush=True)
    alien_texts = build_alien_texts(
        original_texts=original_texts,
        victim_tokenizer=victim_tokenizer,
        alien_tokenizer=alien_tokenizer,
        encode_batch_size=args.encode_batch_size,
    )
    print("✓ Alien texts built from fixed victim tokenizer + alien tokenizer", flush=True)

    print("\n[Step 4] Loading reference corpus...", flush=True)
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
            "victim_tokenizer_path": args.victim_tokenizer_path,
            "alien_tokenizer_path": args.alien_tokenizer_path,
            "observer_tokenizers": observer_specs,
            "dataset_counts": dataset_counts,
            "reference_corpus": args.reference_corpus,
            "reference_size": args.reference_size,
            "n": args.n,
            "train_size": args.train_size,
            "k_known_pairs": args.k_known_pairs,
            "test_size": args.test_size,
            "top_k_tokens": args.top_k_tokens,
            "min_confidence": args.min_confidence,
        },
    )

    for observer_name, observer_path in observer_specs:
        print("\n" + "-" * 80, flush=True)
        print(f"[Observer] {observer_name}", flush=True)
        print("-" * 80, flush=True)
        observer_tokenizer = AutoTokenizer.from_pretrained(observer_path)

        plain_ids = batch_encode_texts(observer_tokenizer, original_texts, args.encode_batch_size)
        alien_ids = batch_encode_texts(observer_tokenizer, alien_texts, args.encode_batch_size)
        length_stats_plain = compute_length_stats(plain_ids)
        length_stats_alien = compute_length_stats(alien_ids)

        known_start = args.train_size
        known_end = known_start + args.k_known_pairs
        test_end = known_end + args.test_size

        known_mappings, known_stats = extract_known_mappings_global(
            known_alien_ids=alien_ids[known_start:known_end],
            known_plain_ids=plain_ids[known_start:known_end],
            min_confidence=args.min_confidence,
        )
        print(
            f"✓ Extracted {len(known_mappings)} known mappings from "
            f"{known_stats['aligned_pairs']}/{known_stats['total_pairs']} aligned pairs",
            flush=True,
        )

        known_only_metrics = evaluate_known_only(
            test_alien_ids=alien_ids[known_end:test_end],
            test_plain_ids=plain_ids[known_end:test_end],
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
            alien_texts=alien_texts[: args.train_size],
            reference_texts=reference_corpus,
            test_alien_texts=alien_texts[known_end:test_end],
            test_ground_truth_texts=original_texts[known_end:test_end],
            tokenizer=observer_tokenizer,
            true_bijection=None,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            top_k_tokens=args.top_k_tokens,
        )

        summary = {
            "observer_name": observer_name,
            "observer_tokenizer_path": observer_path,
            "n": args.n,
            "train_size": args.train_size,
            "k_known_pairs": args.k_known_pairs,
            "test_size": args.test_size,
            "top_k_tokens": args.top_k_tokens,
            "reference_corpus": args.reference_corpus,
            "reference_size": args.reference_size,
            "min_confidence": args.min_confidence,
            "plain_length_stats": length_stats_plain,
            "alien_length_stats": length_stats_alien,
            **known_stats,
            **known_only_metrics,
            "token_accuracy": attack_metrics["token_accuracy"],
            "correct_tokens": attack_metrics["correct_tokens"],
            "total_tokens": attack_metrics["total_tokens"],
            "recovered_mapping_size": len(recovered_mapping),
        }

        run_dir = output_root / observer_name
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
