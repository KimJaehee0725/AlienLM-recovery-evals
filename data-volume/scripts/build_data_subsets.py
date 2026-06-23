#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET_SPECS = [
    {
        "alias": "magpie_pro",
        "path": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "split": "train",
        "weight": 2,
    },
    {
        "alias": "magpie_reasoning",
        "path": "Magpie-Align/Magpie-Reasoning-V1-150K",
        "split": "train",
        "weight": 1,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build random data-volume subsets for rebuttal experiments.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50000, 150000],
        help="Subset sizes to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for subset sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/data-volume/data/subsets"),
        help="Directory where subset JSONL files and metadata are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subset files.",
    )
    return parser.parse_args()


def allocate_counts(total: int, weights: list[int]) -> list[int]:
    allocated = []
    remaining_total = total
    remaining_weight = sum(weights)
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            count = remaining_total
        else:
            count = total * weight // sum(weights)
            remaining_total -= count
            remaining_weight -= weight
        allocated.append(count)
    return allocated


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_by_alias = {}
    dataset_lengths = {}
    for spec in DATASET_SPECS:
        ds = load_dataset(spec["path"], split=spec["split"])
        datasets_by_alias[spec["alias"]] = ds
        dataset_lengths[spec["alias"]] = len(ds)

    weights = [spec["weight"] for spec in DATASET_SPECS]

    manifest = {
        "seed": args.seed,
        "datasets": [
            {
                "alias": spec["alias"],
                "path": spec["path"],
                "split": spec["split"],
                "weight": spec["weight"],
                "length": dataset_lengths[spec["alias"]],
            }
            for spec in DATASET_SPECS
        ],
        "subsets": [],
    }

    for size in args.sizes:
        counts = allocate_counts(size, weights)
        rows = []

        for spec, count in zip(DATASET_SPECS, counts):
            ds = datasets_by_alias[spec["alias"]]
            if count > len(ds):
                raise ValueError(
                    f"Requested {count} rows from {spec['alias']}, but only {len(ds)} are available."
                )
            rng = random.Random(args.seed + size + spec["weight"])
            indices = rng.sample(range(len(ds)), count)
            for index in indices:
                row = dict(ds[index])
                row["source_dataset"] = spec["alias"]
                rows.append(row)

        shuffle_rng = random.Random(args.seed + size * 17)
        shuffle_rng.shuffle(rows)

        subset_name = f"magpie_mix_{size // 1000}k_seed{args.seed}"
        jsonl_path = args.output_dir / f"{subset_name}.jsonl"
        meta_path = args.output_dir / f"{subset_name}.meta.json"

        if not args.overwrite and (jsonl_path.exists() or meta_path.exists()):
            raise FileExistsError(
                f"Subset already exists: {jsonl_path}. Use --overwrite to replace it."
            )

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        metadata = {
            "subset_name": subset_name,
            "total_rows": size,
            "seed": args.seed,
            "source_counts": {
                spec["alias"]: count for spec, count in zip(DATASET_SPECS, counts)
            },
            "output_jsonl": str(jsonl_path),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest["subsets"].append(metadata)

        print(f"Wrote {size} rows to {jsonl_path}")
        print(f"Source counts: {metadata['source_counts']}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
