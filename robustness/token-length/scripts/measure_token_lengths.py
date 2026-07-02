#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
SEED = int(os.environ.get("SEED", "42"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "10000"))
SAMPLE_PATH = Path(
    os.environ.get("SAMPLE_PATH", OUTPUT_DIR / f"magpie_450k_sample{SAMPLE_SIZE}_seed{SEED}.jsonl")
)
SUMMARY_PATH = OUTPUT_DIR / "token_length_summary.json"
MARKDOWN_PATH = OUTPUT_DIR / "token_length_summary.md"

HF_DATASET_CACHE = os.environ.get("HF_DATASETS_CACHE")
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"

DATASET_SPECS = [
    {
        "alias": "magpie_pro",
        "path": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "split": "train",
    },
    {
        "alias": "magpie_reasoning",
        "path": "Magpie-Align/Magpie-Reasoning-V1-150K",
        "split": "train",
    },
]

TOKENIZER_SPECS = {
    "llama3_8b_instruct": os.environ.get(
        "LLAMA_TOKENIZER_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"
    ),
    "qwen25_7b_instruct": os.environ.get(
        "QWEN_TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct"
    ),
    "gemma2_9b_it": os.environ.get("GEMMA_TOKENIZER_PATH", "google/gemma-2-9b-it"),
}


def convert_conversations(conversations: list[dict]) -> list[dict]:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "user": "user",
        "system": "system",
    }
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from"), turn.get("role", "user"))
        messages.append({"role": role, "content": turn.get("value", turn.get("content", ""))})
    return messages


def get_messages(row: dict) -> list[dict]:
    if row.get("conversations"):
        return convert_conversations(row["conversations"])

    instruction = row.get("instruction", "")
    response = row.get("response", "")
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def get_plain_text(row: dict) -> str:
    messages = get_messages(row)
    return "\n\n".join(msg["content"] for msg in messages if msg["content"])


def sample_pool_indices(lengths: list[int], sample_size: int, seed: int) -> list[tuple[int, int, int]]:
    total = sum(lengths)
    if sample_size > total:
        raise ValueError(f"Requested {sample_size} samples from pool of size {total}")

    rng = random.Random(seed)
    global_indices = sorted(rng.sample(range(total), sample_size))

    sampled = []
    offset = 0
    for ds_idx, length in enumerate(lengths):
        upper = offset + length
        for global_idx in global_indices:
            if global_idx < offset:
                continue
            if global_idx >= upper:
                continue
            local_idx = global_idx - offset
            sampled.append((global_idx, ds_idx, local_idx))
        offset = upper
    return sampled


def mean(values: Iterable[int]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": statistics.median(values) if values else 0.0,
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "total": sum(values),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets_by_alias = []
    lengths = []
    for spec in DATASET_SPECS:
        ds = load_dataset(spec["path"], split=spec["split"], cache_dir=HF_DATASET_CACHE)
        datasets_by_alias.append((spec["alias"], ds))
        lengths.append(len(ds))

    sampled_pool = sample_pool_indices(lengths=lengths, sample_size=SAMPLE_SIZE, seed=SEED)

    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=LOCAL_FILES_ONLY)
        for name, path in TOKENIZER_SPECS.items()
    }

    sample_rows = []
    length_values = {name: [] for name in tokenizers}
    length_values_by_source = {name: defaultdict(list) for name in tokenizers}
    source_counter = Counter()

    for idx, (global_idx, ds_idx, local_idx) in enumerate(sampled_pool, start=1):
        source_alias, ds = datasets_by_alias[ds_idx]
        row = ds[local_idx]
        text = get_plain_text(row)

        result_row = {
            "global_index": global_idx,
            "source_dataset": source_alias,
            "local_index": local_idx,
            "char_length": len(text),
            "uuid": row.get("uuid"),
        }

        for tokenizer_name, tokenizer in tokenizers.items():
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            token_len = len(input_ids)
            result_row[f"{tokenizer_name}_tokens"] = token_len
            length_values[tokenizer_name].append(token_len)
            length_values_by_source[tokenizer_name][source_alias].append(token_len)

        sample_rows.append(result_row)
        source_counter[source_alias] += 1

        if idx % 500 == 0:
            print(f"Processed {idx}/{SAMPLE_SIZE} samples")

    with SAMPLE_PATH.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "data_pool": [
            {
                "alias": alias,
                "length": len(ds),
            }
            for alias, ds in datasets_by_alias
        ],
        "sample_source_counts": dict(source_counter),
        "tokenizers": {},
    }

    for tokenizer_name in tokenizers:
        overall = summarize(length_values[tokenizer_name])
        by_source = {
            source_alias: summarize(values)
            for source_alias, values in sorted(length_values_by_source[tokenizer_name].items())
        }
        summary["tokenizers"][tokenizer_name] = {
            "path": TOKENIZER_SPECS[tokenizer_name],
            "overall": overall,
            "by_source": by_source,
        }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ordered = sorted(
        [
            (
                name,
                payload["overall"]["mean"],
                payload["overall"]["total"],
            )
            for name, payload in summary["tokenizers"].items()
        ],
        key=lambda item: item[1],
    )
    baseline_name, baseline_mean, baseline_total = ordered[0]

    lines = [
        "# Token Length Summary",
        "",
        f"- Sample size: `{SAMPLE_SIZE}`",
        f"- Seed: `{SEED}`",
        f"- Sample source counts: `{dict(source_counter)}`",
        f"- Length definition: identical plain text per sample with `add_special_tokens=False`",
        "",
        "| tokenizer | avg tokens/sample | median | stdev | min | max | total tokens | delta vs min avg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for tokenizer_name, payload in sorted(summary["tokenizers"].items()):
        overall = payload["overall"]
        delta_pct = ((overall["mean"] / baseline_mean) - 1.0) * 100 if baseline_mean else 0.0
        lines.append(
            f"| {tokenizer_name} | {overall['mean']:.4f} | {overall['median']:.1f} | "
            f"{overall['stdev']:.4f} | {overall['min']} | {overall['max']} | "
            f"{overall['total']} | {delta_pct:+.2f}% |"
        )

    lines.extend(
        [
            "",
            f"Baseline shortest average tokenizer: `{baseline_name}` ({baseline_mean:.4f} tokens/sample)",
            "",
            "## By Source",
            "",
            "| tokenizer | source | avg tokens/sample | total tokens |",
            "| --- | --- | ---: | ---: |",
        ]
    )

    for tokenizer_name, payload in sorted(summary["tokenizers"].items()):
        for source_alias, source_stats in sorted(payload["by_source"].items()):
            lines.append(
                f"| {tokenizer_name} | {source_alias} | {source_stats['mean']:.4f} | {source_stats['total']} |"
            )

    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["sample_source_counts"], indent=2))
    for tokenizer_name, payload in sorted(summary["tokenizers"].items()):
        print(
            f"{tokenizer_name}: avg={payload['overall']['mean']:.4f}, "
            f"median={payload['overall']['median']:.1f}, total={payload['overall']['total']}"
        )


if __name__ == "__main__":
    main()
