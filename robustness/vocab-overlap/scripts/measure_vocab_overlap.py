#!/usr/bin/env python3

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from transformers import AutoTokenizer


OUTPUT_DIR = Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/vocab-overlap/results")
JSON_PATH = OUTPUT_DIR / "vocab_overlap_summary.json"
MD_PATH = OUTPUT_DIR / "vocab_overlap_summary.md"

TOKENIZER_SPECS = {
    "llama3_8b_instruct": "/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen25_7b_instruct": "/workspace/CACHE/MODELS/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "gemma2_9b_it": "/workspace/CACHE/MODELS/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819",
}


def pairwise_stats(a: set[str], b: set[str]) -> dict:
    inter = a & b
    union = a | b
    return {
        "intersection": len(inter),
        "union": len(union),
        "overlap_vs_a": len(inter) / len(a) if a else 0.0,
        "overlap_vs_b": len(inter) / len(b) if b else 0.0,
        "jaccard": len(inter) / len(union) if union else 0.0,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in TOKENIZER_SPECS.items()
    }

    vocab_sets_all = {name: set(tok.get_vocab().keys()) for name, tok in tokenizers.items()}
    special_sets = {name: set(tok.all_special_tokens) for name, tok in tokenizers.items()}
    vocab_sets_nospecial = {
        name: vocab_sets_all[name] - special_sets[name] for name in tokenizers
    }

    summary = {
        "tokenizers": {},
        "pairwise": {
            "excluding_special_tokens": {},
            "including_special_tokens": {},
        },
        "three_way": {},
    }

    for name in tokenizers:
        summary["tokenizers"][name] = {
            "path": TOKENIZER_SPECS[name],
            "vocab_size_including_special": len(vocab_sets_all[name]),
            "special_token_count": len(special_sets[name]),
            "vocab_size_excluding_special": len(vocab_sets_nospecial[name]),
            "special_tokens": sorted(special_sets[name]),
        }

    for a_name, b_name in combinations(tokenizers.keys(), 2):
        key = f"{a_name}__vs__{b_name}"
        summary["pairwise"]["excluding_special_tokens"][key] = {
            "a": a_name,
            "b": b_name,
            **pairwise_stats(vocab_sets_nospecial[a_name], vocab_sets_nospecial[b_name]),
        }
        summary["pairwise"]["including_special_tokens"][key] = {
            "a": a_name,
            "b": b_name,
            **pairwise_stats(vocab_sets_all[a_name], vocab_sets_all[b_name]),
        }

    three_way_all = set.intersection(*vocab_sets_all.values())
    three_way_nospecial = set.intersection(*vocab_sets_nospecial.values())
    union_all = set.union(*vocab_sets_all.values())
    union_nospecial = set.union(*vocab_sets_nospecial.values())

    summary["three_way"] = {
        "excluding_special_tokens": {
            "intersection": len(three_way_nospecial),
            "union": len(union_nospecial),
            "jaccard": len(three_way_nospecial) / len(union_nospecial) if union_nospecial else 0.0,
            "overlap_vs_each": {
                name: len(three_way_nospecial) / len(vocab_sets_nospecial[name])
                for name in tokenizers
            },
        },
        "including_special_tokens": {
            "intersection": len(three_way_all),
            "union": len(union_all),
            "jaccard": len(three_way_all) / len(union_all) if union_all else 0.0,
            "overlap_vs_each": {
                name: len(three_way_all) / len(vocab_sets_all[name])
                for name in tokenizers
            },
        },
    }

    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Vocab Overlap Summary",
        "",
        "Definition:",
        "- Base unit is exact token string identity from `tokenizer.get_vocab().keys()`.",
        "- Main comparison excludes special tokens.",
        "- Metrics reported: `intersection`, `overlap_vs_a`, `overlap_vs_b`, and `jaccard`.",
        "",
        "## Tokenizer Sizes",
        "",
        "| tokenizer | vocab size incl. special | special token count | vocab size excl. special |",
        "| --- | ---: | ---: | ---: |",
    ]

    for name, payload in summary["tokenizers"].items():
        lines.append(
            f"| `{name}` | {payload['vocab_size_including_special']} | "
            f"{payload['special_token_count']} | {payload['vocab_size_excluding_special']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Overlap Excluding Special Tokens",
            "",
            "| pair | intersection | overlap vs A | overlap vs B | jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for key, payload in summary["pairwise"]["excluding_special_tokens"].items():
        lines.append(
            f"| `{payload['a']}` vs `{payload['b']}` | {payload['intersection']} | "
            f"{format_pct(payload['overlap_vs_a'])} | {format_pct(payload['overlap_vs_b'])} | "
            f"{format_pct(payload['jaccard'])} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Overlap Including Special Tokens",
            "",
            "| pair | intersection | overlap vs A | overlap vs B | jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for key, payload in summary["pairwise"]["including_special_tokens"].items():
        lines.append(
            f"| `{payload['a']}` vs `{payload['b']}` | {payload['intersection']} | "
            f"{format_pct(payload['overlap_vs_a'])} | {format_pct(payload['overlap_vs_b'])} | "
            f"{format_pct(payload['jaccard'])} |"
        )

    nospecial = summary["three_way"]["excluding_special_tokens"]
    allspecial = summary["three_way"]["including_special_tokens"]

    lines.extend(
        [
            "",
            "## Three-Way Overlap",
            "",
            "### Excluding Special Tokens",
            "",
            f"- Intersection size: `{nospecial['intersection']}`",
            f"- Union size: `{nospecial['union']}`",
            f"- Jaccard: `{format_pct(nospecial['jaccard'])}`",
            f"- Overlap vs Llama: `{format_pct(nospecial['overlap_vs_each']['llama3_8b_instruct'])}`",
            f"- Overlap vs Qwen: `{format_pct(nospecial['overlap_vs_each']['qwen25_7b_instruct'])}`",
            f"- Overlap vs Gemma: `{format_pct(nospecial['overlap_vs_each']['gemma2_9b_it'])}`",
            "",
            "### Including Special Tokens",
            "",
            f"- Intersection size: `{allspecial['intersection']}`",
            f"- Union size: `{allspecial['union']}`",
            f"- Jaccard: `{format_pct(allspecial['jaccard'])}`",
            f"- Overlap vs Llama: `{format_pct(allspecial['overlap_vs_each']['llama3_8b_instruct'])}`",
            f"- Overlap vs Qwen: `{format_pct(allspecial['overlap_vs_each']['qwen25_7b_instruct'])}`",
            f"- Overlap vs Gemma: `{format_pct(allspecial['overlap_vs_each']['gemma2_9b_it'])}`",
            "",
        ]
    )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved Markdown to {MD_PATH}")
    for key, payload in summary["pairwise"]["excluding_special_tokens"].items():
        print(
            f"{payload['a']} vs {payload['b']}: "
            f"intersection={payload['intersection']}, "
            f"overlap_vs_a={format_pct(payload['overlap_vs_a'])}, "
            f"overlap_vs_b={format_pct(payload['overlap_vs_b'])}, "
            f"jaccard={format_pct(payload['jaccard'])}"
        )


if __name__ == "__main__":
    main()
