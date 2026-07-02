#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


HF_DATASET_CACHE = os.environ.get("HF_DATASETS_CACHE")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
SAMPLE_PATH = Path(os.environ.get("SAMPLE_PATH", OUTPUT_DIR / "magpie_450k_sample10000_seed42.jsonl"))
JSON_PATH = OUTPUT_DIR / "original_vs_alien_token_length_summary.json"
MD_PATH = OUTPUT_DIR / "original_vs_alien_token_length_summary.md"
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"


DATASET_SPECS = {
    "magpie_pro": {
        "path": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "split": "train",
    },
    "magpie_reasoning": {
        "path": "Magpie-Align/Magpie-Reasoning-V1-150K",
        "split": "train",
    },
}


MODEL_SPECS = {
    "llama3_8b_instruct": {
        "original_tokenizer": os.environ.get(
            "LLAMA_TOKENIZER_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"
        ),
        "alien_tokenizer": os.environ.get(
            "LLAMA_ALIEN_TOKENIZER_PATH", "dsba-lab/llama3-8b-instruct-alienlm-full"
        ),
        "alien_family_note": "Alien tokenizer built from the Qwen-family vocabulary permutation.",
    },
    "qwen25_7b_instruct": {
        "original_tokenizer": os.environ.get("QWEN_TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        "alien_tokenizer": os.environ.get("QWEN_ALIEN_TOKENIZER_PATH", ""),
        "alien_family_note": "Alien tokenizer built from the Llama-family vocabulary permutation.",
    },
    "gemma2_9b_it": {
        "original_tokenizer": os.environ.get("GEMMA_TOKENIZER_PATH", "google/gemma-2-9b-it"),
        "alien_tokenizer": os.environ.get("GEMMA_ALIEN_TOKENIZER_PATH", ""),
        "alien_family_note": "Alien tokenizer built from the Qwen-family vocabulary permutation.",
    },
}

MODEL_SPECS = {
    name: spec for name, spec in MODEL_SPECS.items() if spec["alien_tokenizer"]
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


def get_plain_text(row: dict) -> str:
    if row.get("conversations"):
        messages = convert_conversations(row["conversations"])
        return "\n\n".join(msg["content"] for msg in messages if msg["content"])

    instruction = row.get("instruction", "")
    response = row.get("response", "")
    return "\n\n".join(part for part in [instruction, response] if part)


def summarize(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "total": sum(values),
    }


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_SPECS:
        raise ValueError("No Alien tokenizer configured. Set LLAMA_ALIEN_TOKENIZER_PATH or another *_ALIEN_TOKENIZER_PATH.")

    datasets_by_alias = {
        alias: load_dataset(spec["path"], split=spec["split"], cache_dir=HF_DATASET_CACHE)
        for alias, spec in DATASET_SPECS.items()
    }

    sample_rows = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_rows.append(json.loads(line))

    tokenizers = {}
    for model_name, spec in MODEL_SPECS.items():
        tokenizers[model_name] = {
            "original": AutoTokenizer.from_pretrained(
                spec["original_tokenizer"], local_files_only=LOCAL_FILES_ONLY
            ),
            "alien": AutoTokenizer.from_pretrained(
                spec["alien_tokenizer"], local_files_only=LOCAL_FILES_ONLY
            ),
        }

    results = {
        "sample_size": len(sample_rows),
        "sample_path": str(SAMPLE_PATH),
        "models": {},
    }

    for model_name, toks in tokenizers.items():
        overall = defaultdict(list)
        by_source = defaultdict(lambda: defaultdict(list))
        alien_operational_len_match = 0
        alien_operational_id_match = 0
        wrapper_operational_len_match = 0
        wrapper_operational_id_match = 0

        for idx, sample in enumerate(sample_rows, start=1):
            source = sample["source_dataset"]
            local_index = sample["local_index"]
            text = get_plain_text(datasets_by_alias[source][local_index])

            orig_ids = toks["original"].encode(text, add_special_tokens=False)
            alien_ids_on_plain = toks["alien"].encode(text, add_special_tokens=False)
            alien_text = toks["alien"].decode(orig_ids)
            alien_ids_on_alien_text = toks["alien"].encode(alien_text, add_special_tokens=False)
            base_ids_on_alien_text = toks["original"].encode(alien_text, add_special_tokens=False)

            orig_len = len(orig_ids)
            alien_plain_len = len(alien_ids_on_plain)
            alien_oper_len = len(alien_ids_on_alien_text)
            base_oper_len = len(base_ids_on_alien_text)

            overall["original_plain"].append(orig_len)
            overall["alien_on_plain"].append(alien_plain_len)
            overall["alien_on_alienized"].append(alien_oper_len)
            overall["original_on_alienized"].append(base_oper_len)

            by_source[source]["original_plain"].append(orig_len)
            by_source[source]["alien_on_plain"].append(alien_plain_len)
            by_source[source]["alien_on_alienized"].append(alien_oper_len)
            by_source[source]["original_on_alienized"].append(base_oper_len)

            if alien_oper_len == orig_len:
                alien_operational_len_match += 1
            if alien_ids_on_alien_text == orig_ids:
                alien_operational_id_match += 1
            if base_oper_len == orig_len:
                wrapper_operational_len_match += 1
            if base_ids_on_alien_text == orig_ids:
                wrapper_operational_id_match += 1

            if idx % 500 == 0:
                print(f"{model_name}: processed {idx}/{len(sample_rows)} samples")

        model_result = {
            "original_tokenizer_path": MODEL_SPECS[model_name]["original_tokenizer"],
            "alien_tokenizer_path": MODEL_SPECS[model_name]["alien_tokenizer"],
            "alien_family_note": MODEL_SPECS[model_name]["alien_family_note"],
            "overall": {key: summarize(values) for key, values in overall.items()},
            "by_source": {
                source: {key: summarize(values) for key, values in payload.items()}
                for source, payload in by_source.items()
            },
            "alien_operational_length_match_count": alien_operational_len_match,
            "alien_operational_length_match_rate": rate(alien_operational_len_match, len(sample_rows)),
            "alien_operational_id_match_count": alien_operational_id_match,
            "alien_operational_id_match_rate": rate(alien_operational_id_match, len(sample_rows)),
            "wrapper_operational_length_match_count": wrapper_operational_len_match,
            "wrapper_operational_length_match_rate": rate(wrapper_operational_len_match, len(sample_rows)),
            "wrapper_operational_id_match_count": wrapper_operational_id_match,
            "wrapper_operational_id_match_rate": rate(wrapper_operational_id_match, len(sample_rows)),
        }

        orig_mean = model_result["overall"]["original_plain"]["mean"]
        alien_plain_mean = model_result["overall"]["alien_on_plain"]["mean"]
        alien_oper_mean = model_result["overall"]["alien_on_alienized"]["mean"]
        wrapper_oper_mean = model_result["overall"]["original_on_alienized"]["mean"]

        model_result["deltas"] = {
            "alien_on_plain_vs_original": {
                "avg_token_diff": alien_plain_mean - orig_mean,
                "relative_diff_pct": ((alien_plain_mean / orig_mean) - 1.0) * 100 if orig_mean else 0.0,
                "total_token_diff": model_result["overall"]["alien_on_plain"]["total"]
                - model_result["overall"]["original_plain"]["total"],
            },
            "alien_on_alienized_vs_original": {
                "avg_token_diff": alien_oper_mean - orig_mean,
                "relative_diff_pct": ((alien_oper_mean / orig_mean) - 1.0) * 100 if orig_mean else 0.0,
                "total_token_diff": model_result["overall"]["alien_on_alienized"]["total"]
                - model_result["overall"]["original_plain"]["total"],
            },
            "original_on_alienized_vs_original": {
                "avg_token_diff": wrapper_oper_mean - orig_mean,
                "relative_diff_pct": ((wrapper_oper_mean / orig_mean) - 1.0) * 100 if orig_mean else 0.0,
                "total_token_diff": model_result["overall"]["original_on_alienized"]["total"]
                - model_result["overall"]["original_plain"]["total"],
            },
        }

        results["models"][model_name] = model_result

    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Original vs Alien Token Length Summary",
        "",
        "Definition:",
        "- `original_plain`: original plain text tokenized by the original tokenizer.",
        "- `alien_on_plain`: the same original plain text tokenized directly by the alien tokenizer.",
        "- `alien_on_alienized`: the alienized text obtained by `alien_tokenizer.decode(original_ids)` and then tokenized by the alien tokenizer.",
        "",
        "Interpretation:",
        "- `alien_on_alienized` measures the alien tokenizer on alienized text directly.",
        "- `original_on_alienized` follows the actual single-alien wrapper path used in Axolotl: `plain -> alien_text -> original/base tokenizer encode`.",
        "- `alien_on_plain` is included only as a tokenizer-drift reference on identical raw text.",
        "",
        f"- Sample size: `{len(sample_rows)}`",
        "",
    ]

    for model_name, payload in results["models"].items():
        op_delta = payload["deltas"]["alien_on_alienized_vs_original"]
        wrapper_delta = payload["deltas"]["original_on_alienized_vs_original"]
        plain_delta = payload["deltas"]["alien_on_plain_vs_original"]
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"- Original tokenizer: `{payload['original_tokenizer_path']}`",
                f"- Alien tokenizer: `{payload['alien_tokenizer_path']}`",
                f"- Note: {payload['alien_family_note']}",
                f"- Alien-tokenizer roundtrip length match rate: `{payload['alien_operational_length_match_rate'] * 100:.2f}%`",
                f"- Alien-tokenizer roundtrip exact ID match rate: `{payload['alien_operational_id_match_rate'] * 100:.2f}%`",
                f"- Wrapper-path length match rate: `{payload['wrapper_operational_length_match_rate'] * 100:.2f}%`",
                f"- Wrapper-path exact ID match rate: `{payload['wrapper_operational_id_match_rate'] * 100:.2f}%`",
                "",
                "| measure | avg tokens/sample | median | stdev | min | max | total tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| `original_plain` | {payload['overall']['original_plain']['mean']:.4f} | {payload['overall']['original_plain']['median']:.1f} | {payload['overall']['original_plain']['stdev']:.4f} | {payload['overall']['original_plain']['min']} | {payload['overall']['original_plain']['max']} | {payload['overall']['original_plain']['total']} |",
                f"| `alien_on_plain` | {payload['overall']['alien_on_plain']['mean']:.4f} | {payload['overall']['alien_on_plain']['median']:.1f} | {payload['overall']['alien_on_plain']['stdev']:.4f} | {payload['overall']['alien_on_plain']['min']} | {payload['overall']['alien_on_plain']['max']} | {payload['overall']['alien_on_plain']['total']} |",
                f"| `alien_on_alienized` | {payload['overall']['alien_on_alienized']['mean']:.4f} | {payload['overall']['alien_on_alienized']['median']:.1f} | {payload['overall']['alien_on_alienized']['stdev']:.4f} | {payload['overall']['alien_on_alienized']['min']} | {payload['overall']['alien_on_alienized']['max']} | {payload['overall']['alien_on_alienized']['total']} |",
                f"| `original_on_alienized` | {payload['overall']['original_on_alienized']['mean']:.4f} | {payload['overall']['original_on_alienized']['median']:.1f} | {payload['overall']['original_on_alienized']['stdev']:.4f} | {payload['overall']['original_on_alienized']['min']} | {payload['overall']['original_on_alienized']['max']} | {payload['overall']['original_on_alienized']['total']} |",
                "",
                "Deltas:",
                f"- `alien_on_plain` vs `original_plain`: avg diff `{plain_delta['avg_token_diff']:+.4f}`, relative diff `{plain_delta['relative_diff_pct']:+.2f}%`, total diff `{plain_delta['total_token_diff']:+}`",
                f"- `alien_on_alienized` vs `original_plain`: avg diff `{op_delta['avg_token_diff']:+.4f}`, relative diff `{op_delta['relative_diff_pct']:+.2f}%`, total diff `{op_delta['total_token_diff']:+}`",
                f"- `original_on_alienized` vs `original_plain`: avg diff `{wrapper_delta['avg_token_diff']:+.4f}`, relative diff `{wrapper_delta['relative_diff_pct']:+.2f}%`, total diff `{wrapper_delta['total_token_diff']:+}`",
                "",
            ]
        )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved Markdown to {MD_PATH}")
    for model_name, payload in results["models"].items():
        op_delta = payload["deltas"]["alien_on_alienized_vs_original"]
        wrapper_delta = payload["deltas"]["original_on_alienized_vs_original"]
        print(
            f"{model_name}: alien avg diff={op_delta['avg_token_diff']:+.4f}, "
            f"alien relative={op_delta['relative_diff_pct']:+.4f}%, "
            f"wrapper avg diff={wrapper_delta['avg_token_diff']:+.4f}, "
            f"wrapper relative={wrapper_delta['relative_diff_pct']:+.4f}%"
        )


if __name__ == "__main__":
    main()
