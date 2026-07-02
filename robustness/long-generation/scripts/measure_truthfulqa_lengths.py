#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance

from transformers import AutoTokenizer


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = EXP_DIR / "outputs"
DEFAULT_RESULTS_DIR = EXP_DIR / "results"
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"


@dataclass
class RunSpec:
    backbone: str
    variant: str
    model_path: str
    sample_path: Path
    result_path: Path


MODEL_DEFAULTS = {
    "llama_original": {
        "backbone": "Llama3-8B-Instruct",
        "variant": "original",
        "model_path": os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"),
    },
    "llama_alien": {
        "backbone": "Llama3-8B-Instruct",
        "variant": "alienlm",
        "model_path": os.environ.get(
            "LLAMA_ALIEN_MODEL_PATH", "dsba-lab/llama3-8b-instruct-alienlm-full"
        ),
    },
    "qwen_original": {
        "backbone": "Qwen2.5-7B-Instruct",
        "variant": "original",
        "model_path": os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
    },
    "qwen_alien": {
        "backbone": "Qwen2.5-7B-Instruct",
        "variant": "alienlm",
        "model_path": os.environ.get("QWEN_ALIEN_MODEL_PATH", ""),
    },
    "gemma_original": {
        "backbone": "Gemma2-9B-it",
        "variant": "original",
        "model_path": os.environ.get("GEMMA_MODEL_PATH", "google/gemma-2-9b-it"),
    },
    "gemma_alien": {
        "backbone": "Gemma2-9B-it",
        "variant": "alienlm",
        "model_path": os.environ.get("GEMMA_ALIEN_MODEL_PATH", ""),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure TruthfulQA generation input/output lengths.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("LONG_GENERATION_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
        help="Root containing <model_key>/truthfulqa_gen/0-shot/... outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", DEFAULT_RESULTS_DIR)),
        help="Directory where summary files are written.",
    )
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=Path(os.environ["TRUTHFULQA_RUNS_JSON"])
        if os.environ.get("TRUTHFULQA_RUNS_JSON")
        else None,
        help="Optional JSON list of explicit run specs.",
    )
    return parser.parse_args()


def latest_match(pattern: str) -> Path | None:
    matches = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    return matches[-1] if matches else None


def discover_runs(output_root: Path) -> list[RunSpec]:
    runs = []
    for model_key, meta in MODEL_DEFAULTS.items():
        if not meta["model_path"]:
            continue
        run_root = output_root / model_key / "truthfulqa_gen" / "0-shot"
        sample_path = latest_match(str(run_root / "**" / "samples_truthfulqa_gen*.jsonl"))
        result_path = latest_match(str(run_root / "**" / "results*.json"))
        if sample_path is None or result_path is None:
            continue
        runs.append(
            RunSpec(
                backbone=meta["backbone"],
                variant=meta["variant"],
                model_path=meta["model_path"],
                sample_path=sample_path,
                result_path=result_path,
            )
        )
    return runs


def load_explicit_runs(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text())
    return [
        RunSpec(
            backbone=row["backbone"],
            variant=row["variant"],
            model_path=row["model_path"],
            sample_path=Path(row["sample_path"]),
            result_path=Path(row["result_path"]),
        )
        for row in payload
    ]


def summarize(values: list[int]) -> dict:
    return {
        "mean": mean(values),
        "variance": pvariance(values),
        "min": min(values),
        "max": max(values),
    }


def load_lengths(sample_path: Path, tokenizer) -> dict:
    input_tok = []
    output_tok = []
    input_char = []
    output_char = []

    with sample_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row["arguments"]["gen_args_0"]["arg_0"]
            output = row["filtered_resps"][0] if row.get("filtered_resps") else row["resps"][0][0]

            input_tok.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
            output_tok.append(len(tokenizer.encode(output, add_special_tokens=False)))
            input_char.append(len(prompt))
            output_char.append(len(output))

    return {
        "count": len(input_tok),
        "input_tokens": summarize(input_tok),
        "output_tokens": summarize(output_tok),
        "input_chars": summarize(input_char),
        "output_chars": summarize(output_char),
    }


def load_scores(result_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    res = result["results"]["truthfulqa_gen"]
    return {
        "bleu_acc": res["bleu_acc,none"],
        "rouge1_acc": res["rouge1_acc,none"],
        "rouge2_acc": res["rouge2_acc,none"],
        "rougeL_acc": res["rougeL_acc,none"],
        "bleu_max": res["bleu_max,none"],
        "rougeL_max": res["rougeL_max,none"],
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    runs = load_explicit_runs(args.runs_json) if args.runs_json else discover_runs(args.output_root)
    if not runs:
        raise FileNotFoundError(
            "No TruthfulQA runs found. Set LONG_GENERATION_OUTPUT_ROOT or pass --runs-json."
        )

    rows = []
    for spec in runs:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_path,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )
        lengths = load_lengths(spec.sample_path, tokenizer)
        scores = load_scores(spec.result_path)
        rows.append(
            {
                "backbone": spec.backbone,
                "variant": spec.variant,
                "sample_path": str(spec.sample_path),
                "result_path": str(spec.result_path),
                "model_path": spec.model_path,
                "scores": scores,
                "lengths": lengths,
            }
        )

    json_path = args.results_dir / "truthfulqa_gen_length_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    md_lines = [
        "# TruthfulQA Generation Length Summary",
        "",
        "All token lengths are measured with the tokenizer used by the evaluated model variant.",
        "",
        "| Backbone | Variant | BLEU Acc | ROUGE-L Acc | Mean Input Tok | Var Input Tok | Mean Output Tok | Var Output Tok |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {backbone} | {variant} | {bleu_acc:.4f} | {rougeL_acc:.4f} | {mi:.2f} | {vi:.2f} | {mo:.2f} | {vo:.2f} |".format(
                backbone=row["backbone"],
                variant=row["variant"],
                bleu_acc=row["scores"]["bleu_acc"],
                rougeL_acc=row["scores"]["rougeL_acc"],
                mi=row["lengths"]["input_tokens"]["mean"],
                vi=row["lengths"]["input_tokens"]["variance"],
                mo=row["lengths"]["output_tokens"]["mean"],
                vo=row["lengths"]["output_tokens"]["variance"],
            )
        )

    md_lines.extend(["", "## Detailed Notes", ""])
    for row in rows:
        md_lines.extend(
            [
                f"### {row['backbone']} / {row['variant']}",
                "",
                f"- Sample count: {row['lengths']['count']}",
                f"- BLEU Acc: {row['scores']['bleu_acc']:.4f}",
                f"- ROUGE-L Acc: {row['scores']['rougeL_acc']:.4f}",
                f"- Mean input tokens: {row['lengths']['input_tokens']['mean']:.2f}",
                f"- Input token variance: {row['lengths']['input_tokens']['variance']:.2f}",
                f"- Mean output tokens: {row['lengths']['output_tokens']['mean']:.2f}",
                f"- Output token variance: {row['lengths']['output_tokens']['variance']:.2f}",
                f"- Mean input chars: {row['lengths']['input_chars']['mean']:.2f}",
                f"- Input char variance: {row['lengths']['input_chars']['variance']:.2f}",
                f"- Mean output chars: {row['lengths']['output_chars']['mean']:.2f}",
                f"- Output char variance: {row['lengths']['output_chars']['variance']:.2f}",
                "",
            ]
        )

    md_path = args.results_dir / "truthfulqa_gen_length_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved JSON to {json_path}")
    print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
    main()
