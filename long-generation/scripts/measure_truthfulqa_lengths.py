#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance

from transformers import AutoTokenizer


@dataclass
class RunSpec:
    backbone: str
    variant: str
    model_path: str
    sample_path: str
    result_path: str


RUNS = [
    RunSpec(
        backbone="Llama3-8B-Instruct",
        variant="original",
        model_path="/workspace/CACHE/MODELS/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/meta-llama/Meta-Llama-3-8B-Instruct/truthfulqa_gen/0-shot/__workspace__CACHE__MODELS__models--meta-llama--Meta-Llama-3-8B-Instruct__snapshots__8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/samples_truthfulqa_gen_latest.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/meta-llama/Meta-Llama-3-8B-Instruct/truthfulqa_gen/0-shot/__workspace__CACHE__MODELS__models--meta-llama--Meta-Llama-3-8B-Instruct__snapshots__8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/results_latest.json",
    ),
    RunSpec(
        backbone="Llama3-8B-Instruct",
        variant="alienlm",
        model_path="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/samples_truthfulqa_gen_2025-03-01T11-28-35.238123.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/results_2025-03-01T11-28-35.238123.json",
    ),
    RunSpec(
        backbone="Qwen2.5-7B-Instruct",
        variant="original",
        model_path="/workspace/CACHE/MODELS/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/truthfulqa_gen/0-shot/Qwen__Qwen2.5-7B-Instruct/samples_truthfulqa_gen_2025-04-07T20-47-44.335692.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/truthfulqa_gen/0-shot/Qwen__Qwen2.5-7B-Instruct/results_2025-04-07T20-47-44.335692.json",
    ),
    RunSpec(
        backbone="Qwen2.5-7B-Instruct",
        variant="alienlm",
        model_path="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/samples_truthfulqa_gen_2025-04-07T18-33-11.045150.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Qwen25-7b-Instruct-AlienLM-50-all-tokenizer-v3-32-llama/results_2025-04-07T18-33-11.045150.json",
    ),
    RunSpec(
        backbone="Gemma2-9B-it",
        variant="original",
        model_path="/workspace/CACHE/MODELS/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it/truthfulqa_gen/0-shot/google__gemma-2-9b-it/samples_truthfulqa_gen_2025-04-08T19-47-23.870757.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it/truthfulqa_gen/0-shot/google__gemma-2-9b-it/results_2025-04-08T19-47-23.870757.json",
    ),
    RunSpec(
        backbone="Gemma2-9B-it",
        variant="alienlm",
        model_path="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen",
        sample_path="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen/samples_truthfulqa_gen_2025-04-08T15-16-09.041683.jsonl",
        result_path="/workspace/data2/jaehee/AlienLM/outputs/Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen/truthfulqa_gen/0-shot/__workspace__data2__jaehee__AlienLM__outputs__Gemma2-9b-it-AlienLM-50-all-tokenizer-v3-32-qwen/results_2025-04-08T15-16-09.041683.json",
    ),
]


def resolve_latest(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path.name.endswith("_latest.json") or path.name.endswith("_latest.jsonl"):
        parent = path.parent
        stem = path.name.replace("_latest", "_*")
        matches = sorted(parent.glob(stem))
        if matches:
            return matches[-1]
    raise FileNotFoundError(path_str)


def summarize(values):
    return {
        "mean": mean(values),
        "variance": pvariance(values),
        "min": min(values),
        "max": max(values),
    }


def load_lengths(sample_path: Path, tokenizer):
    input_tok = []
    output_tok = []
    input_char = []
    output_char = []

    with sample_path.open() as f:
        for line in f:
            row = json.loads(line)
            prompt = row["arguments"]["gen_args_0"]["arg_0"]
            if row.get("filtered_resps"):
                output = row["filtered_resps"][0]
            else:
                output = row["resps"][0][0]

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


def load_scores(result_path: Path):
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


def main():
    out_dir = Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/long-generation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in RUNS:
        sample_path = resolve_latest(spec.sample_path)
        result_path = resolve_latest(spec.result_path)
        tokenizer = AutoTokenizer.from_pretrained(spec.model_path, trust_remote_code=True)
        lengths = load_lengths(sample_path, tokenizer)
        scores = load_scores(result_path)
        rows.append(
            {
                "backbone": spec.backbone,
                "variant": spec.variant,
                "sample_path": str(sample_path),
                "result_path": str(result_path),
                "model_path": spec.model_path,
                "scores": scores,
                "lengths": lengths,
            }
        )

    json_path = out_dir / "truthfulqa_gen_length_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    md_lines = [
        "# TruthfulQA Generation Length Summary",
        "",
        "All token lengths are measured with the tokenizer actually used by the evaluated model variant.",
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

    md_lines.extend(
        [
            "",
            "## Detailed Notes",
            "",
        ]
    )

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

    md_path = out_dir / "truthfulqa_gen_length_summary.md"
    md_path.write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
