#!/usr/bin/env python3

import argparse
import glob
import json
import os
from statistics import mean


TASK_ORDER = [
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "truthfulqa_mc1",
    "gsm8k_cot",
]

METRIC_KEY = {
    "mmlu": "acc,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "gsm8k_cot": "exact_match,strict-match",
}

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "truthfulqa_mc1": "TruthfulQA MC1",
    "gsm8k_cot": "GSM8K CoT",
}


def load_run_metrics(run_root: str):
    metrics = {}
    for task in TASK_ORDER:
        pattern = os.path.join(run_root, "main", task, "*-shot", "*", "results_*.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            continue
        latest = paths[-1]
        payload = json.load(open(latest))
        result = payload["results"][task]
        key = METRIC_KEY[task]
        metrics[task] = {
            "metric_key": key,
            "value": float(result[key]),
            "result_path": latest,
        }
    return metrics


def percent(x: float) -> str:
    return f"{x * 100:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=run_root")
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        label, run_root = item.split("=", 1)
        runs[label] = {
            "run_root": run_root,
            "metrics": load_run_metrics(run_root),
        }

    summary = {}
    for label, data in runs.items():
        values = [data["metrics"][task]["value"] for task in TASK_ORDER if task in data["metrics"]]
        summary[label] = {
            "run_root": data["run_root"],
            "macro_average": mean(values) if values else None,
            "tasks": data["metrics"],
        }

    md_lines = [
        "# Data-Volume Main-Suite Comparison",
        "",
        "Scores are reported as percentages. Metric choice follows the paper's main classification/math setup:",
        "- `MMLU`, `WinoGrande`, `TruthfulQA MC1`: `acc`",
        "- `ARC-Easy`, `ARC-Challenge`, `HellaSwag`: `acc_norm`",
        "- `GSM8K CoT`: strict-match exact match",
        "",
        "| Task | " + " | ".join(runs.keys()) + " |",
        "| --- | " + " | ".join(["---:"] * len(runs)) + " |",
    ]

    for task in TASK_ORDER:
        row = [DISPLAY_NAME[task]]
        for label in runs:
            value = summary[label]["tasks"].get(task, {}).get("value")
            row.append(percent(value) if value is not None else "NA")
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.extend(
        [
            "",
            "| Macro Average | " + " | ".join(
                percent(summary[label]["macro_average"]) if summary[label]["macro_average"] is not None else "NA"
                for label in runs
            ) + " |",
            "",
            "## Notes",
            "",
        ]
    )

    for label in runs:
        md_lines.append(f"- `{label}`: `{summary[label]['run_root']}`")

    with open(args.md_out, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
