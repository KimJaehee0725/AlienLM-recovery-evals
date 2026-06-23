#!/usr/bin/env python3

import glob
import json
from pathlib import Path


RESULT_GLOB = "/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/long-generation/*/longbench_core/0-shot/*/results_*.json"
ROOT = Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/long-generation/results")
DETAIL_MD = ROOT / "longbench_core_analysis_detailed.md"
PAPER_MD = ROOT / "longbench_core_paper_notes.md"


MODEL_ORDER = [
    ("llama", "Llama-3-8B-Instruct"),
    ("qwen", "Qwen2.5-7B-Instruct"),
    ("gemma", "Gemma-2-9b-it"),
]


def pct_delta(new: float, base: float) -> float:
    return (new - base) / base if base else 0.0


def load_rows():
    rows = {}
    for path in sorted(glob.glob(RESULT_GLOB)):
        payload = json.loads(Path(path).read_text())
        run_name = path.split("/long-generation/")[1].split("/")[0]
        backbone = run_name.split("_")[0]
        variant = "alien" if run_name.endswith("_alien") else "original"
        rows[(backbone, variant)] = {
            "path": path,
            "gov": payload["results"]["longbench_gov_report_e"]["rouge_score,none"],
            "gov_stderr": payload["results"]["longbench_gov_report_e"]["rouge_score_stderr,none"],
            "qasper": payload["results"]["longbench_qasper_e"]["qa_f1_score,none"],
            "qasper_stderr": payload["results"]["longbench_qasper_e"]["qa_f1_score_stderr,none"],
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "model_name": payload.get("model_name"),
        }
    return rows


def main():
    rows = load_rows()

    detail_lines = [
        "# LongBench Core Analysis",
        "",
        "Tasks: `longbench_gov_report_e` and `longbench_qasper_e`.",
        "Metric types differ across tasks (`ROUGE` vs `QA F1`), so the safest comparison is original vs AlienLM within each task and backbone.",
        "",
        "## Raw Scores",
        "",
        "| Backbone | Variant | GovReport ROUGE | Qasper F1 | Result File |",
        "| --- | --- | ---: | ---: | --- |",
    ]

    for backbone, display_name in MODEL_ORDER:
        for variant in ["original", "alien"]:
            row = rows[(backbone, variant)]
            rel_path = row["path"]
            detail_lines.append(
                f"| {display_name} | {variant} | "
                f"{row['gov']:.4f} ± {row['gov_stderr']:.4f} | "
                f"{row['qasper']:.4f} ± {row['qasper_stderr']:.4f} | "
                f"{rel_path} |"
            )

    detail_lines.extend(
        [
            "",
            "## Original vs AlienLM Delta",
            "",
            "| Backbone | GovReport delta | GovReport relative | Qasper delta | Qasper relative |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    paper_lines = [
        "# LongBench Core: Paper Notes",
        "",
        "## Setup",
        "",
        "- Tasks: `gov_report_e` and `qasper_e` from LongBench.",
        "- Models: Llama, Qwen, Gemma original checkpoints and their AlienLM-adapted counterparts.",
        "- Evaluation: `vLLM`, `0-shot`, single-GPU inference.",
        "",
        "## Main Numbers",
        "",
        "| Backbone | GovReport original | GovReport AlienLM | Qasper original | Qasper AlienLM |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    takeaway_lines = []
    for backbone, display_name in MODEL_ORDER:
        orig = rows[(backbone, "original")]
        alien = rows[(backbone, "alien")]
        gov_delta = alien["gov"] - orig["gov"]
        qasper_delta = alien["qasper"] - orig["qasper"]
        gov_rel = pct_delta(alien["gov"], orig["gov"])
        qasper_rel = pct_delta(alien["qasper"], orig["qasper"])

        detail_lines.append(
            f"| {display_name} | "
            f"{gov_delta:+.4f} | {gov_rel * 100:+.2f}% | "
            f"{qasper_delta:+.4f} | {qasper_rel * 100:+.2f}% |"
        )

        paper_lines.append(
            f"| {display_name} | {orig['gov']:.4f} | {alien['gov']:.4f} | {orig['qasper']:.4f} | {alien['qasper']:.4f} |"
        )

        takeaway_lines.append(
            f"- {display_name}: GovReport {gov_delta:+.4f} ({gov_rel * 100:+.2f}%), "
            f"Qasper {qasper_delta:+.4f} ({qasper_rel * 100:+.2f}%)."
        )

    detail_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `gov_report_e` drops for all three backbones after AlienLM adaptation.",
            "- `qasper_e` drops for Qwen and Gemma, but Llama improves noticeably on this task.",
            "- So the long-generation story is not simply that AlienLM always fails on free-form tasks. A safer statement is that performance remains task-dependent, but the degradation clearly extends beyond MCQA-only settings.",
        ]
    )

    paper_lines.extend(
        [
            "",
            "## Delta Summary",
            "",
            *takeaway_lines,
            "",
            "## Suggested Wording",
            "",
            "We additionally evaluated two LongBench generation tasks, `gov_report_e` and `qasper_e`, across Llama, Qwen, and Gemma backbones. `gov_report_e` degraded consistently for all three models after AlienLM adaptation, while `qasper_e` degraded for Qwen and Gemma but improved for Llama. This indicates that the performance gap is not confined to multiple-choice benchmarks, but also that free-form generation under AlienLM remains task-dependent rather than uniformly degraded.",
        ]
    )

    DETAIL_MD.write_text("\n".join(detail_lines) + "\n")
    PAPER_MD.write_text("\n".join(paper_lines) + "\n")


if __name__ == "__main__":
    main()
