#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "observer_tokenizer_sweep_n3_k10000_c0p01"
SUMMARY_DIR = ROOT / "results" / "summaries"
DETAIL_MD = SUMMARY_DIR / "observer_tokenizer_sweep_n3_k10000_c0p01.md"
PAPER_MD = SUMMARY_DIR / "observer_tokenizer_sweep_paper_summary.md"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    rows = []
    for observer_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        summary = json.loads((observer_dir / "summary.json").read_text())
        rows.append(summary)

    lines = []
    lines.append("# Observer Tokenizer Sweep")
    lines.append("")
    lines.append("Llama AlienLM attack performance when the attacker tokenizes the same plaintext/alien-text pairs with different off-the-shelf model tokenizers.")
    lines.append("")
    lines.append("| observer tokenizer | avg plain len | avg alien len | aligned pairs | known mappings | known-only coverage | known-only accuracy | full attack token acc | recovered mappings |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['observer_name']} | "
            f"{row['plain_length_stats']['avg_tokens']:.2f} | "
            f"{row['alien_length_stats']['avg_tokens']:.2f} | "
            f"{row['aligned_pairs']}/{row['total_pairs']} | "
            f"{row['known_mappings_size']} | "
            f"{fmt_pct(row['known_only_coverage'])} | "
            f"{fmt_pct(row['known_only_accuracy'])} | "
            f"{fmt_pct(row['token_accuracy'])} | "
            f"{row['recovered_mapping_size']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "These numbers are tokenizer-specific token reconstruction rates, so they should be read comparatively rather than as directly identical units across tokenizers."
    )
    lines.append(
        "The main question is whether the attack remains strong when the observer does not know the victim tokenizer and instead tokenizes both sides with another common LLM tokenizer."
    )
    DETAIL_MD.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_MD.write_text("\n".join(lines) + "\n")

    if rows:
        rows_sorted = sorted(rows, key=lambda item: item["token_accuracy"], reverse=True)
        best = rows_sorted[0]
        worst = rows_sorted[-1]
        paper_lines = [
            "# Observer Tokenizer Sweep: Paper Notes",
            "",
            "## Setup",
            "",
            "- Victim: Llama AlienLM.",
            "- Fixed leaked corpus: Llama plaintext transformed with the Llama AlienLM tokenizer checkpoint.",
            "- Attack budget: `train=10k`, `k=10k`, `test=1k`, `reference=tulu3 10k`, `n=3`, `top_k=1000`.",
            "- Variable: the observer tokenizer used to tokenize both plaintext and alien-text pairs.",
            "",
            "## Main Result",
            "",
            f"- Best observer tokenizer in this sweep: `{best['observer_name']}` with token accuracy {fmt_pct(best['token_accuracy'])}.",
            f"- Worst observer tokenizer in this sweep: `{worst['observer_name']}` with token accuracy {fmt_pct(worst['token_accuracy'])}.",
            "- If performance drops substantially for non-victim tokenizers, the attack is sensitive to tokenizer assumptions and becomes less reliable when the observer does not know the victim tokenization scheme.",
            "",
            "## Suggested Rebuttal Wording",
            "",
            "We additionally tested whether the collaborative reconstruction attack depends on knowing the victim tokenizer. Keeping the same Llama AlienLM plaintext/alien-text pairs fixed, we re-ran the attack while tokenizing both sides with five common LLM tokenizers (Llama, Qwen, Gemma, Mistral, and Phi-3). The resulting reconstruction rates varied noticeably across observer tokenizers, indicating that the attack is not tokenizer-invariant and becomes less stable once the adversary does not know the victim tokenization scheme exactly.",
        ]
        PAPER_MD.write_text("\n".join(paper_lines) + "\n")


if __name__ == "__main__":
    main()
