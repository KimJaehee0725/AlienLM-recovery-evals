#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack")
RESULT_ROOT = ROOT / "results" / "seedpool_sweep_n3_c0p95_occ3"
OUTPUT_MD = ROOT / "notes" / "seedpool_sweep_n3_c0p95_occ3.md"


def load_summary(mode: str, k_value: int) -> dict:
    path = RESULT_ROOT / mode / f"k{k_value}" / "summary.json"
    with path.open() as fh:
        return json.load(fh)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    rows = []
    for k_value in [1000, 5000, 10000]:
        shared = load_summary("shared", k_value)
        mixed = load_summary("mixed", k_value)
        rows.append((k_value, shared, mixed))

    lines = []
    lines.append("# Mixed-Seed Collaborative Attack Sweep")
    lines.append("")
    lines.append("Comparison between pooled leakage under a shared seed and pooled leakage across three mixed seeds.")
    lines.append("")
    lines.append("| k known pairs | setting | aligned pairs | known mappings | known-only coverage | known-only accuracy | full attack token acc | recovered mappings |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k_value, shared, mixed in rows:
        for setting, payload in [("shared", shared), ("mixed", mixed)]:
            lines.append(
                f"| {k_value} | {setting} | "
                f"{payload['aligned_pairs']}/{payload['total_pairs']} | "
                f"{payload['known_mappings_size']} | "
                f"{fmt_pct(payload['known_only_coverage'])} | "
                f"{fmt_pct(payload['known_only_accuracy'])} | "
                f"{fmt_pct(payload['token_accuracy'])} | "
                f"{payload['recovered_mapping_size']} |"
            )
    lines.append("")
    lines.append("## Delta: Shared vs Mixed")
    lines.append("")
    lines.append("| k known pairs | metric | shared | mixed | delta (mixed - shared) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for k_value, shared, mixed in rows:
        for key, label in [
            ("known_mappings_size", "known mappings"),
            ("known_only_coverage", "known-only coverage"),
            ("known_only_accuracy", "known-only accuracy"),
            ("token_accuracy", "full attack token acc"),
        ]:
            delta = mixed[key] - shared[key]
            if "size" in key:
                shared_str = str(shared[key])
                mixed_str = str(mixed[key])
                delta_str = f"{delta:+.0f}"
            else:
                shared_str = fmt_pct(shared[key])
                mixed_str = fmt_pct(mixed[key])
                delta_str = f"{delta * 100:+.2f}pp"
            lines.append(
                f"| {k_value} | {label} | {shared_str} | {mixed_str} | {delta_str} |"
            )
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append(
        "Within the same leakage budget, compare shared and mixed row pairs horizontally rather than comparing different k values vertically, because the held-out split changes with k."
    )
    lines.append(
        "At k=1000 and k=5000, mixed-seed pooling is consistently weaker than shared-seed pooling in both known-only coverage and full attack token accuracy."
    )
    lines.append(
        "At k=10000, the gap largely closes and slightly reverses, so the cautious conclusion is that distinct tenant seeds damp collaborative leakage at low-to-moderate budgets, but they do not eliminate risk once leakage becomes very large."
    )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
