import json
from pathlib import Path


ROOT = Path("/workspace/codes/AlienLMv2/icml2026-rebuttal/collaborative-attack/results/known_pairs_sweep_n3_c0p01")


def extract_k(summary: Path) -> int:
    for part in summary.parts:
        if part.startswith("k") and part[1:].isdigit():
            return int(part[1:])
    return -1


def main() -> None:
    rows = []
    for summary in sorted(ROOT.rglob("summary.json"), key=extract_k):
        data = json.loads(summary.read_text())
        rows.append(
            {
                "k_known_pairs": extract_k(summary),
                "token_accuracy": data.get("token_accuracy"),
                "correct_tokens": data.get("correct_tokens"),
                "total_tokens": data.get("total_tokens"),
                "known_mappings_size": data.get("known_mappings_size"),
                "correct_mappings": data.get("correct_mappings"),
                "total_mappings": data.get("total_mappings"),
                "bijection_recovery_rate": data.get("bijection_recovery_rate"),
            }
        )

    print("| k known pairs | token acc | known mappings | correct mappings | total mappings | bijection recovery |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        token_acc = row["token_accuracy"] * 100 if row["token_accuracy"] is not None else 0.0
        bij_rate = row["bijection_recovery_rate"] * 100 if row["bijection_recovery_rate"] is not None else 0.0
        print(
            f"| {row['k_known_pairs']} | {token_acc:.4f}% | {row['known_mappings_size']} | "
            f"{row['correct_mappings']} | {row['total_mappings']} | {bij_rate:.4f}% |"
        )


if __name__ == "__main__":
    main()
