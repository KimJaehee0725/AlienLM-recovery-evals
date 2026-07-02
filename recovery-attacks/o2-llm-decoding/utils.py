"""
Utility functions
"""
import json
import os
import random
from typing import List, Dict, Tuple
from datetime import datetime


def load_test_data(data_file: str,
                  n_samples: int = None,
                  non_parallel: bool = False) -> List[Dict[str, str]]:
    """
    Load test data (JSONL format)

    Args:
        data_file: Path to the JSONL file
            - Parallel corpus: each line is {"original": "...", "alien": "..."}
            - Non-parallel corpus: each line is {"alien": "..."} (non_parallel=True)
        n_samples: Number of samples (if None, use all)
        non_parallel: Non-parallel corpus mode (encrypted text only, no original)

    Returns:
        Parallel: [{"original": "...", "alien": "..."}, ...]
        Non-parallel: [{"alien": "..."}, ...]
    """
    data = []

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                if non_parallel:
                    # Non-parallel corpus: the "alien" key is required, "original" is optional (for evaluation)
                    if "alien" in item:
                        data_item = {"alien": item["alien"]}
                        # Store the original together if available, for evaluation
                        if "original" in item:
                            data_item["original"] = item["original"]
                        data.append(data_item)
                    else:
                        print(f"Warning: Skipping invalid line (missing 'alien' key): {line[:100]}...")
                else:
                    # Parallel corpus: both the "original" and "alien" keys must be present
                    if "original" in item and "alien" in item:
                        data.append({
                            "original": item["original"],
                            "alien": item["alien"]
                        })
                    else:
                        print(f"Warning: Skipping invalid line (missing keys): {line[:100]}...")
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

    if n_samples:
        data = data[:n_samples]

    return data


def prepare_known_pairs(all_data: List[Dict[str, str]],
                       n_pairs: int,
                       seed: int = 42,
                       non_parallel: bool = False) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Split into known pairs (for few-shot) and test data

    Args:
        all_data: All data
        n_pairs: Number of known pairs
        seed: Random seed
        non_parallel: Non-parallel corpus mode (no original text)

    Returns:
        (known_pairs, test_data)
    """
    random.seed(seed)
    shuffled = all_data.copy()
    random.shuffle(shuffled)

    if non_parallel:
        # Non-parallel corpus: known pairs cannot be created (there are no originals)
        # A separate parallel corpus is required to use few-shot
        print("Warning: Non-parallel corpus mode. Known pairs cannot be created without original texts.")
        print("Few-shot experiments require parallel corpus for known pairs.")
        known_pairs = []
    else:
        known_pairs = [
            (d["original"], d["alien"])
            for d in shuffled[:n_pairs]
        ]

    test_data = shuffled[n_pairs:]

    return known_pairs, test_data


def save_results(results: Dict,
                output_dir: str,
                experiment_name: str = None):
    """
    Save results

    Args:
        results: Results to save
        output_dir: Output directory
        experiment_name: Experiment name
    """
    os.makedirs(output_dir, exist_ok=True)

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"

    filepath = os.path.join(output_dir, f"{experiment_name}.json")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")


def load_results(filepath: str) -> Dict:
    """Load saved results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_comparison_table(all_results: Dict[int, Dict]) -> str:
    """
    Create a comparison table across several few-shot sizes

    Args:
        all_results: {n_shots: aggregated_metrics}

    Returns:
        Markdown table string
    """
    headers = ["N-shots", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "LLM-Judge"]

    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * len(headers)) + "|\n"

    for n_shots in sorted(all_results.keys()):
        metrics = all_results[n_shots]

        row = [
            str(n_shots),
            f"{metrics.get('bleu_mean', 0):.2f}",
            f"{metrics.get('rouge1_f_mean', 0):.2f}",
            f"{metrics.get('rouge2_f_mean', 0):.2f}",
            f"{metrics.get('rougeL_f_mean', 0):.2f}",
            f"{metrics.get('llm_overall_mean', 0):.2f}"
        ]

        table += "| " + " | ".join(row) + " |\n"

    return table


def print_sample_results(results: List[Dict], n_samples: int = 3, non_parallel: bool = False):
    """Print sample results"""
    print(f"\n{'='*80}")
    print(f"SAMPLE RESULTS (showing {n_samples} examples)")
    print(f"{'='*80}\n")

    for i, result in enumerate(results[:n_samples]):
        print(f"Sample {i+1}:")
        if not non_parallel and "original" in result:
            print(f"  Original:  {result['original'][:100]}...")
        print(f"  Alien:     {result['alien'][:100]}...")
        print(f"  Decrypted: {result['decrypted'][:100]}...")

        if "evaluation" in result:
            eval_dict = result["evaluation"]
            if not non_parallel:
                print(f"  BLEU:      {eval_dict.get('bleu', 0):.2f}")
                print(f"  ROUGE-L:   {eval_dict.get('rougeL_f', 0):.2f}")
            if "llm_overall" in eval_dict:
                print(f"  LLM Judge: {eval_dict.get('llm_overall', 0):.2f}/3")

        print()
