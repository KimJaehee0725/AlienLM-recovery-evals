"""
Run the LLM decryption attack experiment
"""
import argparse
import os
from typing import List, Dict

from config import AttackConfig, EvaluationConfig
from llm_attack import LLMDeciphermentAttack
from llm_evaluator import LLMEvaluator
from utils import (
    load_test_data,
    prepare_known_pairs,
    save_results,
    create_comparison_table,
    print_sample_results
)


def run_single_experiment(
    data_file: str,
    n_shots: int = 0,
    n_test_samples: int = 100,
    use_llm_judge: bool = True,
    output_dir: str = "./results",
    max_concurrent_requests: int = None,
    attack_model: str = None,
    judge_model: str = None,
    non_parallel: bool = False
):
    """
    Run an experiment with a single few-shot size

    Args:
        data_file: JSONL data file (format {"original": "...", "alien": "..."})
        n_shots: Few-shot size (0 = zero-shot)
        n_test_samples: Number of test samples
        use_llm_judge: Whether to use the LLM judge
        output_dir: Directory to save results
        max_concurrent_requests: Limit on the number of concurrent requests (uses the default if None)
        non_parallel: Non-parallel attack mode (uses the non-parallel template; evaluation is identical to the parallel mode)
    """
    print(f"\n{'#'*80}")
    print(f"# LLM DECIPHERMENT ATTACK EXPERIMENT")
    if non_parallel and n_shots > 0:
        print(f"# Mode: NON-PARALLEL ATTACK (using non-parallel template for few-shot)")
        print(f"# Note: Zero-shot uses parallel template, few-shot uses non-parallel template")
    print(f"# Shots: {n_shots}")
    print(f"# Test samples: {n_test_samples}")
    if max_concurrent_requests:
        print(f"# Max concurrent requests: {max_concurrent_requests}")
    print(f"{'#'*80}\n")

    # 1. Configuration
    attack_config_kwargs = {"n_test_samples": n_test_samples}
    eval_config_kwargs = {"output_dir": output_dir}

    if max_concurrent_requests is not None:
        attack_config_kwargs["max_concurrent_requests"] = max_concurrent_requests
        eval_config_kwargs["max_concurrent_requests"] = max_concurrent_requests

    if attack_model is not None:
        attack_config_kwargs["model"] = attack_model

    if judge_model is not None:
        eval_config_kwargs["judge_model"] = judge_model

    attack_config = AttackConfig(**attack_config_kwargs)
    eval_config = EvaluationConfig(**eval_config_kwargs)

    # 2. Load data
    print("Loading data...")
    all_data = load_test_data(data_file)

    # 3. Split known pairs from the test data
    if n_shots > 0:
        known_pairs, test_data = prepare_known_pairs(
            all_data,
            n_pairs=n_shots,
            seed=42
        )
        test_data = test_data[:n_test_samples]
    else:
        known_pairs = None
        test_data = all_data[:n_test_samples]

    print(f"Test data: {len(test_data)} samples")
    if known_pairs:
        print(f"Known pairs: {len(known_pairs)}")

    # 4. Run the attack (use the non-parallel template if non-parallel mode is enabled)
    attacker = LLMDeciphermentAttack(attack_config)
    attack_results = attacker.run_experiment(test_data, known_pairs, non_parallel=non_parallel)

    # 5. Evaluate (proceed identically to the parallel mode - using the original text)
    evaluator = LLMEvaluator(eval_config)
    evaluated_results = evaluator.evaluate_batch(
        attack_results,
        use_llm_judge=use_llm_judge
    )

    # 6. Aggregate
    aggregated_metrics = evaluator.aggregate_results(evaluated_results)
    evaluator.print_summary(aggregated_metrics)

    # 7. Print samples
    print_sample_results(evaluated_results, n_samples=3)

    # 8. Save results
    final_results = {
        "config": {
            "non_parallel": non_parallel,
            "n_shots": n_shots,
            "n_test_samples": n_test_samples,
            "model": attack_config.model,
            "judge_model": eval_config.judge_model if use_llm_judge else None
        },
        "aggregated_metrics": aggregated_metrics,
        "detailed_results": evaluated_results
    }

    experiment_name = f"llm_attack_{n_shots}shot"
    save_results(final_results, output_dir, experiment_name)

    return final_results


def run_progressive_experiment(
    data_file: str,
    few_shot_sizes: List[int] = None,
    n_test_samples: int = 100,
    use_llm_judge: bool = True,
    output_dir: str = "./results",
    max_concurrent_requests: int = None,
    attack_model: str = None,
    judge_model: str = None,
    non_parallel: bool = False
):
    """
    Progressive experiment over several few-shot sizes

    Args:
        data_file: JSONL data file (format {"original": "...", "alien": "..."})
        few_shot_sizes: List of few-shot sizes (e.g. [0, 1, 3, 5, 10])
        n_test_samples: Number of test samples
        use_llm_judge: Whether to use the LLM judge
        output_dir: Directory to save results
        max_concurrent_requests: Limit on the number of concurrent requests (uses the default if None)
        non_parallel: Non-parallel attack mode (uses the non-parallel template; evaluation is identical to the parallel mode)
    """
    if few_shot_sizes is None:
        few_shot_sizes = [0, 1, 3, 5, 10]

    print(f"\n{'#'*80}")
    print(f"# PROGRESSIVE LLM ATTACK EXPERIMENT")
    if non_parallel:
        print(f"# Mode: NON-PARALLEL ATTACK (using non-parallel template for few-shot)")
        print(f"# Note: Zero-shot uses parallel template, few-shot uses non-parallel template")
    print(f"# Few-shot sizes: {few_shot_sizes}")
    print(f"# Test samples: {n_test_samples}")
    if max_concurrent_requests:
        print(f"# Max concurrent requests: {max_concurrent_requests}")
    print(f"{'#'*80}\n")

    all_results = {}

    for n_shots in few_shot_sizes:
        result = run_single_experiment(
            data_file=data_file,
            n_shots=n_shots,
            n_test_samples=n_test_samples,
            use_llm_judge=use_llm_judge,
            output_dir=output_dir,
            max_concurrent_requests=max_concurrent_requests,
            attack_model=attack_model,
            judge_model=judge_model,
            non_parallel=non_parallel
        )

        all_results[n_shots] = result["aggregated_metrics"]

    # Build and print the comparison table
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS FEW-SHOT SIZES")
    print(f"{'='*80}\n")

    comparison_table = create_comparison_table(all_results)
    print(comparison_table)

    # Save the full results
    progressive_results = {
        "config": {
            "non_parallel": non_parallel,
            "few_shot_sizes": few_shot_sizes,
            "n_test_samples": n_test_samples
        },
        "results_by_shots": all_results,
        "comparison_table": comparison_table
    }

    save_results(progressive_results, output_dir, "progressive_experiment")

    return progressive_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Decipherment Attack Experiment"
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSONL data file (each line: {\"original\": \"...\", \"alien\": \"...\"})"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "progressive"],
        default="progressive",
        help="Experiment mode: single or progressive"
    )

    parser.add_argument(
        "--n_shots",
        type=int,
        default=0,
        help="Number of few-shot examples (for single mode)"
    )

    parser.add_argument(
        "--few_shot_sizes",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5, 10],
        help="Few-shot sizes for progressive mode"
    )

    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=100,
        help="Number of test samples"
    )

    parser.add_argument(
        "--no_llm_judge",
        action="store_true",
        help="Disable LLM-as-a-judge evaluation"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./llm_attack_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=None,
        help="Maximum number of concurrent requests (default: from config)"
    )

    parser.add_argument(
        "--attack_model",
        type=str,
        default=None,
        help="Attack model name (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini'). Default: from config"
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Judge model name (e.g., 'gpt-5.1', 'gpt-4o', 'gpt-4o-mini'). Default: from config"
    )

    parser.add_argument(
        "--non_parallel",
        action="store_true",
        help="Use non-parallel attack template (indicates leaked encrypted data). Evaluation uses reference text same as parallel mode."
    )

    args = parser.parse_args()

    # Run the experiment
    if args.mode == "single":
        run_single_experiment(
            data_file=args.data_file,
            n_shots=args.n_shots,
            n_test_samples=args.n_test_samples,
            use_llm_judge=not args.no_llm_judge,
            output_dir=args.output_dir,
            max_concurrent_requests=args.max_concurrent_requests,
            attack_model=args.attack_model,
            judge_model=args.judge_model,
            non_parallel=args.non_parallel
        )
    else:  # progressive
        run_progressive_experiment(
            data_file=args.data_file,
            few_shot_sizes=args.few_shot_sizes,
            n_test_samples=args.n_test_samples,
            use_llm_judge=not args.no_llm_judge,
            output_dir=args.output_dir,
            max_concurrent_requests=args.max_concurrent_requests,
            attack_model=args.attack_model,
            judge_model=args.judge_model,
            non_parallel=args.non_parallel
        )


if __name__ == "__main__":
    main()
