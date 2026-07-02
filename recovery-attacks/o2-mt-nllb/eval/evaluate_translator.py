#!/usr/bin/env python3
"""
Translator model evaluation script.

Uses a trained translator model to decode the evaluation data and score it.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add the path so the evaluation modules in the sibling "llm" folder can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "llm"))
from llm_evaluator import LLMEvaluator
from config import EvaluationConfig


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize DDP.

    Args:
        rank: rank of the current process
        world_size: total number of processes
        backend: communication backend (nccl for GPU)

    Note:
        When launched with torchrun the environment variables are set
        automatically, so the already-set environment variables are used.
    """
    # Check the environment variables set by torchrun; fall back to defaults otherwise
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'

    if 'MASTER_PORT' not in os.environ:
        # Find an available port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        os.environ['MASTER_PORT'] = str(port)
        if rank == 0:
            print(f"Using MASTER_PORT: {port}")

    # Initialize DDP
    # When launched with torchrun the environment variables are already set
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Set the GPU for the current process
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)


def cleanup_ddp():
    """Tear down DDP."""
    dist.destroy_process_group()


def load_model_and_tokenizer(model_path: str, rank: int = 0, use_ddp: bool = False):
    """
    Load the trained model and tokenizer.

    Args:
        model_path: path where the trained model is stored
        rank: rank of the current process (when using DDP)
        use_ddp: whether DDP is in use

    Returns:
        model, tokenizer
    """
    if rank == 0 or not use_ddp:
        print(f"Loading model from {model_path}...")

    # Load the tokenizer (identical across all processes)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Set the device
    if use_ddp:
        device = f"cuda:{rank}"
        model = model.to(device)
        # Wrap with DDP
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        if rank == 0:
            print(f"Model wrapped with DDP on GPU {rank}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if rank == 0 or not use_ddp:
            print(f"Model loaded on {device}")

    model.eval()

    # Set the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if rank == 0 or not use_ddp:
        print(f"Model loaded successfully")

    return model, tokenizer, device


def load_eval_data(data_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load the evaluation data.

    Args:
        data_path: path to the evaluation data file (JSONL format)
        max_samples: maximum number of samples (load everything if None)

    Returns:
        list of evaluation data
    """
    print(f"Loading evaluation data from {data_path}...")
    if max_samples is not None:
        print(f"Limiting to {max_samples} samples")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples is not None and len(data) >= max_samples:
                    break

    print(f"Loaded {len(data)} samples")
    return data


def decrypt_texts(
    model,
    tokenizer,
    encrypted_texts: List[str],
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 8,
    rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
) -> List[str]:
    """
    Decode (translate) the encoded texts.

    Args:
        model: trained translation model (handled automatically if wrapped in DDP)
        tokenizer: tokenizer
        encrypted_texts: list of encoded texts
        device: device to use
        max_length: maximum generation length
        batch_size: batch size
        rank: rank of the current process (when using DDP)
        world_size: total number of processes (when using DDP)
        use_ddp: whether DDP is in use

    Returns:
        list of decoded texts
    """
    if rank == 0 or not use_ddp:
        print(f"Decrypting {len(encrypted_texts)} texts...")

    # Distribute the data when using DDP
    if use_ddp and world_size > 1:
        # Compute the data index range each process handles
        chunk_size = len(encrypted_texts) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(encrypted_texts)
        process_texts = encrypted_texts[start_idx:end_idx]
        if rank == 0:
            print(f"DDP: Rank {rank} processing texts {start_idx} to {end_idx}")
    else:
        process_texts = encrypted_texts

    decrypted_texts = []

    # Process in batches
    progress_bar = tqdm(range(0, len(process_texts), batch_size), desc=f"Decrypting [Rank {rank}]") if rank == 0 or not use_ddp else range(0, len(process_texts), batch_size)

    for i in progress_bar:
        batch_texts = process_texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move to the device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            # For DDP models, access the underlying module
            if use_ddp and isinstance(model, DDP):
                generated_ids = model.module.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                )
            else:
                generated_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                )

        # Decode
        batch_decrypted = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        decrypted_texts.extend(batch_decrypted)

    # Gather results from all processes when using DDP
    if use_ddp and world_size > 1:
        # Collect every process's results into a list
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, decrypted_texts)

        # Merge all results on rank 0
        if rank == 0:
            all_decrypted = []
            for result in gathered_results:
                all_decrypted.extend(result)
            return all_decrypted
        else:
            return []  # Other processes return an empty list

    return decrypted_texts


def evaluate_results(
    evaluator: LLMEvaluator,
    results: List[Dict],
    use_llm_judge: bool = True,
) -> List[Dict]:
    """
    Score the decoding results.

    Args:
        evaluator: LLMEvaluator instance
        results: list of results to score (each item has original, decrypted, alien)
        use_llm_judge: whether to use the LLM judge

    Returns:
        results with evaluation scores added
    """
    print(f"Evaluating {len(results)} samples...")

    # Run the evaluation
    evaluated_results = evaluator.evaluate_batch(
        results,
        use_llm_judge=use_llm_judge,
        non_parallel=False,  # False because a reference original is available
    )

    return evaluated_results


def save_results(results: List[Dict], output_path: str, metrics: Dict = None):
    """
    Save the evaluation results.

    Args:
        results: list of evaluation results
        output_path: path to save to
        metrics: aggregated metrics (optional)
    """
    print(f"Saving results to {output_path}...")

    # Create the directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save the results
    output_data = {
        "results": results,
    }

    if metrics:
        output_data["metrics"] = metrics

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")


def main_worker(rank: int, world_size: int, args):
    """
    Main function executed in each process.

    Args:
        rank: rank of the current process (read from environment variables under torchrun)
        world_size: total number of processes (read from environment variables under torchrun)
        args: command-line arguments
    """
    # Read rank and world_size from the environment variables set by torchrun
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = rank

    # Initialize DDP
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)

    try:
        # Print GPU info (rank 0 only)
        if rank == 0:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
                print(f"Available GPUs: {num_gpus}")
                print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
                if use_ddp:
                    print(f"DDP enabled with {world_size} processes")
                else:
                    print("Using single GPU")

        # 1. Load the model
        if rank == 0:
            print("=" * 80)
            print("Step 1: Loading trained model")
            print("=" * 80)
        model, tokenizer, device = load_model_and_tokenizer(
            args.model_path,
            rank=rank,
            use_ddp=use_ddp
        )

        # 2. Load the evaluation data (rank 0 only)
        if rank == 0:
            print("\n" + "=" * 80)
            print("Step 2: Loading evaluation data")
            print("=" * 80)
            data_path = Path(__file__).parent / args.data_path
            eval_data = load_eval_data(str(data_path), max_samples=args.max_samples)
        else:
            eval_data = None

        # Broadcast the data to all processes when using DDP
        if use_ddp:
            eval_data_list = [eval_data] if rank == 0 else [None]
            dist.broadcast_object_list(eval_data_list, src=0)
            eval_data = eval_data_list[0]

        # 3. Run decoding
        if rank == 0:
            print("\n" + "=" * 80)
            print("Step 3: Decrypting encrypted texts")
            print("=" * 80)
        encrypted_texts = [item["alien"] for item in eval_data]
        decrypted_texts = decrypt_texts(
            model,
            tokenizer,
            encrypted_texts,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            use_ddp=use_ddp,
        )

        # Result preparation and scoring only run on rank 0
        if rank == 0:
            # Add decrypted_text to the source data
            for item, decrypted in zip(eval_data, decrypted_texts):
                item["decrypted_text"] = decrypted

            # Build the results to score
            results = []
            for item in eval_data:
                results.append({
                    "original": item.get("original", ""),
                    "alien": item.get("alien", ""),
                    "decrypted": item.get("decrypted_text", ""),
                })

            # 4. Run the evaluation
            print("\n" + "=" * 80)
            print("Step 4: Evaluating results")
            print("=" * 80)

            # Evaluation config
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OpenAI API key not found. LLM judge evaluation may fail.")
                print("Please set --api_key or OPENAI_API_KEY environment variable.")

            eval_config = EvaluationConfig(
                api_key=api_key,
                judge_model=args.judge_model,
                max_concurrent_requests=args.max_concurrent_requests,
                use_bleu=True,
                use_rouge=True,
            )

            # Create the evaluator
            evaluator = LLMEvaluator(eval_config)

            # Run the evaluation
            evaluated_results = evaluate_results(
                evaluator,
                results,
                use_llm_judge=args.use_llm_judge,
            )

            # Aggregate the metrics
            metrics = evaluator.aggregate_results(evaluated_results)

            # Print the summary
            evaluator.print_summary(metrics)

            # 5. Save the results
            print("\n" + "=" * 80)
            print("Step 5: Saving results")
            print("=" * 80)
            output_path = Path(__file__).parent / args.output_path

            # Save the evaluation results
            save_results(evaluated_results, str(output_path), metrics=metrics)

            # Save the source data + decrypted_text as a separate file
            output_dir = Path(__file__).parent / Path(args.output_path).parent
            decrypted_data_path = output_dir / "decrypted_data.jsonl"
            print(f"\nSaving decrypted data to {decrypted_data_path}...")
            os.makedirs(output_dir, exist_ok=True)
            with open(decrypted_data_path, 'w', encoding='utf-8') as f:
                for item in eval_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Decrypted data saved to {decrypted_data_path}")

            print("\n" + "=" * 80)
            print("Evaluation completed!")
            print("=" * 80)
    finally:
        # Tear down DDP
        if use_ddp:
            cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained translator model")

    # Model path
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained translator model",
    )

    # Data path
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/test_data.jsonl",
        help="Path to evaluation data (JSONL format)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )

    # Output path
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results",
    )

    # Evaluation settings
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        default=True,
        help="Use LLM-as-a-judge for evaluation",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not set, will use OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-5.1",
        help="LLM judge model name",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=10,
        help="Max concurrent requests for LLM judge",
    )

    # Generation settings
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for decryption",
    )

    # DDP settings
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by torchrun)",
    )

    args = parser.parse_args()

    # Check DDP environment variables (set automatically when launched with torchrun)
    # torchrun sets RANK, WORLD_SIZE, and LOCAL_RANK automatically
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DDP mode (launched with torchrun)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        main_worker(rank, world_size, args)
    else:
        # Single-GPU mode
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
