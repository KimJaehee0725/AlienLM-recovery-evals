"""
N-gram Attack Evaluation Script

Settings:
- reference corpus: tulu3, acereason, slimorca
- train dataset size: 50,000
- test dataset size: 50,000
- top_k_tokens: 10,000
- n: 2, 3, 4
"""

import argparse
import os
import sys
import signal
import atexit
from transformers import AutoTokenizer
from ngram_datasets import load_magpie_data, load_reference_corpus, run_ngram_attack_experiment

# Unbuffered output: prefer the `python -u` flag or the PYTHONUNBUFFERED
# environment variable. Here we use flush=True on print calls instead.


# Global: store cleanup functions
cleanup_functions = []


def cleanup_on_exit():
    """Run cleanup functions on program exit."""
    for func in cleanup_functions:
        try:
            func()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)


def signal_handler(signum, frame):
    """Signal handler (Ctrl+C, etc.)."""
    print("\nReceived interrupt signal. Cleaning up...")
    cleanup_on_exit()
    sys.exit(1)


def check_existing_processes():
    """Check for previous processes."""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.split('\n')
        current_pid = os.getpid()
        script_name = os.path.basename(__file__)

        for line in lines:
            if script_name in line and str(current_pid) not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    print(f"Warning: Found another {script_name} process (PID: {pid})")
                    return True
    except Exception as e:
        print(f"Could not check processes: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description='N-gram Attack Evaluation')
    parser.add_argument('--reference_corpus', type=str, required=True,
                        choices=['tulu3', 'acereason', 'slimorca'],
                        help='Reference corpus name')
    parser.add_argument('--n', type=int, required=True,
                        choices=[2, 3, 4],
                        help='N-gram size')
    parser.add_argument('--train_size', type=int, default=50000,
                        help='Training dataset size')
    parser.add_argument('--test_size', type=int, default=50000,
                        help='Test dataset size')
    parser.add_argument('--top_k_tokens', type=int, default=10000,
                        help='Top K tokens to use')
    parser.add_argument('--reference_size', type=int, default=None,
                        help='Maximum size of reference corpus (None for all)')
    parser.add_argument('--output_dir', type=str, default='./attack_results',
                        help='Output directory')
    parser.add_argument('--num_proc', type=int, default=8,
                        help='Number of processes for parallel processing')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--org_tokenizer_path', type=str,
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Original tokenizer path (HF id or local path)')
    parser.add_argument('--alien_tokenizer_path', type=str,
                        default=os.environ.get('ALIEN_TOKENIZER_PATH'),
                        help='Alien tokenizer path (local path to the AlienLM checkpoint; '
                             'defaults to the ALIEN_TOKENIZER_PATH environment variable)')
    parser.add_argument('--cache_dir', type=str,
                        default=os.environ.get('HF_DATASETS_CACHE'),
                        help='Cache directory for datasets (defaults to the '
                             'HF_DATASETS_CACHE environment variable)')


    args = parser.parse_args()

    if not args.alien_tokenizer_path:
        parser.error(
            "An alien tokenizer is required: pass --alien_tokenizer_path or set "
            "the ALIEN_TOKENIZER_PATH environment variable."
        )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)

    # Check for previous processes
    if check_existing_processes():
        print("Warning: Other instances may be running. Continuing anyway...")

    print("=" * 80, flush=True)
    print("N-GRAM ATTACK EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Reference Corpus: {args.reference_corpus}", flush=True)
    print(f"N-gram size: {args.n}", flush=True)
    print(f"Train size: {args.train_size}", flush=True)
    print(f"Test size: {args.test_size}", flush=True)
    print(f"Top K tokens: {args.top_k_tokens}", flush=True)
    print(f"Reference size: {args.reference_size if args.reference_size else 'All'}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load tokenizers
    print("\n[Step 1] Loading tokenizers...", flush=True)
    org_tok = AutoTokenizer.from_pretrained(args.org_tokenizer_path)
    alien_tok = AutoTokenizer.from_pretrained(args.alien_tokenizer_path)
    print("✓ Tokenizers loaded", flush=True)

    # 2. Load Magpie data
    print("\n[Step 2] Loading Magpie data...", flush=True)
    alien_corpus, original_corpus = load_magpie_data(
        org_tok=org_tok,
        alien_tok=alien_tok,
        cache_dir=args.cache_dir,
        batch_size=1024,
        num_proc=args.num_proc
    )
    print(f"✓ Loaded {len(alien_corpus)} samples from Magpie datasets", flush=True)

    # 3. Split train/test
    print("\n[Step 3] Splitting train/test data...", flush=True)
    total_needed = args.train_size + args.test_size
    if len(alien_corpus) < total_needed:
        print(f"Warning: Only {len(alien_corpus)} samples available, but {total_needed} needed")
        print(f"Using all available samples: train={len(alien_corpus) - args.test_size}, test={args.test_size}")
        train_alien = alien_corpus[:-args.test_size]
        train_original = original_corpus[:-args.test_size]
        test_alien = alien_corpus[-args.test_size:]
        test_original = original_corpus[-args.test_size:]
    else:
        train_alien = alien_corpus[:args.train_size]
        train_original = original_corpus[:args.train_size]
        test_alien = alien_corpus[args.train_size:args.train_size + args.test_size]
        test_original = original_corpus[args.train_size:args.train_size + args.test_size]

    print(f"✓ Train: {len(train_alien)} samples", flush=True)
    print(f"✓ Test: {len(test_alien)} samples", flush=True)

    # 4. Load reference corpus
    print(f"\n[Step 4] Loading reference corpus: {args.reference_corpus}...", flush=True)
    reference_corpus = load_reference_corpus(
        dataset_name=args.reference_corpus,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        max_size=args.reference_size
    )
    print(f"✓ Loaded {len(reference_corpus)} texts from {args.reference_corpus}", flush=True)

    # 5. Create output directory
    ref_size_str = f"_ref{args.reference_size}" if args.reference_size else ""
    output_dir = os.path.join(
        args.output_dir,
        f"{args.reference_corpus}_n{args.n}_train{args.train_size}_test{args.test_size}_topk{args.top_k_tokens}{ref_size_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 6. Run experiment
    print("\n[Step 5] Running N-gram attack experiment...", flush=True)
    results = run_ngram_attack_experiment(
        alien_corpus=train_alien,
        reference_corpus=reference_corpus,
        test_alien=test_alien,
        test_ground_truth=test_original,
        original_tokenizer=org_tok,
        alien_tokenizer=alien_tok,
        n=args.n,
        output_dir=output_dir,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        top_k_tokens=args.top_k_tokens
    )

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION COMPLETED", flush=True)
    print("=" * 80, flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    return results


if __name__ == "__main__":
    main()
