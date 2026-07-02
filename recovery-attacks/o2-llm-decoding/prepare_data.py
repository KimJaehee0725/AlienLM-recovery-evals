"""
Data preparation script for the LLM Attack

Uses the first 1000 samples of the Magpie data to generate encrypted-plaintext pairs.

Note: this script depends on the AlienLM codebase. It imports `load_magpie_data`
and `translate_text` from the n-gram module of the upstream AlienLM repository.
Set ALIENLM_CODE_ROOT to the root of that repository so the module can be located.
"""
import argparse
import json
import os
import sys
from transformers import AutoTokenizer

# Add the path so we can use the translate_text function from ngram_datasets.
# Resolve the AlienLM code root from the environment (falls back to a sibling n-gram dir).
ALIENLM_CODE_ROOT = os.environ.get("ALIENLM_CODE_ROOT")
if ALIENLM_CODE_ROOT:
    sys.path.append(os.path.join(ALIENLM_CODE_ROOT, "attack_scenario", "n-gram"))
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'n-gram'))
from ngram_datasets import load_magpie_data, translate_text


def prepare_llm_attack_data(
    org_tokenizer_path: str,
    alien_tokenizer_path: str,
    output_dir: str = "./data",
    n_samples: int = 1000,
    cache_dir: str = './cache',
    batch_size: int = 1024,
    num_proc: int = 8
):
    """
    Prepare data for the LLM attack

    Args:
        org_tokenizer_path: Path to the original tokenizer
        alien_tokenizer_path: Path to the Alien tokenizer
        output_dir: Output directory
        n_samples: Number of samples to generate (default: 1000)
        cache_dir: Dataset cache directory
        batch_size: Batch size
        num_proc: Number of parallel processes
    """
    print("=" * 80)
    print("PREPARING LLM ATTACK DATA")
    print("=" * 80)
    print(f"Original tokenizer: {org_tokenizer_path}")
    print(f"Alien tokenizer: {alien_tokenizer_path}")
    print(f"Number of samples: {n_samples}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # 1. Load tokenizers
    print("\n[Step 1] Loading tokenizers...")
    org_tok = AutoTokenizer.from_pretrained(org_tokenizer_path)
    alien_tok = AutoTokenizer.from_pretrained(alien_tokenizer_path)
    print("Tokenizers loaded")

    # 2. Load Magpie data
    print(f"\n[Step 2] Loading Magpie data (first {n_samples} samples)...")
    alien_corpus, original_corpus = load_magpie_data(
        org_tok=org_tok,
        alien_tok=alien_tok,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_proc=num_proc
    )

    # Use only the first n_samples
    if len(alien_corpus) < n_samples:
        print(f"Warning: Only {len(alien_corpus)} samples available, but {n_samples} requested")
        print(f"Using all available samples: {len(alien_corpus)}")
        n_samples = len(alien_corpus)

    alien_texts = alien_corpus[:n_samples]
    original_texts = original_corpus[:n_samples]

    print(f"Loaded {len(alien_texts)} samples")

    # 3. Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save in JSONL format
    print(f"\n[Step 3] Saving data to JSONL file...")
    jsonl_file = os.path.join(output_dir, "test_data.jsonl")

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for original, alien in zip(original_texts, alien_texts):
            # Save each sample as a JSON object (one per line)
            data_item = {
                "original": original,
                "alien": alien
            }
            f.write(json.dumps(data_item, ensure_ascii=False) + '\n')

    print(f"Saved {len(alien_texts)} samples")
    print(f"  JSONL file: {jsonl_file}")

    # 5. Print a sample
    print(f"\n[Step 4] Sample data preview:")
    print("-" * 80)
    print(f"Sample 1:")
    print(f"  Original: {original_texts[0][:200]}...")
    print(f"  Alien:    {alien_texts[0][:200]}...")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETED")
    print("=" * 80)
    print(f"File saved to: {jsonl_file}")
    print("=" * 80)

    return jsonl_file


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LLM attack data from Magpie dataset'
    )
    parser.add_argument(
        '--org_tokenizer_path',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='Original tokenizer path'
    )
    parser.add_argument(
        '--alien_tokenizer_path',
        type=str,
        default=os.environ.get("ALIEN_TOKENIZER_PATH", ""),
        help='Alien tokenizer path (or set ALIEN_TOKENIZER_PATH)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for data files'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=os.environ.get("HF_HOME", "./cache"),
        help='Cache directory for datasets'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=8,
        help='Number of processes for parallel processing'
    )

    args = parser.parse_args()

    prepare_llm_attack_data(
        org_tokenizer_path=args.org_tokenizer_path,
        alien_tokenizer_path=args.alien_tokenizer_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )


if __name__ == "__main__":
    main()
