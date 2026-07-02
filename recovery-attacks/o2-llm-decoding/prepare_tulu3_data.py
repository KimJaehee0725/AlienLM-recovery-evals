"""
Data preparation script for the LLM Attack using the Tulu3 dataset

Downloads the allenai/tulu-3-sft-olmo-2-mixture-0225 dataset and generates
encrypted-plaintext pairs.

Note: this script depends on the AlienLM codebase. It imports `translate_text`
from the n-gram module of the upstream AlienLM repository. Set ALIENLM_CODE_ROOT
to the root of that repository so the module can be located.
"""
import argparse
import json
import os
import sys
from transformers import AutoTokenizer
from datasets import load_dataset

# Add the path so we can use the translate_text function from ngram_datasets.
# Resolve the AlienLM code root from the environment (falls back to a sibling n-gram dir).
ALIENLM_CODE_ROOT = os.environ.get("ALIENLM_CODE_ROOT")
if ALIENLM_CODE_ROOT:
    sys.path.append(os.path.join(ALIENLM_CODE_ROOT, "attack_scenario", "n-gram"))
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'n-gram'))
from ngram_datasets import translate_text


def load_tulu3_data(
    org_tok,
    alien_tok,
    cache_dir: str = './cache',
    batch_size: int = 1024,
    num_proc: int = 8,
    n_samples: int = 1000
) -> tuple[list[str], list[str]]:
    """
    Load and preprocess the Tulu3 dataset

    Args:
        org_tok: Original tokenizer
        alien_tok: Alien tokenizer
        cache_dir: Cache directory
        batch_size: Batch size
        num_proc: Number of parallel processes
        n_samples: Number of samples to generate

    Returns:
        alien_corpus: List of Alien texts
        original_corpus: List of Original texts
    """
    print("Loading Tulu3 dataset (allenai/tulu-3-sft-olmo-2-mixture-0225)...")

    # Load the dataset
    ds = load_dataset('allenai/tulu-3-sft-olmo-2-mixture-0225', cache_dir=cache_dir)

    def extract_text_from_messages(example):
        """Extract text from messages"""
        messages = example.get("messages", [])
        # Concatenate the content of all messages into a single text
        contents = []
        for msg in messages:
            if msg.get("content"):
                contents.append(msg["content"])
        return {"text": "\n".join(contents) if contents else ""}

    # Extract texts
    print("Extracting texts from messages...")
    processed = ds.map(extract_text_from_messages, batched=False, num_proc=num_proc)

    # Remove empty texts
    def filter_empty(example):
        return len(example["text"].strip()) > 0

    processed = processed.filter(filter_empty, num_proc=num_proc)

    # Use only n_samples
    if len(processed['train']) < n_samples:
        print(f"Warning: Only {len(processed['train'])} samples available, but {n_samples} requested")
        print(f"Using all available samples: {len(processed['train'])}")
        n_samples = len(processed['train'])

    # Batch translation function
    def batch_translate(batch):
        org_texts, alien_texts = [], []
        for text in batch["text"]:
            if text.strip():  # Only process non-empty texts
                org_texts.append(text)
                alien_texts.append(translate_text(text, org_tok, alien_tok, "org2alien"))
        return {"org_seq": org_texts, "alien_seq": alien_texts}

    # Select only the first n_samples
    subset = processed['train'].select(range(n_samples))

    # Run the encryption
    print("Translating texts to alien format...")
    translated = subset.map(
        batch_translate,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc
    )

    alien_corpus = translated['alien_seq']
    original_corpus = translated['org_seq']

    print(f"Loaded {len(alien_corpus)} samples from Tulu3 dataset")
    return alien_corpus, original_corpus


def prepare_tulu3_attack_data(
    org_tokenizer_path: str,
    alien_tokenizer_path: str,
    output_dir: str = "./data",
    n_samples: int = 1000,
    cache_dir: str = './cache',
    batch_size: int = 1024,
    num_proc: int = 8
):
    """
    Prepare data for the LLM attack using the Tulu3 dataset

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
    print("PREPARING TULU3 LLM ATTACK DATA")
    print("=" * 80)
    print(f"Dataset: allenai/tulu-3-sft-olmo-2-mixture-0225")
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

    # 2. Load and convert the Tulu3 data
    print(f"\n[Step 2] Loading and processing Tulu3 data (first {n_samples} samples)...")
    alien_texts, original_texts = load_tulu3_data(
        org_tok=org_tok,
        alien_tok=alien_tok,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_proc=num_proc,
        n_samples=n_samples
    )

    print(f"Loaded {len(alien_texts)} samples")

    # 3. Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save in JSONL format
    print(f"\n[Step 3] Saving data to JSONL file...")
    jsonl_file = os.path.join(output_dir, "test_data_tulu3.jsonl")

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
        description='Prepare LLM attack data from Tulu3 dataset'
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

    prepare_tulu3_attack_data(
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
