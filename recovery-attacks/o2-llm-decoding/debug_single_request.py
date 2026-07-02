"""
Debugging script for a single-sample LLM attack
"""
import json
import time
import os
import sys
from openai import OpenAI
from config import AttackConfig, PromptConfig
from utils import load_test_data


def debug_single_request(data_file: str, sample_idx: int = 0, api_key: str = None):
    """
    Send a request for a single sample and inspect the response

    Args:
        data_file: Path to the JSONL data file
        sample_idx: Index of the sample to test (default: 0)
        api_key: OpenAI API key (read from the environment variable if None)
    """
    print("=" * 80)
    print("DEBUGGING SINGLE REQUEST")
    print("=" * 80)

    # 1. Load configuration
    # Get the API key from the environment variable or set it directly
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found!")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        print("Or pass it with: --api_key 'your-key'")
        sys.exit(1)

    config = AttackConfig(api_key=api_key)
    prompt_config = PromptConfig()
    client = OpenAI(api_key=config.api_key)

    print(f"\nModel: {config.model}")
    print(f"Max completion tokens: {config.max_completion_tokens}")
    print(f"API Key: {config.api_key[:20]}..." if config.api_key else "Not set")

    # 2. Load data
    print(f"\nLoading data from: {data_file}")
    all_data = load_test_data(data_file)

    if sample_idx >= len(all_data):
        print(f"Error: Sample index {sample_idx} is out of range (total: {len(all_data)})")
        return

    sample = all_data[sample_idx]
    original_text = sample["original"]
    alien_text = sample["alien"]

    print(f"\nSample {sample_idx}:")
    print(f"  Original length: {len(original_text)} chars")
    print(f"  Alien length: {len(alien_text)} chars")
    print(f"\n  Original (first 200 chars):")
    print(f"    {original_text[:200]}...")
    print(f"\n  Alien (first 200 chars):")
    print(f"    {alien_text[:200]}...")

    # 3. Build the prompt
    prompt = prompt_config.zero_shot_template.format(alien_text=alien_text)

    print(f"\n{'='*80}")
    print("PROMPT:")
    print(f"{'='*80}")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print(f"{'='*80}")
    print(f"Prompt length: {len(prompt)} chars")

    # 4. Send the API request
    print(f"\n{'='*80}")
    print("SENDING REQUEST...")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # For GPT-5 models, use the responses API
        if config.model.startswith("gpt-5"):
            response = client.responses.create(
                model=config.model,
                input=prompt,
                reasoning={"effort": "low"},
                text={"verbosity": "low"}
            )
            content = response.output_text if hasattr(response, 'output_text') else ""
        else:
            # Standard models use the chat.completions API
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=config.max_completion_tokens
            )
            content = response.choices[0].message.content if response.choices[0].message.content else ""

        elapsed_time = time.time() - start_time

        print(f"\nRequest completed in {elapsed_time:.2f} seconds")
        print(f"Response status: Success")

        # 5. Inspect the response
        if content is None:
            print("Warning: Response content is None!")
            return

        if len(content) == 0:
            print("Warning: Response content is empty!")
            print(f"\nFull response object: {response}")
            return

        print(f"\n{'='*80}")
        print("RESPONSE:")
        print(f"{'='*80}")
        print(content[:1000] + "..." if len(content) > 1000 else content)
        print(f"{'='*80}")
        print(f"Response length: {len(content)} chars")

        # 6. Response details
        print(f"\n{'='*80}")
        print("RESPONSE DETAILS:")
        print(f"{'='*80}")

        if config.model.startswith("gpt-5"):
            # GPT-5 responses API
            print(f"Model used: {config.model}")
            print(f"Response type: responses API")
            if hasattr(response, 'model'):
                print(f"Response model: {response.model}")
        else:
            # Standard chat.completions API
            print(f"Model used: {response.model}")
            if hasattr(response, 'choices') and len(response.choices) > 0:
                print(f"Finish reason: {response.choices[0].finish_reason}")
            if hasattr(response, 'usage'):
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Completion tokens: {response.usage.completion_tokens}")
                print(f"Total tokens: {response.usage.total_tokens}")

        # 7. Comparison
        print(f"\n{'='*80}")
        print("COMPARISON:")
        print(f"{'='*80}")
        print(f"Original: {original_text[:200]}...")
        print(f"Decrypted: {content[:200]}...")

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nRequest failed after {elapsed_time:.2f} seconds")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug single LLM attack request")
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/test_data.jsonl",
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to test (default: 0)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not set, will use OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    debug_single_request(args.data_file, args.sample_idx, api_key=args.api_key)
