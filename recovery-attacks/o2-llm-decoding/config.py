"""
LLM Attack experiment configuration
"""
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AttackConfig:
    """Attack configuration"""
    # OpenAI API settings
    api_key: Optional[str] = None  # If None, read from environment variable
    model: str = "gpt-5.1"  # GPT-5: "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro" | Standard: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
    max_completion_tokens: int = 2000  # Used only by the chat.completions API (GPT-5 uses the responses API)

    # Experiment settings
    n_test_samples: int = 100  # Number of test samples
    few_shot_sizes: List[int] = None  # [1, 3, 5, 10, 20]

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds

    # Async processing settings
    max_concurrent_requests: int = 32  # Limit on the number of concurrent requests

    def __post_init__(self):
        if self.few_shot_sizes is None:
            self.few_shot_sizes = [0, 1, 3, 5, 10]  # 0 = zero-shot

        # Read the API key from the environment variable if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key not found. Please set OPENAI_API_KEY environment variable "
                    "or pass api_key to AttackConfig."
                )


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # LLM-as-a-judge settings
    api_key: Optional[str] = None  # If None, read from environment variable
    judge_model: str = "gpt-5.1"

    # Automatic evaluation metrics
    use_bleu: bool = True
    use_rouge: bool = True

    # Result saving
    output_dir: str = "./llm_attack_results"
    save_intermediate: bool = True  # Whether to save intermediate results

    # Async processing settings
    max_concurrent_requests: int = 10  # Limit on the number of concurrent requests

    def __post_init__(self):
        # Read the API key from the environment variable if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key not found. Please set OPENAI_API_KEY environment variable "
                    "or pass api_key to EvaluationConfig."
                )


@dataclass
class PromptConfig:
    """Prompt template configuration"""

    # Zero-shot prompt
    zero_shot_template: str = """You are analyzing an encrypted text where each token has been substituted with another token via a bijection (one-to-one mapping).

ENCRYPTED TEXT:
{alien_text}

TASK:
Decrypt this text by inferring the original English content. The encryption method:
- Uses vocabulary-level bijection (token substitution)
- Preserves sentence structure and grammar
- Each encrypted token consistently maps to the same original token

Provide your best decryption attempt. Output ONLY the decrypted text without explanations.

DECRYPTED TEXT:"""

    # Few-shot prompt
    few_shot_template: str = """You are analyzing an encrypted text where each token has been substituted with another token via a bijection (one-to-one mapping).

Here are some examples of encrypted text and their original versions:

{examples}

TASK:
Now decrypt this new text using the patterns you learned. The encryption method:
- Uses vocabulary-level bijection (token substitution)
- Preserves sentence structure and grammar
- Each encrypted token consistently maps to the same original token

ENCRYPTED TEXT:
{alien_text}

Provide your best decryption attempt. Output ONLY the decrypted text without explanations.

DECRYPTED TEXT:"""

    # LLM-as-a-judge prompt
    judge_template: str = """You are evaluating the quality of a decryption attempt. You must respond with ONLY a valid JSON object, nothing else.

ORIGINAL TEXT:
{original_text}
DECRYPTED TEXT:
{decrypted_text}

Criteria for Evaluation:
1. Semantic Similarity (1-3):
   - 1 = Poor (the meaning is largely different)
   - 2 = Unclear (some meaning overlaps, but important parts differ)
   - 3 = Excellent (the meaning is almost fully preserved)

2. Structural Similarity (1-3):
   - 1 = Poor (the structure or phrasing is significantly different)
   - 2 = Unclear (partially similar structure, but noticeable differences)
   - 3 = Excellent (the structure closely matches the original)

3. Overall Quality (1-3):
   - 1 = Poor (low-quality reconstruction)
   - 2 = Unclear (partially correct but inconsistent)
   - 3 = Excellent (high-quality, faithful reconstruction)

Scoring Instructions:
- Scores MUST be integers in the range 1-3 (inclusive). Do not use decimals or fractions.
- If either the original text or the decrypted text is missing, empty, or invalid, set all score fields to null and provide an error message in reasoning.

Output Requirements:
- You MUST respond with ONLY a valid JSON object, no markdown, no code blocks, no explanations before or after.
- Do NOT include ```json or ``` markers.
- Do NOT include any text outside the JSON object.
- The JSON must be valid and parseable.

Required JSON Format:
{{
"semantic_similarity": <integer 1-3, or null if error>,
"structural_similarity": <integer 1-3, or null if error>,
"overall_quality": <integer 1-3, or null if error>,
"reasoning": "<brief explanation or error message, up to 300 characters>"
}}

Now output ONLY the JSON object:"""

    # Few-shot prompt for a non-parallel corpus
    few_shot_template_non_parallel: str = """You are analyzing an encrypted text where each token has been substituted with another token via a bijection (one-to-one mapping).

Below are several encrypted text samples generated by the same encryption scheme:

{examples}

TASK:
Now decrypt this new text using the patterns you learned. The encryption method:
- Uses vocabulary-level bijection (token substitution)
- Preserves sentence structure and grammar
- Each encrypted token consistently maps to the same original token

ENCRYPTED TEXT:
{alien_text}

Provide your best decryption attempt. Output ONLY the decrypted text without explanations.

DECRYPTED TEXT:"""

    def format_few_shot_examples(self, examples: List[tuple]) -> str:
        """Format few-shot examples"""
        formatted = []
        for i, (original, encrypted) in enumerate(examples, 1):
            formatted.append(f"EXAMPLE {i}:")
            formatted.append(f"Encrypted: {encrypted}")
            formatted.append(f"Original: {original}")
            formatted.append("")
        return "\n".join(formatted)
