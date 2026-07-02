"""
LLM-based decryption attack
"""
import asyncio
from typing import List, Dict, Tuple, Optional
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm

from config import AttackConfig, PromptConfig


class LLMDeciphermentAttack:
    """Decryption attack using OpenAI models"""

    def __init__(self, config: AttackConfig):
        """
        Args:
            config: Attack configuration
        """
        self.config = config
        self.prompt_config = PromptConfig()
        self.client = AsyncOpenAI(api_key=config.api_key)

        # Result storage
        self.results = []

    async def _call_openai(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        """
        Asynchronous OpenAI API call (with retry logic)

        Args:
            prompt: Input prompt
            semaphore: Semaphore to limit the number of concurrent requests

        Returns:
            Model response
        """
        async with semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    # GPT-5 models: use the responses API
                    if self.config.model.startswith("gpt-5"):
                        response = await self.client.responses.create(
                            model=self.config.model,
                            input=prompt,
                            reasoning={"effort": "low"},
                            text={"verbosity": "low"}
                        )
                        return response.output_text.strip() if response.output_text else ""

                    # Standard models: use the chat.completions API
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=self.config.max_completion_tokens
                    )

                    content = response.choices[0].message.content
                    return content.strip() if content else ""

                except RateLimitError as e:
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit (attempt {attempt+1}/{self.config.max_retries}). Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Rate limit exceeded after {self.config.max_retries} attempts: {e}")
                        raise RuntimeError(f"Rate limit exceeded: {e}")
                except APIError as e:
                    if attempt < self.config.max_retries - 1:
                        print(f"API error (attempt {attempt+1}/{self.config.max_retries}): {e}")
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        print(f"API error after {self.config.max_retries} attempts: {e}")
                        raise RuntimeError(f"API error: {e}")
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        print(f"API call failed (attempt {attempt+1}/{self.config.max_retries}): {e}")
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        print(f"API call failed after {self.config.max_retries} attempts: {e}")
                        raise RuntimeError(f"Failed to get response from OpenAI API after {self.config.max_retries} attempts: {e}")

    async def zero_shot_attack(self, alien_text: str, semaphore: asyncio.Semaphore, non_parallel: bool = False) -> str:
        """
        Zero-shot decryption attempt (asynchronous)

        Args:
            alien_text: Encrypted text
            semaphore: Semaphore to limit the number of concurrent requests
            non_parallel: Whether non-parallel mode is enabled (ignored for zero-shot; behaves the same as parallel)

        Returns:
            Decryption attempt text
        """
        # Zero-shot behaves the same as the parallel setting
        prompt = self.prompt_config.zero_shot_template.format(
            alien_text=alien_text
        )

        decrypted = await self._call_openai(prompt, semaphore)
        return decrypted

    async def few_shot_attack(self,
                             alien_text: str,
                             examples: List[Tuple[str, str]],
                             semaphore: asyncio.Semaphore,
                             non_parallel: bool = False) -> str:
        """
        Few-shot decryption attempt (asynchronous)

        Args:
            alien_text: Encrypted text
            examples: [(original, encrypted), ...] example pairs
            semaphore: Semaphore to limit the number of concurrent requests
            non_parallel: Whether non-parallel mode is enabled

        Returns:
            Decryption attempt text
        """
        examples_str = self.prompt_config.format_few_shot_examples(examples)

        if non_parallel:
            prompt = self.prompt_config.few_shot_template_non_parallel.format(
                examples=examples_str,
                alien_text=alien_text
            )
        else:
            prompt = self.prompt_config.few_shot_template.format(
                examples=examples_str,
                alien_text=alien_text
            )

        decrypted = await self._call_openai(prompt, semaphore)
        return decrypted

    async def _process_single_sample(self,
                                    sample: Dict[str, str],
                                    index: int,
                                    is_few_shot: bool,
                                    known_pairs: Optional[List[Tuple[str, str]]],
                                    attack_type: str,
                                    n_shots: int,
                                    semaphore: asyncio.Semaphore,
                                    non_parallel: bool = False) -> Dict:
        """Process a single sample (asynchronous)"""
        # Even in non-parallel mode, the test data contains the original text (for evaluation)
        original_text = sample.get("original", None)
        alien_text = sample["alien"]

        # Decryption attempt (use the non-parallel template if non-parallel mode is enabled)
        if is_few_shot and known_pairs:
            decrypted = await self.few_shot_attack(alien_text, known_pairs, semaphore, non_parallel)
        else:
            decrypted = await self.zero_shot_attack(alien_text, semaphore, non_parallel)

        result = {
            "index": index,
            "alien": alien_text,
            "decrypted": decrypted,
            "attack_type": attack_type,
            "n_shots": n_shots
        }

        # Always include the original text for evaluation
        if original_text:
            result["original"] = original_text

        return result

    def run_experiment(self,
                      test_data: List[Dict[str, str]],
                      known_pairs: Optional[List[Tuple[str, str]]] = None,
                      non_parallel: bool = False) -> List[Dict]:
        """
        Run the full experiment (asynchronous)

        Args:
            test_data: [{"original": "...", "alien": "..."}, ...] or [{"alien": "..."}, ...]
            known_pairs: Known pairs for few-shot (optional)
            non_parallel: Non-parallel corpus mode (no original text)

        Returns:
            List of experiment results
        """
        return asyncio.run(self._run_experiment_async(test_data, known_pairs, non_parallel))

    async def _run_experiment_async(self,
                                   test_data: List[Dict[str, str]],
                                   known_pairs: Optional[List[Tuple[str, str]]] = None,
                                   non_parallel: bool = False) -> List[Dict]:
        """Run the experiment asynchronously"""
        # Decide between zero-shot and few-shot
        is_few_shot = known_pairs is not None and len(known_pairs) > 0
        n_shots = len(known_pairs) if is_few_shot else 0

        attack_type = f"{n_shots}-shot" if is_few_shot else "zero-shot"
        print(f"\n{'='*80}")
        print(f"Running {attack_type} LLM Attack (Async)")
        if non_parallel and is_few_shot:
            print("Mode: Non-parallel attack (using non-parallel template for few-shot, zero-shot uses parallel template)")
        print(f"Model: {self.config.model}")
        print(f"Test samples: {len(test_data)}")
        print(f"Max concurrent requests: {self.config.max_concurrent_requests}")
        if is_few_shot:
            print(f"Known pairs: {n_shots}")
        print(f"{'='*80}\n")

        # Semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Create a task for every sample
        tasks = [
            self._process_single_sample(
                sample, i, is_few_shot, known_pairs, attack_type, n_shots, semaphore, non_parallel
            )
            for i, sample in enumerate(test_data)
        ]

        # tqdm for progress display
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{attack_type} attack"):
            result = await coro
            results.append(result)

            # Intermediate logging (optional)
            if len(results) % 10 == 0:
                print(f"\nProgress: {len(results)}/{len(test_data)}")
                print(f"Sample decryption:")
                if "original" in result:
                    print(f"  Original: {result['original'][:100]}...")
                print(f"  Decrypted: {result['decrypted'][:100]}...")

        # Sort by index order
        results.sort(key=lambda x: x["index"])

        self.results.extend(results)
        return results

    def run_progressive_attack(self,
                              test_data: List[Dict[str, str]],
                              all_pairs: List[Tuple[str, str]],
                              few_shot_sizes: Optional[List[int]] = None) -> Dict[int, List[Dict]]:
        """
        Progressive attack experiment over various few-shot sizes

        Args:
            test_data: Test data
            all_pairs: All available (original, encrypted) pairs
            few_shot_sizes: List of few-shot sizes to try

        Returns:
            Dict of the form {n_shots: results}
        """
        if few_shot_sizes is None:
            few_shot_sizes = self.config.few_shot_sizes

        all_results = {}

        for n_shots in few_shot_sizes:
            print(f"\n{'#'*80}")
            print(f"# Experiment with {n_shots} shots")
            print(f"{'#'*80}")

            if n_shots == 0:
                # Zero-shot
                known_pairs = None
            else:
                # Few-shot: use the first n_shots pairs
                known_pairs = all_pairs[:n_shots]

            results = self.run_experiment(test_data, known_pairs)
            all_results[n_shots] = results

            # Print quick statistics
            self._print_quick_stats(results)

        return all_results

    def _print_quick_stats(self, results: List[Dict]):
        """Print quick statistics"""
        n_samples = len(results)
        n_empty = sum(1 for r in results if not r["decrypted"])

        print(f"\nQuick Statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Empty responses: {n_empty}")
        print(f"  Success rate: {(n_samples - n_empty) / n_samples * 100:.1f}%")
