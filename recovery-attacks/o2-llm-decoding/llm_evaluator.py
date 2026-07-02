"""
Evaluation of LLM decryption results
"""
import asyncio
import json
from typing import List, Dict, Optional
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

# Automatic metrics
from sacrebleu import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer

from config import EvaluationConfig, PromptConfig

class LLMEvaluator:
    """LLM-based evaluation and automatic evaluation metrics"""

    def __init__(self, config: EvaluationConfig):
        """
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.prompt_config = PromptConfig()
        self.client = AsyncOpenAI(api_key=config.api_key)

        # Initialize the ROUGE scorer
        if config.use_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

    async def evaluate_with_llm_judge_no_reference(self,
                                                  encrypted: str,
                                                  decrypted: str,
                                                  semaphore: asyncio.Semaphore) -> Dict:
        """
        Evaluate with LLM-as-a-judge (without a reference, asynchronous)

        Args:
            encrypted: Encrypted text
            decrypted: Decrypted text
            semaphore: Semaphore to limit the number of concurrent requests

        Returns:
            Evaluation result dict
        """
        prompt = self.prompt_config.judge_template.format(
            encrypted_text=encrypted,
            decrypted_text=decrypted
        )

        response_text = ""
        try:
            async with semaphore:
                # GPT-5 models: use the responses API
                if self.config.judge_model.startswith("gpt-5"):
                    response = await self.client.responses.create(
                        model=self.config.judge_model,
                        input=prompt,
                        reasoning={"effort": "low"},
                        text={"verbosity": "low"}
                    )
                    response_text = response.output_text or ""

                    if not response_text or not response_text.strip():
                        return {
                            "coherence": 0,
                            "linguistic_quality": 0,
                            "overall_quality": 0,
                            "reasoning": "Empty response from judge model (GPT-5.1)"
                        }
                else:
                    # Standard models: use the chat.completions API
                    response = await self.client.chat.completions.create(
                        model=self.config.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=500,
                        response_format={"type": "json_object"}
                    )

                    content = response.choices[0].message.content
                    if not content:
                        return {
                            "coherence": 0,
                            "linguistic_quality": 0,
                            "overall_quality": 0,
                            "reasoning": "Empty response from judge model"
                        }

                    response_text = content.strip()

            # JSON parsing
            response_text = response_text.strip()
            if not response_text:
                return {
                    "coherence": 0,
                    "linguistic_quality": 0,
                    "overall_quality": 0,
                    "reasoning": "Empty response text after stripping"
                }

            # Strip Markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Extract only the JSON object
            response_text = response_text.strip()
            if response_text.startswith("{"):
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]

            evaluation = json.loads(response_text)
            return evaluation

        except json.JSONDecodeError as e:
            print(f"LLM judge JSON parsing failed: {e}")
            if response_text:
                print(f"Response text: {response_text[:200]}...")
            return {
                "coherence": 0,
                "linguistic_quality": 0,
                "overall_quality": 0,
                "reasoning": f"JSON parsing failed: {str(e)}"
            }
        except Exception as e:
            print(f"LLM judge failed: {e}")
            return {
                "coherence": 0,
                "linguistic_quality": 0,
                "overall_quality": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    async def evaluate_with_llm_judge(self,
                                     original: str,
                                     decrypted: str,
                                     semaphore: asyncio.Semaphore) -> Dict:
        """
        Evaluate with LLM-as-a-judge (asynchronous)

        Args:
            original: Original text
            decrypted: Decrypted text
            semaphore: Semaphore to limit the number of concurrent requests

        Returns:
            Evaluation result dict
        """
        prompt = self.prompt_config.judge_template.format(
            original_text=original,
            decrypted_text=decrypted
        )

        response_text = ""
        try:
            async with semaphore:
                # GPT-5 models: use the responses API
                if self.config.judge_model.startswith("gpt-5"):
                    # The responses API does not support the JSON format parameter, so force it via the prompt only
                    response = await self.client.responses.create(
                        model=self.config.judge_model,
                        input=prompt,
                        reasoning={"effort": "low"},
                        text={"verbosity": "low"}
                    )
                    response_text = response.output_text or ""

                    # Check for an empty response
                    if not response_text or not response_text.strip():
                        return {
                            "semantic_similarity": 0,
                            "structural_similarity": 0,
                            "overall_quality": 0,
                            "reasoning": "Empty response from judge model (GPT-5.1)"
                        }
                else:
                    # Standard models: use the chat.completions API
                    response = await self.client.chat.completions.create(
                        model=self.config.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=500,
                        response_format={"type": "json_object"}  # Force JSON format
                    )

                    content = response.choices[0].message.content
                    if not content:
                        return {
                            "semantic_similarity": 0,
                            "structural_similarity": 0,
                            "overall_quality": 0,
                            "reasoning": "Empty response from judge model"
                        }

                    response_text = content.strip()

            # JSON parsing
            # The response may sometimes be returned in ```json ... ``` form
            response_text = response_text.strip()
            if not response_text:
                return {
                    "semantic_similarity": 0,
                    "structural_similarity": 0,
                    "overall_quality": 0,
                    "reasoning": "Empty response text after stripping"
                }

            # Strip Markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Extract only the JSON object (the part that starts and ends with braces)
            response_text = response_text.strip()
            if response_text.startswith("{"):
                # Extract from the first { to the last }
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]

            evaluation = json.loads(response_text)
            return evaluation

        except json.JSONDecodeError as e:
            print(f"LLM judge JSON parsing failed: {e}")
            if response_text:
                print(f"Response text: {response_text[:200]}...")
            return {
                "semantic_similarity": 0,
                "structural_similarity": 0,
                "overall_quality": 0,
                "reasoning": f"JSON parsing failed: {str(e)}"
            }
        except Exception as e:
            print(f"LLM judge failed: {e}")
            return {
                "semantic_similarity": 0,
                "structural_similarity": 0,
                "overall_quality": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    def compute_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute the BLEU score

        Args:
            reference: Reference text (the original)
            hypothesis: Hypothesis text (the decryption result)

        Returns:
            BLEU scores
        """
        if not self.config.use_bleu:
            return {}

        # Sentence-level BLEU
        # sacrebleu's sentence_bleu takes strings and tokenizes them automatically
        try:
            bleu = sentence_bleu(hypothesis, [reference])
            # Convert numpy types to native Python types
            precisions = [float(p) for p in bleu.precisions] if hasattr(bleu, 'precisions') else []
            return {
                "bleu": float(bleu.score),
                "bleu_precisions": precisions  # 1-gram ~ 4-gram
            }
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            return {"bleu": 0.0}

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores

        Args:
            reference: Reference text
            hypothesis: Hypothesis text

        Returns:
            ROUGE scores
        """
        if not self.config.use_rouge:
            return {}

        scores = self.rouge_scorer.score(reference, hypothesis)

        # Convert numpy types to native Python types
        return {
            "rouge1_f": float(scores['rouge1'].fmeasure),
            "rouge1_p": float(scores['rouge1'].precision),
            "rouge1_r": float(scores['rouge1'].recall),
            "rouge2_f": float(scores['rouge2'].fmeasure),
            "rouge2_p": float(scores['rouge2'].precision),
            "rouge2_r": float(scores['rouge2'].recall),
            "rougeL_f": float(scores['rougeL'].fmeasure),
            "rougeL_p": float(scores['rougeL'].precision),
            "rougeL_r": float(scores['rougeL'].recall),
        }

    def compute_exact_match(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Check whether the two strings match exactly"""
        exact_match = 1.0 if reference.strip() == hypothesis.strip() else 0.0
        return {"exact_match": exact_match}

    async def evaluate_single(self,
                              original: Optional[str],
                              decrypted: str,
                              use_llm_judge: bool,
                              semaphore: Optional[asyncio.Semaphore] = None,
                              non_parallel: bool = False,
                              encrypted: Optional[str] = None) -> Dict:
        """
        Evaluate a single sample (all metrics, asynchronous)

        Args:
            original: Original text (if None, evaluate without a reference)
            decrypted: Decrypted text
            use_llm_judge: Whether to use the LLM judge
            semaphore: Semaphore to limit the number of concurrent requests
            non_parallel: Non-parallel corpus mode (the attack does not use the original; evaluation uses the original if available)
            encrypted: Encrypted text

        Returns:
            All evaluation metrics
        """
        evaluation = {}

        # If the original is available, evaluate against it (automatic metrics + LLM judge)
        if original:
            # Automatic metrics can be used
            evaluation.update(self.compute_bleu(original, decrypted))
            evaluation.update(self.compute_rouge(original, decrypted))
            evaluation.update(self.compute_exact_match(original, decrypted))

            # LLM-as-a-judge (compared against the original)
            if use_llm_judge and semaphore:
                llm_eval = await self.evaluate_with_llm_judge(original, decrypted, semaphore)
                evaluation["llm_semantic"] = llm_eval.get("semantic_similarity", 0)
                evaluation["llm_structural"] = llm_eval.get("structural_similarity", 0)
                evaluation["llm_overall"] = llm_eval.get("overall_quality", 0)
                evaluation["llm_reasoning"] = llm_eval.get("reasoning", "")
        else:
            # If there is no original, evaluate quality without a reference (LLM judge only)
            if use_llm_judge and semaphore and encrypted:
                llm_eval = await self.evaluate_with_llm_judge_no_reference(encrypted, decrypted, semaphore)
                evaluation["llm_coherence"] = llm_eval.get("coherence", 0)
                evaluation["llm_linguistic_quality"] = llm_eval.get("linguistic_quality", 0)
                evaluation["llm_overall"] = llm_eval.get("overall_quality", 0)
                evaluation["llm_reasoning"] = llm_eval.get("reasoning", "")

        return evaluation

    def evaluate_batch(self,
                      results: List[Dict],
                      use_llm_judge: bool = True,
                      non_parallel: bool = False) -> List[Dict]:
        """
        Batch evaluation (asynchronous)

        Args:
            results: [{"original": "...", "decrypted": "...", ...}, ...] or [{"alien": "...", "decrypted": "...", ...}, ...]
            use_llm_judge: Whether to use the LLM judge
            non_parallel: Non-parallel corpus mode (no original text)

        Returns:
            results with the evaluation appended
        """
        return asyncio.run(self._evaluate_batch_async(results, use_llm_judge, non_parallel))

    async def _evaluate_batch_async(self,
                                   results: List[Dict],
                                   use_llm_judge: bool,
                                   non_parallel: bool = False) -> List[Dict]:
        """Asynchronous batch evaluation"""
        print(f"\n{'='*80}")
        print(f"Evaluating {len(results)} samples (Async)")
        if non_parallel:
            print("Mode: Non-parallel corpus (attack without reference, evaluation uses reference if available)")
        # Count the number of samples that have a reference text
        n_with_reference = sum(1 for r in results if r.get("original"))
        if n_with_reference > 0:
            print(f"Samples with reference text: {n_with_reference}/{len(results)}")
        if use_llm_judge:
            print(f"LLM Judge: {self.config.judge_model}")
            print(f"Max concurrent requests: {self.config.max_concurrent_requests}")
        print(f"{'='*80}\n")

        # Semaphore to limit the number of concurrent requests (only the LLM judge is asynchronous)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests) if use_llm_judge else None

        # Create an evaluation task for every sample
        async def evaluate_with_index(idx, result):
            # During evaluation, use the original if available (a reference may exist for evaluation even in non_parallel mode)
            original = result.get("original", None)
            encrypted = result.get("alien", None)
            decrypted = result.get("decrypted", "")

            evaluation = await self.evaluate_single(
                original=original,
                decrypted=decrypted,
                use_llm_judge=use_llm_judge,
                semaphore=semaphore,
                non_parallel=non_parallel,
                encrypted=encrypted
            )
            return idx, evaluation

        tasks = [
            evaluate_with_index(i, result)
            for i, result in enumerate(results)
        ]

        # tqdm for progress display
        evaluations = [None] * len(results)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            idx, evaluation = await coro
            evaluations[idx] = evaluation

        # Append the evaluation to the results
        evaluated_results = []
        for result, evaluation in zip(results, evaluations):
            result_with_eval = result.copy()
            result_with_eval["evaluation"] = evaluation
            evaluated_results.append(result_with_eval)

        return evaluated_results

    def aggregate_results(self, evaluated_results: List[Dict]) -> Dict:
        """
        Aggregate all results

        Args:
            evaluated_results: Results that have been evaluated

        Returns:
            Aggregated statistics
        """
        metrics = {}

        # Collect all evaluation metrics
        all_metrics = {}
        for result in evaluated_results:
            eval_dict = result.get("evaluation", {})
            for key, value in eval_dict.items():
                if key == "llm_reasoning":
                    continue  # Skip text field
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Compute mean and standard deviation (convert numpy types to native Python types)
        for key, values in all_metrics.items():
            metrics[f"{key}_mean"] = float(np.mean(values))
            metrics[f"{key}_std"] = float(np.std(values))
            metrics[f"{key}_min"] = float(np.min(values))
            metrics[f"{key}_max"] = float(np.max(values))

        # Store LLM judge scores structured by category
        if "llm_semantic" in all_metrics or "llm_structural" in all_metrics or "llm_overall" in all_metrics:
            metrics["llm_judge_scores"] = {}

            if "llm_semantic" in all_metrics:
                semantic_values = all_metrics["llm_semantic"]
                # Convert numpy types to native Python types
                semantic_values_float = [float(v) for v in semantic_values]
                metrics["llm_judge_scores"]["semantic"] = {
                    "mean": float(np.mean(semantic_values_float)),
                    "std": float(np.std(semantic_values_float)),
                    "min": float(np.min(semantic_values_float)),
                    "max": float(np.max(semantic_values_float)),
                    "values": semantic_values_float  # Per-sample scores
                }

            if "llm_structural" in all_metrics:
                structural_values = all_metrics["llm_structural"]
                # Convert numpy types to native Python types
                structural_values_float = [float(v) for v in structural_values]
                metrics["llm_judge_scores"]["structural"] = {
                    "mean": float(np.mean(structural_values_float)),
                    "std": float(np.std(structural_values_float)),
                    "min": float(np.min(structural_values_float)),
                    "max": float(np.max(structural_values_float)),
                    "values": structural_values_float  # Per-sample scores
                }

            if "llm_overall" in all_metrics:
                overall_values = all_metrics["llm_overall"]
                # Convert numpy types to native Python types
                overall_values_float = [float(v) for v in overall_values]
                metrics["llm_judge_scores"]["overall"] = {
                    "mean": float(np.mean(overall_values_float)),
                    "std": float(np.std(overall_values_float)),
                    "min": float(np.min(overall_values_float)),
                    "max": float(np.max(overall_values_float)),
                    "values": overall_values_float  # Per-sample scores
                }

        # Additional statistics
        metrics["n_samples"] = len(evaluated_results)
        metrics["n_empty"] = sum(1 for r in evaluated_results if not r.get("decrypted", ""))

        return metrics

    def print_summary(self, aggregated_metrics: Dict):
        """Print a summary of the results"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")

        print(f"\nSamples: {aggregated_metrics['n_samples']}")
        print(f"Empty responses: {aggregated_metrics['n_empty']}")

        # Key metrics
        key_metrics = ["bleu", "rouge1_f", "rouge2_f", "rougeL_f"]
        if "llm_overall_mean" in aggregated_metrics:
            key_metrics.append("llm_overall")

        print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)

        for metric in key_metrics:
            mean_key = f"{metric}_mean"
            if mean_key in aggregated_metrics:
                mean_val = aggregated_metrics[mean_key]
                std_val = aggregated_metrics.get(f"{metric}_std", 0)
                min_val = aggregated_metrics.get(f"{metric}_min", 0)
                max_val = aggregated_metrics.get(f"{metric}_max", 0)

                print(f"{metric:<20} {mean_val:<10.2f} {std_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f}")

        # Print LLM judge scores by category
        if "llm_judge_scores" in aggregated_metrics:
            print(f"\n{'='*80}")
            print("LLM JUDGE SCORES BY CATEGORY")
            print(f"{'='*80}")
            print(f"\n{'Category':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 60)

            judge_scores = aggregated_metrics["llm_judge_scores"]
            for category in ["semantic", "structural", "overall"]:
                if category in judge_scores:
                    cat_data = judge_scores[category]
                    print(f"{category:<20} {cat_data['mean']:<10.2f} {cat_data['std']:<10.2f} {cat_data['min']:<10.2f} {cat_data['max']:<10.2f}")

        print("="*80)
