"""
N-gram Attack Evaluation Module

Core functionality:
- Dataset loading (Magpie, Tulu3, AceReason, SlimOrca)
- N-gram based bijection recovery attack
- Evaluation and result saving
"""

from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, Tuple, Union, List
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import os


# ============================================================================
# Dataset loaders
# ============================================================================

def translate_text(text: str, org_tok, alien_tok, direction: str) -> str:
    """Convert text between the original and alien tokenizer."""
    if direction == 'org2alien':
        ids = org_tok.encode(text, add_special_tokens=False)
        return alien_tok.decode(ids)
    elif direction == 'alien2org':
        ids = alien_tok.encode(text, add_special_tokens=False)
        return org_tok.decode(ids)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def load_magpie_data(
    org_tok,
    alien_tok,
    cache_dir: str = None,
    batch_size: int = 1024,
    num_proc: int = 8
) -> Tuple[List[str], List[str]]:
    """
    Load and preprocess the Magpie datasets (reasoning + pro).

    Returns:
        alien_corpus: list of alien texts
        original_corpus: list of original texts
    """
    print("Loading Magpie datasets...")

    pro = load_dataset('Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered', cache_dir=cache_dir)
    reasoning = load_dataset('Magpie-Align/Magpie-Reasoning-V1-150K', cache_dir=cache_dir)

    def batch_translate(batch):
        org_texts, alien_texts = [], []
        for inst, resp in zip(batch["instruction"], batch["response"]):
            merged = inst + resp
            org_texts.append(merged)
            alien_texts.append(translate_text(merged, org_tok, alien_tok, "org2alien"))
        return {"org_seq": org_texts, "alien_seq": alien_texts}

    pro = pro.map(batch_translate, batched=True, batch_size=batch_size, num_proc=num_proc)
    reasoning = reasoning.map(batch_translate, batched=True, batch_size=batch_size, num_proc=num_proc)

    alien_corpus = reasoning['train']['alien_seq'] + pro['train']['alien_seq']
    original_corpus = reasoning['train']['org_seq'] + pro['train']['org_seq']

    print(f"Loaded {len(alien_corpus)} samples from Magpie datasets")
    return alien_corpus, original_corpus


def load_reference_corpus(
    dataset_name: str,
    cache_dir: str = None,
    num_proc: int = 8,
    max_size: int = None
) -> List[str]:
    """
    Load and preprocess a reference corpus dataset.

    Args:
        dataset_name: "tulu3", "acereason", "slimorca"
        cache_dir: cache directory
        num_proc: number of parallel processes
        max_size: maximum number of texts (None for all)

    Returns:
        list of texts
    """
    print(f"Loading reference corpus: {dataset_name}...")

    if dataset_name.lower() == "tulu3":
        ds = load_dataset('allenai/tulu-3-sft-olmo-2-mixture', cache_dir=cache_dir)
        def extract_messages(example):
            contents = [m["content"] for m in example.get("messages", [])]
            return {"contents": contents}
        processed = ds.map(extract_messages, batched=False, num_proc=num_proc)
        # Iterate only as far as needed for efficiency
        corpus = []
        for item in processed['train']:
            corpus.extend(item['contents'])
            if max_size is not None and len(corpus) >= max_size:
                break
        if max_size is not None:
            corpus = corpus[:max_size]

    elif dataset_name.lower() == "acereason":
        file_list = [f'sft_data.parquet/data-00{str(i).zfill(3)}-of-00187.arrow' for i in range(1, 25)]
        ds = load_dataset('nvidia/AceReason-1.1-SFT', data_files=file_list)
        def extract_texts(example):
            texts = []
            if example.get("input"):
                texts.append(example["input"])
            if example.get("output"):
                texts.append(example["output"])
            return {"text": texts}
        processed = ds.map(extract_texts, batched=False, num_proc=num_proc)
        # Flatten the list of lists into a single list
        corpus = []
        for item in processed['train']:
            corpus.extend(item['text'])

    elif dataset_name.lower() == "slimorca":
        ds = load_dataset('Open-Orca/SlimOrca', cache_dir=cache_dir)
        def extract_conversations(example):
            texts = []
            for msg in example.get("conversations", []):
                if msg.get("value"):
                    texts.append(msg["value"])
            return {"text": texts}
        processed = ds.map(extract_conversations, batched=False, num_proc=num_proc)
        # Flatten the list of lists into a single list
        corpus = []
        for item in processed['train']:
            corpus.extend(item['text'])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: tulu3, acereason, slimorca")

    # Apply size limit
    if max_size is not None and len(corpus) > max_size:
        print(f"Limiting corpus size from {len(corpus)} to {max_size}")
        corpus = corpus[:max_size]

    print(f"Loaded {len(corpus)} texts from {dataset_name}")
    return corpus


# ============================================================================
# Bijection extraction
# ============================================================================

class BijectionExtractor:
    """Extract the true bijection from the original and alien tokenizers."""

    def __init__(self, original_tokenizer, alien_tokenizer):
        self.orig_tokenizer = original_tokenizer
        self.alien_tokenizer = alien_tokenizer

    def extract_bijection(self, exclude_special_tokens: bool = True) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Extract the bijection by matching the IDs of identical token strings across both tokenizers."""
        forward_bijection = {}  # {original_id: alien_id}
        inverse_bijection = {}  # {alien_id: original_id}

        orig_vocab = self.orig_tokenizer.get_vocab()
        alien_vocab = self.alien_tokenizer.get_vocab()

        special_tokens = set(self.orig_tokenizer.all_special_tokens) if exclude_special_tokens else set()

        print(f"Original vocabulary size: {len(orig_vocab)}")
        print(f"Alien vocabulary size: {len(alien_vocab)}")

        for token_string, orig_id in orig_vocab.items():
            if token_string in special_tokens:
                continue
            if token_string in alien_vocab:
                alien_id = alien_vocab[token_string]
                forward_bijection[orig_id] = alien_id
                inverse_bijection[alien_id] = orig_id

        print(f"Extracted bijection size: {len(forward_bijection)}")
        print(f"Coverage: {len(forward_bijection)/len(orig_vocab)*100:.2f}%")

        return forward_bijection, inverse_bijection


# ============================================================================
# Known Plaintext Attack - Token Mapping Extraction
# ============================================================================

class KnownPlaintextExtractor:
    """Extract token mappings from ciphertext-plaintext pairs."""

    def __init__(self, original_tokenizer, alien_tokenizer):
        self.orig_tokenizer = original_tokenizer
        self.alien_tokenizer = alien_tokenizer

    def extract_mappings_from_pairs(self,
                                    ciphertext_texts: List[str],
                                    plaintext_texts: List[str],
                                    min_confidence: float = 0.5) -> Dict[int, int]:
        """
        Extract token mappings from ciphertext-plaintext pairs.

        Procedure:
        1. Encode the ciphertext with the original tokenizer to get input ids
        2. Encode the plaintext with the original tokenizer to get input ids
        3. Treat tokens at the same position as a true bijection

        Args:
            ciphertext_texts: list of ciphertext texts
            plaintext_texts: list of plaintext texts (in the same order as ciphertext_texts)
            min_confidence: minimum confidence (fraction of pairs in which the same mapping appears)

        Returns:
            {alien_token_id: original_token_id} mapping dictionary
        """
        print(f"Extracting token mappings from {len(ciphertext_texts)} ciphertext-plaintext pairs...")

        # Store the mappings extracted from each pair
        all_mappings = []  # List of Dict[int, int]

        for i, (ciphertext, plaintext) in enumerate(zip(ciphertext_texts, plaintext_texts)):
            if i % 100 == 0:
                print(f"  Processing pair {i+1}/{len(ciphertext_texts)}...")

            # 1. Encode the ciphertext with the original tokenizer
            cipher_ids_orig = self.orig_tokenizer.encode(ciphertext, add_special_tokens=False)
            # 2. Encode the plaintext with the original tokenizer
            plain_ids_orig = self.orig_tokenizer.encode(plaintext, add_special_tokens=False)

            # 3. Treat tokens at the same position as a true bijection.
            # Also encode the ciphertext with the alien tokenizer to obtain alien token IDs.
            cipher_ids_alien = self.alien_tokenizer.encode(ciphertext, add_special_tokens=False)

            # Only extract a mapping when the lengths match
            if len(cipher_ids_orig) == len(plain_ids_orig) and len(cipher_ids_alien) == len(cipher_ids_orig):
                pair_mapping = {}
                for alien_id, orig_id_cipher, orig_id_plain in zip(cipher_ids_alien, cipher_ids_orig, plain_ids_orig):
                    # Treat tokens at the same position as a true bijection
                    pair_mapping[alien_id] = orig_id_plain
                if pair_mapping:
                    all_mappings.append(pair_mapping)

        # Aggregate the mappings extracted across all pairs
        final_mapping = self._aggregate_mappings(all_mappings, min_confidence)

        print(f"Extracted {len(final_mapping)} token mappings from known plaintext pairs")

        return final_mapping

    def _align_sequences(self,
                        cipher_ids: List[int],
                        plain_ids: List[int],
                        ciphertext: str,
                        plaintext: str) -> Dict[int, int]:
        """
        Align two token sequences and extract a mapping.

        Strategy:
        1. If lengths match, use position-based mapping
        2. If lengths differ, attempt character-level alignment
        3. String matching after decoding tokens
        """
        mapping = {}

        # Strategy 1: lengths match and token IDs can be matched directly
        if len(cipher_ids) == len(plain_ids):
            for c_id, p_id in zip(cipher_ids, plain_ids):
                # Map tokens at the same position (low confidence, but collect for now)
                mapping[c_id] = p_id

        # Strategy 2: decode tokens and match on strings
        try:
            cipher_tokens = self.alien_tokenizer.convert_ids_to_tokens(cipher_ids)
            plain_tokens = self.orig_tokenizer.convert_ids_to_tokens(plain_ids)

            # Cases where token strings match exactly
            cipher_token_set = {self.alien_tokenizer.convert_tokens_to_string([t]) for t in cipher_tokens}
            plain_token_set = {self.orig_tokenizer.convert_tokens_to_string([t]) for t in plain_tokens}

            # Find common token strings
            common_strings = cipher_token_set & plain_token_set

            # For each common string, find a mapping
            for token_str in common_strings:
                # Find the token ID in the ciphertext with this string
                for c_idx, c_tok in enumerate(cipher_tokens):
                    if self.alien_tokenizer.convert_tokens_to_string([c_tok]) == token_str:
                        # Find the token ID in the plaintext with this string
                        for p_idx, p_tok in enumerate(plain_tokens):
                            if self.orig_tokenizer.convert_tokens_to_string([p_tok]) == token_str:
                                mapping[cipher_ids[c_idx]] = plain_ids[p_idx]
                                break
        except Exception as e:
            # Ignore token decoding failures and continue
            pass

        # Strategy 3: character-level alignment (simple heuristic).
        # Compare the character sequences of the ciphertext and plaintext for a rough alignment.
        if not mapping and len(cipher_ids) > 0 and len(plain_ids) > 0:
            # Simple ratio-based mapping
            ratio = len(plain_ids) / len(cipher_ids) if len(cipher_ids) > 0 else 1.0
            for i, c_id in enumerate(cipher_ids):
                p_idx = int(i * ratio)
                if p_idx < len(plain_ids):
                    mapping[c_id] = plain_ids[p_idx]

        return mapping

    def _aggregate_mappings(self,
                           all_mappings: List[Dict[int, int]],
                           min_confidence: float) -> Dict[int, int]:
        """
        Aggregate mappings extracted from multiple pairs into a final mapping.

        Args:
            all_mappings: list of mappings extracted from each pair
            min_confidence: minimum confidence (e.g. 0.5 = must appear in at least 50% of pairs)

        Returns:
            final mapping dictionary
        """
        # Count occurrences of each (alien_id, original_id) pair
        mapping_counts = defaultdict(lambda: defaultdict(int))
        total_pairs = len(all_mappings)

        for pair_mapping in all_mappings:
            for alien_id, orig_id in pair_mapping.items():
                mapping_counts[alien_id][orig_id] += 1

        # Keep only mappings that satisfy the confidence threshold
        final_mapping = {}
        for alien_id, orig_id_counts in mapping_counts.items():
            # Find the most frequent original_id
            if orig_id_counts:
                most_common_orig_id, count = max(orig_id_counts.items(), key=lambda x: x[1])
                confidence = count / total_pairs

                if confidence >= min_confidence:
                    # Conflict check: if already mapped to a different original_id, keep the higher-confidence one
                    if alien_id not in final_mapping:
                        final_mapping[alien_id] = most_common_orig_id
                    else:
                        # If a mapping already exists, compare confidence
                        existing_orig_id = final_mapping[alien_id]
                        existing_count = orig_id_counts.get(existing_orig_id, 0)
                        if count > existing_count:
                            final_mapping[alien_id] = most_common_orig_id

        return final_mapping


# ============================================================================
# N-gram Attack
# ============================================================================

class NGramAttack:
    """Attack that recovers the bijection from n-gram statistics."""

    def __init__(self, n: int = 2, top_k_tokens: int = None, known_mappings: Dict[int, int] = None):
        """
        Args:
            n: n-gram size
            top_k_tokens: use only the top-k tokens by frequency (None for all)
            known_mappings: known token mappings {alien_token_id: original_token_id}
        """
        self.n = n
        self.top_k_tokens = top_k_tokens
        self.known_mappings = known_mappings or {}
        self.token_freq_alien = Counter()
        self.token_freq_ref = Counter()
        self.ngram_freq_alien = defaultdict(Counter)
        self.ngram_freq_ref = defaultdict(Counter)

    def prepare_corpus(self,
                      texts: Union[List[str], Dataset],
                      tokenizer: PreTrainedTokenizer,
                      num_proc: int = 8,
                      batch_size: int = 1000) -> List[List[int]]:
        """Convert texts to token ID sequences (using datasets.map)."""
        if isinstance(texts, Dataset):
            print(f"Preparing corpus: {len(texts)} samples")
            print("using datasets.map for faster processing")
            if 'text' not in texts.column_names:
                if len(texts) == 0:
                    return []
                texts = texts.rename_column(texts.column_names[0], 'text')

            def tokenize_batch(batch):
                results = tokenizer.batch_encode_plus(
                    batch['text'],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                return {"input_ids": results['input_ids']}

            tokenized = texts.map(
                tokenize_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=texts.column_names,
            )
            return tokenized['input_ids']
        else:
            all_input_ids = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                results = tokenizer.batch_encode_plus(
                    batch_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                all_input_ids.extend(results['input_ids'])
            return all_input_ids

    def extract_statistics(self, token_sequences: List[List[int]], is_alien: bool):
        """Extract unigram and n-gram statistics from token ID sequences."""
        freq_dict = self.token_freq_alien if is_alien else self.token_freq_ref
        ngram_dict = self.ngram_freq_alien if is_alien else self.ngram_freq_ref

        for sequence in token_sequences:
            freq_dict.update(sequence)
            for i in range(len(sequence) - self.n + 1):
                ngram = tuple(sequence[i:i + self.n])
                ngram_dict[ngram[0]][ngram] += 1

    def _compute_context_similarity(self, ngrams1: Counter, ngrams2: Counter) -> float:
        """Compute the similarity between two n-gram distributions."""
        if not ngrams1 or not ngrams2:
            return 0.0

        total1, total2 = sum(ngrams1.values()), sum(ngrams2.values())
        if total1 == 0 or total2 == 0:
            return 0.0

        dist1 = {k: v / total1 for k, v in ngrams1.items()}
        dist2 = {k: v / total2 for k, v in ngrams2.items()}
        common_keys = set(dist1.keys()) & set(dist2.keys())

        if not common_keys:
            return 0.0

        return sum(min(dist1[k], dist2[k]) for k in common_keys)

    def compute_similarity_matrix(self, exclude_alien_tokens: set = None, exclude_ref_tokens: set = None) -> Tuple[np.ndarray, List[int], List[int]]:
        """Compute the similarity matrix between alien token IDs and reference token IDs.

        Args:
            exclude_alien_tokens: set of alien token IDs to exclude (tokens covered by known mappings)
            exclude_ref_tokens: set of reference token IDs to exclude (tokens covered by known mappings)
        """
        # Sort tokens by frequency
        alien_tokens_all = sorted(self.token_freq_alien.keys(),
                                 key=lambda x: self.token_freq_alien[x], reverse=True)
        ref_tokens_all = sorted(self.token_freq_ref.keys(),
                               key=lambda x: self.token_freq_ref[x], reverse=True)

        # Keep only the top k
        if self.top_k_tokens is not None:
            alien_tokens_candidate = alien_tokens_all[:self.top_k_tokens]
            ref_tokens_candidate = ref_tokens_all[:self.top_k_tokens]
            print(f"Using top {self.top_k_tokens} tokens (by frequency)")
        else:
            alien_tokens_candidate = alien_tokens_all
            ref_tokens_candidate = ref_tokens_all

        # Exclude tokens covered by known mappings
        exclude_alien_tokens = exclude_alien_tokens or set()
        exclude_ref_tokens = exclude_ref_tokens or set()

        alien_tokens = [tok for tok in alien_tokens_candidate if tok not in exclude_alien_tokens]
        ref_tokens = [tok for tok in ref_tokens_candidate if tok not in exclude_ref_tokens]

        print(f"Alien vocabulary size: {len(alien_tokens)} (excluded {len(exclude_alien_tokens)} known mappings)")
        print(f"Reference vocabulary size: {len(ref_tokens)} (excluded {len(exclude_ref_tokens)} known mappings)")
        if self.known_mappings:
            print(f"Excluding {len(exclude_alien_tokens)} known token mappings from estimation")

        sim_matrix = np.zeros((len(alien_tokens), len(ref_tokens)))

        for i, alien_tok in enumerate(alien_tokens):
            alien_freq = self.token_freq_alien[alien_tok]
            alien_ngrams = self.ngram_freq_alien[alien_tok]

            for j, ref_tok in enumerate(ref_tokens):
                ref_freq = self.token_freq_ref[ref_tok]
                ref_ngrams = self.ngram_freq_ref[ref_tok]

                freq_sim = 1 - abs(alien_freq - ref_freq) / max(alien_freq, ref_freq, 1)
                context_sim = self._compute_context_similarity(alien_ngrams, ref_ngrams)
                base_sim = 0.5 * freq_sim + 0.5 * context_sim

                sim_matrix[i, j] = base_sim

        return sim_matrix, alien_tokens, ref_tokens

    def solve_mapping(self) -> Dict[int, int]:
        """Find the optimal token mapping using the Hungarian algorithm."""
        # Apply known mappings first
        recovered_mapping = {}
        known_count = 0

        # Add known mappings to recovered_mapping
        for alien_tok, orig_tok in self.known_mappings.items():
            recovered_mapping[alien_tok] = orig_tok
            known_count += 1

        if known_count > 0:
            print(f"Using {known_count} known mappings from plaintext pairs")

        # Exclude tokens covered by known mappings before computing the similarity matrix
        exclude_alien_tokens = set(self.known_mappings.keys())
        exclude_ref_tokens = set(self.known_mappings.values())

        sim_matrix, alien_tokens, ref_tokens = self.compute_similarity_matrix(
            exclude_alien_tokens=exclude_alien_tokens,
            exclude_ref_tokens=exclude_ref_tokens
        )

        # Apply the Hungarian algorithm to the remaining tokens only
        if len(alien_tokens) > 0 and len(ref_tokens) > 0:
            cost_matrix = 1 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Add the Hungarian algorithm results to recovered_mapping
            for i, j in zip(row_ind, col_ind):
                alien_tok = alien_tokens[i]
                ref_tok = ref_tokens[j]
                # Add only if not a known mapping (avoid duplicates already added above)
                if alien_tok not in recovered_mapping:
                    recovered_mapping[alien_tok] = ref_tok

        print(f"Recovered {len(recovered_mapping)} token mappings (including {known_count} from known plaintext)")
        return recovered_mapping

    def apply_mapping(self, alien_sequences: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
        """Apply the recovered mapping to the alien token sequences."""
        return [[mapping.get(tok, -1) for tok in seq] for seq in alien_sequences]

    def evaluate(self,
                recovered_sequences: List[List[int]],
                ground_truth_sequences: List[List[int]],
                recovered_mapping: Dict[int, int] = None,
                true_bijection: Dict[int, int] = None) -> Dict:
        """Evaluate the attack success rate."""
        total_tokens = correct_tokens = 0

        for rec_seq, gt_seq in zip(recovered_sequences, ground_truth_sequences):
            min_len = min(len(rec_seq), len(gt_seq))
            for i in range(min_len):
                total_tokens += 1
                if rec_seq[i] == gt_seq[i]:
                    correct_tokens += 1

        metrics = {
            'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0,
            'correct_tokens': correct_tokens,
            'total_tokens': total_tokens
        }

        if recovered_mapping and true_bijection:
            correct_mappings = sum(
                1 for alien_id, pred_id in recovered_mapping.items()
                if alien_id in true_bijection and pred_id == true_bijection[alien_id]
            )
            total_mappings = sum(1 for alien_id in recovered_mapping.keys() if alien_id in true_bijection)

            metrics['bijection_accuracy'] = correct_mappings / total_mappings if total_mappings > 0 else 0
            metrics['correct_mappings'] = correct_mappings
            metrics['total_mappings'] = total_mappings

        return metrics

    def attack(self,
               alien_texts: Union[List[str], Dataset],
               reference_texts: Union[List[str], Dataset],
               test_alien_texts: Union[List[str], Dataset],
               test_ground_truth_texts: Union[List[str], Dataset],
               tokenizer: PreTrainedTokenizer,
               true_bijection: Dict[int, int] = None,
               num_proc: int = 8,
               batch_size: int = 1000,
               top_k_tokens: int = None) -> Tuple[Dict, Dict[int, int]]:
        """
        Run the full N-gram attack pipeline.

        Args:
            top_k_tokens: use only the top-k tokens by frequency (None for all)
        """
        # Update top_k_tokens if passed as an argument
        if top_k_tokens is not None:
            self.top_k_tokens = top_k_tokens

        print("=" * 80)
        print(f"{self.n}-GRAM ATTACK")
        if self.top_k_tokens is not None:
            print(f"Using top {self.top_k_tokens} tokens by frequency")
        print("=" * 80)

        print("\n[Step 1] Converting texts to token IDs...")
        alien_corpus_ids = self.prepare_corpus(alien_texts, tokenizer, num_proc, batch_size)
        ref_corpus_ids = self.prepare_corpus(reference_texts, tokenizer, num_proc, batch_size)
        test_alien_ids = self.prepare_corpus(test_alien_texts, tokenizer, num_proc, batch_size)
        test_gt_ids = self.prepare_corpus(test_ground_truth_texts, tokenizer, num_proc, batch_size)

        print(f"  Alien corpus: {len(alien_corpus_ids)} sequences")
        print(f"  Reference corpus: {len(ref_corpus_ids)} sequences")
        print(f"  Test data: {len(test_alien_ids)} sequences")

        print("\n[Step 2] Extracting n-gram statistics...")
        self.extract_statistics(alien_corpus_ids, is_alien=True)
        self.extract_statistics(ref_corpus_ids, is_alien=False)
        print(f"  Alien vocabulary: {len(self.token_freq_alien)} tokens")
        print(f"  Reference vocabulary: {len(self.token_freq_ref)} tokens")

        print("\n[Step 3] Solving optimal token mapping...")
        recovered_mapping = self.solve_mapping()

        print("\n[Step 4] Applying mapping to test data...")
        recovered_sequences = self.apply_mapping(test_alien_ids, recovered_mapping)

        print("\n[Step 5] Evaluating attack results...")
        metrics = self.evaluate(recovered_sequences, test_gt_ids, recovered_mapping, true_bijection)

        print("\n" + "=" * 80)
        print("ATTACK RESULTS")
        print("=" * 80)
        print(f"Token-level Accuracy:    {metrics['token_accuracy']*100:.2f}%")
        print(f"  Correct tokens:        {metrics['correct_tokens']}/{metrics['total_tokens']}")
        if 'bijection_accuracy' in metrics:
            print(f"Bijection Recovery Rate: {metrics['bijection_accuracy']*100:.2f}%")
            print(f"  Correct mappings:      {metrics['correct_mappings']}/{metrics['total_mappings']}")
        print("=" * 80)

        return metrics, recovered_mapping


# ============================================================================
# Integrated evaluation class
# ============================================================================

class NGramAttackWithEvaluation:
    """Complete N-gram attack evaluation that uses the true bijection."""

    def __init__(self, n: int = 2, top_k_tokens: int = None, known_mappings: Dict[int, int] = None):
        """
        Args:
            n: n-gram size
            top_k_tokens: use only the top-k tokens by frequency (None for all)
            known_mappings: known token mappings {alien_token_id: original_token_id}
        """
        self.n = n
        self.top_k_tokens = top_k_tokens
        self.known_mappings = known_mappings or {}
        self.attack = NGramAttack(n=n, top_k_tokens=top_k_tokens, known_mappings=self.known_mappings)

    def run_full_experiment(self,
                           alien_texts: Union[List[str], Dataset],
                           reference_texts: Union[List[str], Dataset],
                           test_alien_texts: Union[List[str], Dataset],
                           test_ground_truth_texts: Union[List[str], Dataset],
                           original_tokenizer: PreTrainedTokenizer,
                           alien_tokenizer: PreTrainedTokenizer,
                           num_proc: int = 8,
                           batch_size: int = 1000,
                           top_k_tokens: int = None,
                           known_mappings: Dict[int, int] = None):
        """
        Complete N-gram attack experiment including the true bijection.

        Args:
            top_k_tokens: use only the top-k tokens by frequency (None for all)
            known_mappings: known token mappings {alien_token_id: original_token_id} (updated if passed)
        """
        # Update top_k_tokens if passed as an argument
        if top_k_tokens is not None:
            self.top_k_tokens = top_k_tokens
            self.attack.top_k_tokens = top_k_tokens

        # Update known_mappings if passed as an argument
        if known_mappings is not None:
            self.known_mappings = known_mappings
            self.attack.known_mappings = known_mappings

        print("=" * 80)
        print("N-GRAM ATTACK EXPERIMENT WITH TRUE BIJECTION")
        if self.top_k_tokens is not None:
            print(f"Using top {self.top_k_tokens} tokens by frequency")
        if self.known_mappings:
            print(f"Using {len(self.known_mappings)} known token mappings from plaintext pairs")
        print("=" * 80)

        print("\n[Step 1] Extracting true bijection...")
        extractor = BijectionExtractor(original_tokenizer, alien_tokenizer)
        true_forward, true_inverse = extractor.extract_bijection()

        print("\n[Step 2] Running N-gram attack...")
        metrics, recovered_mapping = self.attack.attack(
            alien_texts=alien_texts,
            reference_texts=reference_texts,
            test_alien_texts=test_alien_texts,
            test_ground_truth_texts=test_ground_truth_texts,
            tokenizer=original_tokenizer,
            true_bijection=true_inverse,
            num_proc=num_proc,
            batch_size=batch_size,
            top_k_tokens=self.top_k_tokens
        )

        final_results = {
            'token_accuracy': metrics['token_accuracy'],
            'correct_tokens': metrics['correct_tokens'],
            'total_tokens': metrics['total_tokens'],
            'recovered_mapping_size': len(recovered_mapping),
            'true_bijection_size': len(true_inverse),
            'top_k_tokens': self.top_k_tokens,
            'known_mappings_size': len(self.known_mappings) if self.known_mappings else 0
        }

        if 'bijection_accuracy' in metrics:
            final_results['bijection_recovery_rate'] = metrics['bijection_accuracy']
            final_results['correct_mappings'] = metrics['correct_mappings']
            final_results['total_mappings'] = metrics['total_mappings']

        self.print_final_results(final_results)
        return final_results, recovered_mapping, true_inverse

    def print_final_results(self, results: Dict):
        """Print the final results."""
        print("\n" + "=" * 80)
        print("FINAL EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Token-level Accuracy:      {results['token_accuracy']*100:.2f}%")
        print(f"  Correct/Total:           {results['correct_tokens']}/{results['total_tokens']}")
        if 'bijection_recovery_rate' in results:
            print(f"Bijection Recovery Rate:   {results['bijection_recovery_rate']*100:.2f}%")
            print(f"  Correct/Total:           {results['correct_mappings']}/{results['total_mappings']}")
        if results.get('known_mappings_size', 0) > 0:
            print(f"Known Mappings Size:        {results['known_mappings_size']}")
        print(f"Recovered Mapping Size:    {results['recovered_mapping_size']}")
        print(f"True Bijection Size:       {results['true_bijection_size']}")
        print("=" * 80)

    def save_results(self, results: Dict, recovered_mapping: Dict[int, int], output_dir: str = "./attack_results"):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        with open(f"{output_dir}/recovered_mapping.json", 'w') as f:
            json.dump({str(k): v for k, v in recovered_mapping.items()}, f, indent=2)
        print(f"\nResults saved to {output_dir}/")


# ============================================================================
# Convenience function
# ============================================================================

def run_ngram_attack_experiment(
    alien_corpus: Union[List[str], Dataset],
    reference_corpus: Union[List[str], Dataset],
    test_alien: Union[List[str], Dataset],
    test_ground_truth: Union[List[str], Dataset],
    original_tokenizer: PreTrainedTokenizer,
    alien_tokenizer: PreTrainedTokenizer,
    n: int = 2,
    output_dir: str = "./attack_results",
    num_proc: int = 8,
    batch_size: int = 1000,
    top_k_tokens: int = None,
    known_mappings: Dict[int, int] = None
) -> Dict:
    """
    Run the full N-gram attack experiment (convenience function).

    Args:
        alien_corpus: observed alien texts
        reference_corpus: public English data
        test_alien: alien texts for testing
        test_ground_truth: ground-truth texts for testing
        original_tokenizer: original tokenizer
        alien_tokenizer: alien tokenizer
        n: n-gram size
        output_dir: directory to save results
        num_proc: number of parallel processes
        batch_size: batch size
        top_k_tokens: use only the top-k tokens by frequency (None for all)
        known_mappings: known token mappings {alien_token_id: original_token_id}

    Returns:
        experiment result dict
    """
    experiment = NGramAttackWithEvaluation(n=n, top_k_tokens=top_k_tokens, known_mappings=known_mappings)
    results, recovered_mapping, true_bijection = experiment.run_full_experiment(
        alien_texts=alien_corpus,
        reference_texts=reference_corpus,
        test_alien_texts=test_alien,
        test_ground_truth_texts=test_ground_truth,
        original_tokenizer=original_tokenizer,
        alien_tokenizer=alien_tokenizer,
        num_proc=num_proc,
        batch_size=batch_size,
        top_k_tokens=top_k_tokens,
        known_mappings=known_mappings
    )
    experiment.save_results(results, recovered_mapping, output_dir)
    return results
