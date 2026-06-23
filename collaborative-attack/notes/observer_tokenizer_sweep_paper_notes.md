# Observer Tokenizer Sweep: Paper Notes

## Setup

- Victim: Llama AlienLM.
- Fixed leaked corpus: Llama plaintext transformed with the Llama AlienLM tokenizer checkpoint.
- Attack budget: `train=10k`, `k=10k`, `test=1k`, `reference=tulu3 10k`, `n=3`, `top_k=1000`.
- Variable: the observer tokenizer used to tokenize both plaintext and alien-text pairs.

## Main Result

- Best observer tokenizer in this sweep: `llama3` with token accuracy 1.19%.
- Worst observer tokenizer in this sweep: `qwen25` with token accuracy 0.05%.
- If performance drops substantially for non-victim tokenizers, the attack is sensitive to tokenizer assumptions and becomes less reliable when the observer does not know the victim tokenization scheme.

## Suggested Rebuttal Wording

We additionally tested whether the collaborative reconstruction attack depends on knowing the victim tokenizer. Keeping the same Llama AlienLM plaintext/alien-text pairs fixed, we re-ran the attack while tokenizing both sides with five common LLM tokenizers (Llama, Qwen, Gemma, Mistral, and Phi-3). The resulting reconstruction rates varied noticeably across observer tokenizers, indicating that the attack is not tokenizer-invariant and becomes less stable once the adversary does not know the victim tokenization scheme exactly.
