from transformers import AutoTokenizer
from typing import List, Union


def load_tokenizers(original_tokenizer_path: str, alien_tokenizer_path: str):
    """Load original and alien tokenizers."""
    original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
    alien_tokenizer = AutoTokenizer.from_pretrained(alien_tokenizer_path)
    return original_tokenizer, alien_tokenizer


def build_translator(original_tokenizer, alien_tokenizer):
    """Build a translator that converts original text to alien (encrypted) text."""
    class Translator(object):
        def __init__(self, original_tokenizer, alien_tokenizer):
            self.original_tokenizer = original_tokenizer
            self.alien_tokenizer = alien_tokenizer

        def __call__(self, text: Union[str, List[str]]):
            """Translate original text to alien text.

            Args:
                text: Single string or list of strings

            Returns:
                Translated alien text(s)
            """
            if isinstance(text, str):
                original_token_ids = self.original_tokenizer.encode(text, add_special_tokens=False)
                alien_text = self.alien_tokenizer.decode(original_token_ids, skip_special_tokens=True)
                return alien_text
            else:
                # Batch processing
                alien_texts = []
                for t in text:
                    original_token_ids = self.original_tokenizer.encode(t, add_special_tokens=False)
                    alien_text = self.alien_tokenizer.decode(original_token_ids, skip_special_tokens=True)
                    alien_texts.append(alien_text)
                return alien_texts

    return Translator(original_tokenizer, alien_tokenizer)
