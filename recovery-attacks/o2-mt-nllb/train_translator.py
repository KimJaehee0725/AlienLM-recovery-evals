#!/usr/bin/env python3
"""
Train a translation model to decrypt alien text back to original text.
Uses facebook/nllb-200-3.3B model with 4 GPU support via DDP.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset as HFDataset
import logging

from translator import load_tokenizers, build_translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="facebook/nllb-200-3.3B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    original_tokenizer_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Path to original tokenizer"}
    )
    alien_tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to alien tokenizer (required)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset to use"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name of the dataset"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column name containing the text data"}
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length"}
    )
    max_target_length: int = field(
        default=512,
        metadata={"help": "Maximum target sequence length"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets"}
    )


class TranslationDataset(Dataset):
    """Dataset that generates encrypted-original pairs on the fly."""

    def __init__(
        self,
        texts: List[str],
        translator,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
    ):
        self.texts = texts
        self.translator = translator
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        original_text = self.texts[idx]

        # Generate encrypted text using translator
        encrypted_text = self.translator(original_text)

        # Tokenize (return numpy arrays, not tensors - DataCollator will handle tensor conversion)
        model_inputs = self.tokenizer(
            encrypted_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # Return as lists/arrays, not tensors
        )

        labels = self.tokenizer(
            original_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # Return as lists/arrays, not tensors
        )

        # Convert labels to numpy array and replace padding token id's with -100
        labels = np.array(labels["input_ids"], dtype=np.int64)
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


def prepare_dataset(
    dataset_name: Optional[str],
    text_column: str,
    translator,
    tokenizer,
    max_source_length: int = 512,
    max_target_length: int = 512,
    num_samples: Optional[int] = None,
    dataset_config_name: Optional[str] = None,
):
    """Prepare dataset for training."""
    if dataset_name:
        # Support multiple datasets separated by comma (e.g., "magpie,slimorca")
        dataset_names = [name.strip() for name in dataset_name.split(",")]
        texts = []

        for ds_name in dataset_names:
            # Check if it's Magpie dataset (special handling)
            is_magpie = (
                ds_name.lower() == "magpie" or
                "Magpie" in ds_name or
                "magpie" in ds_name.lower()
            )

            # Check if it's SlimOrca dataset (special handling)
            is_slimorca = (
                ds_name.lower() == "slimorca" or
                "slimorca" in ds_name.lower() or
                "slim-orca" in ds_name.lower() or
                "SlimOrca" in ds_name
            )

            if is_magpie:
                # Load Magpie datasets and extract instruction/response separately
                logger.info("Loading Magpie datasets...")
                magpie_texts = []

                # Helper function to extract texts from a single sample (for Magpie-Pro) - optimized with map
                def extract_texts_from_pro_sample(example):
                    """Extract texts from Magpie-Pro sample (conversations format)."""
                    extracted = []
                    if "conversations" in example and isinstance(example["conversations"], list):
                        for conv in example["conversations"]:
                            if isinstance(conv, dict) and "value" in conv:
                                text = conv["value"]
                                if text and isinstance(text, str) and text.strip():
                                    extracted.append(text.strip())
                    # Fallback: check for instruction/response format
                    elif "instruction" in example and example["instruction"] and example["instruction"].strip():
                        extracted.append(example["instruction"].strip())
                    elif "response" in example and example["response"] and example["response"].strip():
                        extracted.append(example["response"].strip())
                    return {"texts": extracted}

                # Helper function to extract texts from a single sample (for Magpie-Reasoning) - optimized with map
                def extract_texts_from_reasoning_sample(example):
                    """Extract texts from Magpie-Reasoning sample (instruction/response format)."""
                    extracted = []
                    # Check for instruction/response format
                    if "instruction" in example and example["instruction"] and example["instruction"].strip():
                        extracted.append(example["instruction"].strip())
                    if "response" in example and example["response"] and example["response"].strip():
                        extracted.append(example["response"].strip())
                    # Also check conversations format if available
                    if "conversations" in example and isinstance(example["conversations"], list):
                        for conv in example["conversations"]:
                            if isinstance(conv, dict) and "value" in conv:
                                text = conv["value"]
                                if text and isinstance(text, str) and text.strip():
                                    extracted.append(text.strip())
                    return {"texts": extracted}

                # Load Magpie-Pro-300K-Filtered
                try:
                    logger.info("Loading Magpie-Pro-300K-Filtered...")
                    pro_dataset = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split="train")

                    # Use map function for efficient parallel processing
                    logger.info("Extracting texts from Magpie-Pro using map function...")
                    pro_dataset_mapped = pro_dataset.map(
                        extract_texts_from_pro_sample,
                        batched=False,
                        num_proc=8,  # Use 8 processes for parallel processing
                        remove_columns=pro_dataset.column_names,
                        desc="Processing Magpie-Pro"
                    )

                    # Extract all texts
                    pro_texts = []
                    for item in pro_dataset_mapped:
                        pro_texts.extend(item["texts"])

                    magpie_texts.extend(pro_texts)
                    logger.info(f"Extracted {len(pro_texts)} texts from Magpie-Pro ({len(pro_dataset)} samples)")
                except Exception as e:
                    logger.warning(f"Failed to load Magpie-Pro-300K-Filtered: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())

                # Load Magpie-Reasoning-V1-150K
                try:
                    logger.info("Loading Magpie-Reasoning-V1-150K...")
                    reasoning_dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V1-150K", split="train")

                    # Use map function for efficient parallel processing
                    logger.info("Extracting texts from Magpie-Reasoning using map function...")
                    reasoning_dataset_mapped = reasoning_dataset.map(
                        extract_texts_from_reasoning_sample,
                        batched=False,
                        num_proc=8,  # Use 8 processes for parallel processing
                        remove_columns=reasoning_dataset.column_names,
                        desc="Processing Magpie-Reasoning"
                    )

                    # Extract all texts
                    reasoning_texts = []
                    for item in reasoning_dataset_mapped:
                        reasoning_texts.extend(item["texts"])

                    magpie_texts.extend(reasoning_texts)
                    logger.info(f"Extracted {len(reasoning_texts)} texts from Magpie-Reasoning ({len(reasoning_dataset)} samples)")
                except Exception as e:
                    logger.warning(f"Failed to load Magpie-Reasoning-V1-150K: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())

                texts.extend(magpie_texts)
                logger.info(f"Total texts from Magpie datasets: {len(magpie_texts)}")

            elif is_slimorca:
                # Load SlimOrca dataset
                logger.info("Loading SlimOrca dataset...")
                try:
                    slimorca_dataset = load_dataset("Open-Orca/SlimOrca", split="train")

                    # Helper function to extract texts from SlimOrca sample
                    def extract_texts_from_slimorca_sample(example):
                        """Extract texts from SlimOrca sample (conversations format)."""
                        extracted = []
                        if "conversations" in example and isinstance(example["conversations"], list):
                            for conv in example["conversations"]:
                                if isinstance(conv, dict) and "value" in conv:
                                    text = conv["value"]
                                    if text and isinstance(text, str) and text.strip():
                                        extracted.append(text.strip())
                        return {"texts": extracted}

                    # Use map function for efficient parallel processing
                    logger.info("Extracting texts from SlimOrca using map function...")
                    slimorca_dataset_mapped = slimorca_dataset.map(
                        extract_texts_from_slimorca_sample,
                        batched=False,
                        num_proc=8,  # Use 8 processes for parallel processing
                        remove_columns=slimorca_dataset.column_names,
                        desc="Processing SlimOrca"
                    )

                    # Extract all texts
                    slimorca_texts = []
                    for item in slimorca_dataset_mapped:
                        slimorca_texts.extend(item["texts"])

                    texts.extend(slimorca_texts)
                    logger.info(f"Extracted {len(slimorca_texts)} texts from SlimOrca ({len(slimorca_dataset)} samples)")
                except Exception as e:
                    logger.warning(f"Failed to load SlimOrca: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            else:
                # Regular dataset loading
                logger.info(f"Loading dataset: {ds_name}")
                if dataset_config_name:
                    dataset = load_dataset(ds_name, dataset_config_name, split="train")
                else:
                    dataset = load_dataset(ds_name, split="train")
                if num_samples:
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                # Extract texts from dataset
                if hasattr(dataset, text_column):
                    ds_texts = getattr(dataset, text_column)
                else:
                    ds_texts = [item[text_column] for item in dataset]
                # Ensure texts is a list
                if not isinstance(ds_texts, list):
                    ds_texts = list(ds_texts)
                texts.extend(ds_texts)
                logger.info(f"Extracted {len(ds_texts)} texts from {ds_name}")

        # Filter out empty texts (should already be filtered, but double-check)
        texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]

        # Check if we have any texts
        if len(texts) == 0:
            raise ValueError(
                f"No texts extracted from datasets: {dataset_name}! "
                "Please check if the datasets are loaded correctly and contain the expected fields."
            )

        # Limit total samples if specified
        total_texts = len(texts)
        if num_samples and total_texts > num_samples:
            import random
            random.seed(42)  # For reproducibility
            texts = random.sample(texts, num_samples)
            logger.info(f"Randomly sampled {num_samples} samples from {total_texts} total texts")
        else:
            logger.info(f"Using all {total_texts} extracted texts from {len(dataset_names)} dataset(s)")
    else:
        # Use dummy data for testing
        logger.info("Using dummy dataset")
        texts = [
            "Hello, world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can learn complex patterns from data.",
        ] * 100  # Repeat for more samples

    logger.info(f"Prepared {len(texts)} samples")

    # Create dataset
    dataset = TranslationDataset(
        texts=texts,
        translator=translator,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train translation model for decryption")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="facebook/nllb-200-3.3B",
                        help="Path to pretrained model")
    parser.add_argument("--original_tokenizer_path", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to original tokenizer")
    parser.add_argument("--alien_tokenizer_path", type=str,
                        default=os.environ.get("ALIEN_TOKENIZER_PATH"),
                        required="ALIEN_TOKENIZER_PATH" not in os.environ,
                        help="Path to alien tokenizer (or set ALIEN_TOKENIZER_PATH)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for models")

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="magpie",
                        help="Dataset name from HuggingFace. Use 'magpie' for Magpie datasets (default)")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text (not used for Magpie datasets)")
    parser.add_argument("--max_source_length", type=int, default=512,
                        help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Maximum target sequence length")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to use (for testing)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/translator_model",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 precision")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--report_to", type=str, default="none",
                        help="Reporting tool (wandb, tensorboard, etc.)")

    args = parser.parse_args()

    # Load tokenizers
    logger.info("Loading tokenizers...")
    original_tokenizer, alien_tokenizer = load_tokenizers(
        args.original_tokenizer_path,
        args.alien_tokenizer_path
    )

    # Build translator
    logger.info("Building translator...")
    translator = build_translator(original_tokenizer, alien_tokenizer)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare dataset
    logger.info("Preparing dataset...")
    train_dataset = prepare_dataset(
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        translator=translator,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_samples=args.num_samples,
        dataset_config_name=args.dataset_config_name,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    # DDP will be automatically used when running with torchrun (no FSDP)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        # DDP settings (automatically enabled with torchrun)
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        # Other settings
        remove_unused_columns=False,
        predict_with_generate=False,
        include_inputs_for_metrics=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss}")


if __name__ == "__main__":
    main()
