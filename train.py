#!/usr/bin/env python3
"""
Training script for commit message LLM.

This script fine-tunes a small language model using QLoRA to generate
commit messages from Git diffs.
"""

import sys
from pathlib import Path

# Set up default config path
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    sys.path.insert(0, str(Path(__file__).parent))
    from commit_message_llm.config import set_default_config_path
    set_default_config_path(config_path)

import math
from typing import Any

import torch
from datasets import Dataset

from commit_message_llm.cli import parse_args_train, update_config_from_args
from commit_message_llm.config import load_config
from commit_message_llm.config.schema import ConfigSchema
from commit_message_llm.data import DataProcessor, DatasetSplit
from commit_message_llm.data.tokenizer import TextTokenizer
from commit_message_llm.evaluation import Evaluator
from commit_message_llm.model import ModelSetup
from commit_message_llm.training import TrainerWrapper
from commit_message_llm.utils import setup_logging, get_gpu_memory_info, print_gpu_utilization
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_or_tokenize_dataset(
    config: ConfigSchema,
    tokenizer: Any,
    data_processor: DataProcessor,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load tokenized datasets from cache or tokenize from scratch.

    Args:
        config: Configuration object.
        tokenizer: Tokenizer instance.
        data_processor: DataProcessor instance.

    Returns:
        Tuple of (train, validation, test) tokenized datasets.
    """
    from datasets import load_from_disk

    text_tokenizer = TextTokenizer(tokenizer, config.data, config.model)

    # Check if we can use cached tokenized data
    if config.data.use_cached_tokenized:
        cached_paths = {
            "train": config.data.tokenized_train_path,
            "validation": config.data.tokenized_val_path,
            "test": config.data.tokenized_test_path,
        }

        all_exist = all(
            Path(p).exists() and (Path(p) / "dataset_info.json").exists()
            for p in cached_paths.values() if p
        )

        if all_exist:
            logger.info("Loading cached tokenized datasets...")
            train_ds = load_from_disk(cached_paths["train"])  # type: ignore
            val_ds = load_from_disk(cached_paths["validation"])  # type: ignore
            test_ds = load_from_disk(cached_paths["test"])  # type: ignore
            logger.info(f"  Train: {len(train_ds)} samples")
            logger.info(f"  Validation: {len(val_ds)} samples")
            logger.info(f"  Test: {len(test_ds)} samples")
            return train_ds, val_ds, test_ds

    # Tokenize from scratch
    logger.info("Tokenizing datasets from scratch...")

    splits = data_processor.get_splits()

    train_ds = text_tokenizer.tokenize_dataset(splits.train)
    val_ds = text_tokenizer.tokenize_dataset(splits.validation)
    test_ds = text_tokenizer.tokenize_dataset(splits.test)

    # Save to cache
    if config.data.tokenized_train_path:
        logger.info("Saving tokenized datasets to cache...")
        train_ds.save_to_disk(config.data.tokenized_train_path)
        val_ds.save_to_disk(config.data.tokenized_val_path)
        test_ds.save_to_disk(config.data.tokenized_test_path)

    return train_ds, val_ds, test_ds


def print_gpu_info() -> None:
    """Print GPU information."""
    info = get_gpu_memory_info(0)
    if info:
        logger.info(f"GPU: {info.total_mb} MB total memory")
    print_gpu_utilization()


def main() -> int:
    """
    Main training function.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Parse arguments
    args = parse_args_train()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        config = ConfigSchema()

    # Update config with CLI args
    config = update_config_from_args(config, args)

    # Set up logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file,
        log_to_console=config.logging.log_to_console,
    )

    logger.info("=" * 80)
    logger.info("Commit Message LLM Training")
    logger.info("=" * 80)

    # Print GPU info
    print_gpu_info()

    # Set random seeds
    torch.manual_seed(config.training.seed)

    # Initialize model setup
    model_setup = ModelSetup(config.model)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = model_setup.load_tokenizer()

    # Initialize data processor
    data_processor = DataProcessor(config.data)

    # Infer columns if not specified
    if config.data.diff_column is None or config.data.message_column is None:
        data_processor.infer_columns()

    # Get tokenized datasets
    train_ds, val_ds, test_ds = load_or_tokenize_dataset(
        config,
        tokenizer,
        data_processor,
    )

    # Validate datasets are not empty
    if len(train_ds) == 0:
        logger.error("Training dataset is empty!")
        return 1

    # Load and prepare model
    logger.info("Loading model...")
    model = model_setup.load_model()

    # Prepare for k-bit training
    model = model_setup.prepare_for_kbit_training(model)

    # Apply LoRA
    model = model_setup.apply_lora(model)

    # Print GPU info after model load
    print_gpu_info()

    # Create trainer
    trainer_wrapper = TrainerWrapper(model, tokenizer, config)

    # Train or eval only
    if args.eval_only:
        logger.info("Running evaluation only...")

        val_metrics, test_metrics = trainer_wrapper.evaluate(val_ds, test_ds)

        logger.info("Evaluation Results:")
        if val_metrics:
            val_loss = val_metrics.get("val_loss")
            logger.info(f"  Validation loss: {val_loss:.4f}")
            logger.info(f"  Validation perplexity: {math.exp(val_loss):.2f}")
        if test_metrics:
            test_loss = test_metrics.get("test_loss")
            logger.info(f"  Test loss: {test_loss:.4f}")
            logger.info(f"  Test perplexity: {math.exp(test_loss):.2f}")

        return 0

    # Run training
    train_result = trainer_wrapper.train(train_ds, val_ds)

    # Evaluate
    if not args.skip_eval:
        val_metrics, test_metrics = trainer_wrapper.evaluate(val_ds, test_ds)

        logger.info("=" * 80)
        logger.info("Training Results:")
        logger.info(f"  Training time: {train_result.train_runtime:.2f} seconds")
        logger.info(f"  Samples/second: {train_result.train_samples_per_second:.2f}")

        if val_metrics:
            val_loss = val_metrics.get("val_loss")
            logger.info(f"  Validation loss: {val_loss:.4f}")
            logger.info(f"  Validation perplexity: {math.exp(val_loss):.2f}")

        if test_metrics:
            test_loss = test_metrics.get("test_loss")
            logger.info(f"  Test loss: {test_loss:.4f}")
            logger.info(f"  Test perplexity: {math.exp(test_loss):.2f}")

        logger.info("=" * 80)

        # Qualitative evaluation
        evaluator = Evaluator(model, tokenizer, config)
        evaluator.sample_eval(test_ds)

    # Save model
    trainer_wrapper.save_model()

    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
