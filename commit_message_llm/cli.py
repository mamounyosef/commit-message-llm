"""Command-line interface for commit message LLM."""

import argparse
from pathlib import Path
from typing import Any


def parse_args_train() -> argparse.Namespace:
    """
    Parse command-line arguments for training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for commit message generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name on Hugging Face",
    )
    data_group.add_argument(
        "--train-samples",
        type=int,
        help="Number of training samples to use",
    )
    data_group.add_argument(
        "--val-samples",
        type=int,
        help="Number of validation samples to use",
    )
    data_group.add_argument(
        "--test-samples",
        type=int,
        help="Number of test samples to use",
    )
    data_group.add_argument(
        "--use-cached-tokenized",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Use cached tokenized data if available",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-id",
        type=str,
        help="Base model ID on Hugging Face",
    )
    model_group.add_argument(
        "--max-length",
        type=int,
        help="Maximum sequence length",
    )
    model_group.add_argument(
        "--lora-r",
        type=int,
        help="LoRA rank",
    )
    model_group.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha",
    )

    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for checkpoints",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        help="Per-device batch size",
    )
    training_group.add_argument(
        "--gradient-accumulation",
        type=int,
        help="Gradient accumulation steps",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    training_group.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps",
    )
    training_group.add_argument(
        "--epochs",
        type=float,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Early stopping patience",
    )

    # Logging
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    logging_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path",
    )
    logging_group.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires existing checkpoint)",
    )

    return parser.parse_args()


def parse_args_infer() -> argparse.Namespace:
    """
    Parse command-line arguments for inference.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate commit messages from Git diffs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="qwen2.5-coder-0.5b-qlora",
        help="Path to trained model or model ID",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Don't load LoRA adapters (load base model only)",
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--diff",
        type=str,
        help="Git diff text as string",
    )
    input_group.add_argument(
        "--diff-file",
        type=str,
        help="Path to file containing diff text",
    )
    input_group.add_argument(
        "--git",
        action="store_true",
        help="Use current git diff as input",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: read diff from stdin",
    )

    # Generation
    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate",
    )
    gen_group.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only if --do-sample is set)",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (only if --do-sample is set)",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for generated commit messages",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show full diff in output",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on",
    )

    return parser.parse_args()


def update_config_from_args(config: Any, args: argparse.Namespace) -> Any:
    """
    Update configuration with command-line arguments.

    Args:
        config: Configuration object.
        args: Parsed command-line arguments.

    Returns:
        Updated configuration object.
    """
    # Data
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.train_samples is not None:
        config.data.train_samples = args.train_samples
    if args.val_samples is not None:
        config.data.val_samples = args.val_samples
    if args.test_samples is not None:
        config.data.test_samples = args.test_samples
    if args.use_cached_tokenized is not None:
        config.data.use_cached_tokenized = args.use_cached_tokenized

    # Model
    if args.model_id:
        config.model.model_id = args.model_id
    if args.max_length is not None:
        config.model.max_length = args.max_length
    if args.lora_r is not None:
        config.model.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.model.lora_alpha = args.lora_alpha

    # Training
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience

    # Logging
    if args.log_level:
        config.logging.level = args.log_level
    if args.log_file:
        config.logging.log_file = args.log_file
    if args.no_console_log:
        config.logging.log_to_console = False

    # Seed
    if args.seed is not None:
        config.training.seed = args.seed

    return config


__all__ = [
    "parse_args_train",
    "parse_args_infer",
    "update_config_from_args",
]
