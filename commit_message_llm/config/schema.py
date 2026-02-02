"""Configuration schema for commit message LLM training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    # Base model
    model_id: str = "Qwen/Qwen2.5-Coder-0.5B"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Tokenization
    max_length: int = 512
    use_fast_tokenizer: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Generation
    max_new_tokens: int = 30
    do_sample: bool = False
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3

    # Attention
    attn_implementation: str = "sdpa"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "max_length": self.max_length,
            "use_fast_tokenizer": self.use_fast_tokenizer,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_bias": self.lora_bias,
            "lora_target_modules": self.lora_target_modules,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "attn_implementation": self.attn_implementation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        model_data = data.get("model", {})
        return cls(
            model_id=model_data.get("model_id", "Qwen/Qwen2.5-Coder-0.5B"),
            load_in_4bit=model_data.get("load_in_4bit", True),
            bnb_4bit_quant_type=model_data.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_data.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=model_data.get("bnb_4bit_compute_dtype", "bfloat16"),
            max_length=model_data.get("max_length", 512),
            use_fast_tokenizer=model_data.get("use_fast_tokenizer", True),
            lora_r=model_data.get("lora_r", 16),
            lora_alpha=model_data.get("lora_alpha", 32),
            lora_dropout=model_data.get("lora_dropout", 0.05),
            lora_bias=model_data.get("lora_bias", "none"),
            lora_target_modules=model_data.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            max_new_tokens=model_data.get("max_new_tokens", 30),
            do_sample=model_data.get("do_sample", False),
            repetition_penalty=model_data.get("repetition_penalty", 1.1),
            no_repeat_ngram_size=model_data.get("no_repeat_ngram_size", 3),
            attn_implementation=model_data.get("attn_implementation", "sdpa"),
        )


@dataclass
class DataConfig:
    """Data processing configuration parameters."""

    # Dataset
    dataset_name: str = "Maxscha/commitbench"
    dataset_cache_dir: str | None = None

    # Column names (will be auto-inferred if not set)
    diff_column: str | None = None
    message_column: str | None = None

    # Filtering
    min_diff_chars: int = 50
    max_diff_chars: int = 8000
    min_message_chars: int = 3
    min_message_words: int = 3

    # Dataset reduction
    train_samples: int = 120000
    val_samples: int = 15000
    test_samples: int = 15000
    shuffle_seed: int = 9105

    # Text separator
    separator: str = "\n\nCommit message:\n"

    # Bad message patterns
    bad_exact_messages: set[str] = field(default_factory=lambda: {
        "update", "updated", "fix", "fixed", "wip", ".", "..", "...", "temp", "test",
    })

    # Precomputed tokenized data paths
    tokenized_train_path: str | None = "tokenized_data/train"
    tokenized_val_path: str | None = "tokenized_data/validation"
    tokenized_test_path: str | None = "tokenized_data/test"

    cleaned_data_path: str | None = "cleaned_data"

    # Use pre-tokenized data if available
    use_cached_tokenized: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_cache_dir": self.dataset_cache_dir,
            "diff_column": self.diff_column,
            "message_column": self.message_column,
            "min_diff_chars": self.min_diff_chars,
            "max_diff_chars": self.max_diff_chars,
            "min_message_chars": self.min_message_chars,
            "min_message_words": self.min_message_words,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "shuffle_seed": self.shuffle_seed,
            "separator": self.separator,
            "bad_exact_messages": list(self.bad_exact_messages),
            "tokenized_train_path": self.tokenized_train_path,
            "tokenized_val_path": self.tokenized_val_path,
            "tokenized_test_path": self.tokenized_test_path,
            "cleaned_data_path": self.cleaned_data_path,
            "use_cached_tokenized": self.use_cached_tokenized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        data_config = data.get("data", {})
        return cls(
            dataset_name=data_config.get("dataset_name", "Maxscha/commitbench"),
            dataset_cache_dir=data_config.get("dataset_cache_dir"),
            diff_column=data_config.get("diff_column"),
            message_column=data_config.get("message_column"),
            min_diff_chars=data_config.get("min_diff_chars", 50),
            max_diff_chars=data_config.get("max_diff_chars", 8000),
            min_message_chars=data_config.get("min_message_chars", 3),
            min_message_words=data_config.get("min_message_words", 3),
            train_samples=data_config.get("train_samples", 120000),
            val_samples=data_config.get("val_samples", 15000),
            test_samples=data_config.get("test_samples", 15000),
            shuffle_seed=data_config.get("shuffle_seed", 9105),
            separator=data_config.get("separator", "\n\nCommit message:\n"),
            bad_exact_messages=set(data_config.get("bad_exact_messages", [
                "update", "updated", "fix", "fixed", "wip", ".", "..", "...", "temp", "test",
            ])),
            tokenized_train_path=data_config.get("tokenized_train_path", "tokenized_data/train"),
            tokenized_val_path=data_config.get("tokenized_val_path", "tokenized_data/validation"),
            tokenized_test_path=data_config.get("tokenized_test_path", "tokenized_data/test"),
            cleaned_data_path=data_config.get("cleaned_data_path", "cleaned_data"),
            use_cached_tokenized=data_config.get("use_cached_tokenized", True),
        )


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Output
    output_dir: str = "qwen2.5-coder-0.5b-qlora"

    # Batch sizes
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 6
    gradient_accumulation_steps: int = 8

    # Learning rate
    learning_rate: float = 1.8e-4
    num_train_epochs: float = 2.0
    max_steps: int = 6000

    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.04

    # Logging and saving
    logging_steps: int = 30
    save_steps: int = 300
    eval_steps: int = 300

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Optimization
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0

    # Data loading
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    group_by_length: bool = True

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    # Reporting
    report_to: list[str] = field(default_factory=list)

    # Seeds
    seed: int = 42

    # Evaluation
    eval_samples: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "optim": self.optim,
            "max_grad_norm": self.max_grad_norm,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "group_by_length": self.group_by_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "report_to": self.report_to,
            "seed": self.seed,
            "eval_samples": self.eval_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        training_data = data.get("training", {})
        return cls(
            output_dir=training_data.get("output_dir", "qwen2.5-coder-0.5b-qlora"),
            per_device_train_batch_size=training_data.get("per_device_train_batch_size", 6),
            per_device_eval_batch_size=training_data.get("per_device_eval_batch_size", 6),
            gradient_accumulation_steps=training_data.get("gradient_accumulation_steps", 8),
            learning_rate=training_data.get("learning_rate", 1.8e-4),
            num_train_epochs=training_data.get("num_train_epochs", 2.0),
            max_steps=training_data.get("max_steps", 6000),
            lr_scheduler_type=training_data.get("lr_scheduler_type", "cosine"),
            warmup_ratio=training_data.get("warmup_ratio", 0.04),
            logging_steps=training_data.get("logging_steps", 30),
            save_steps=training_data.get("save_steps", 300),
            eval_steps=training_data.get("eval_steps", 300),
            fp16=training_data.get("fp16", False),
            bf16=training_data.get("bf16", True),
            optim=training_data.get("optim", "paged_adamw_8bit"),
            max_grad_norm=training_data.get("max_grad_norm", 1.0),
            dataloader_num_workers=training_data.get("dataloader_num_workers", 8),
            dataloader_pin_memory=training_data.get("dataloader_pin_memory", True),
            group_by_length=training_data.get("group_by_length", True),
            gradient_checkpointing=training_data.get("gradient_checkpointing", True),
            early_stopping_patience=training_data.get("early_stopping_patience", 5),
            early_stopping_threshold=training_data.get("early_stopping_threshold", 0.0),
            report_to=training_data.get("report_to", []),
            seed=training_data.get("seed", 42),
            eval_samples=training_data.get("eval_samples", 5),
        )


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""

    level: str = "INFO"
    log_file: str | None = "logs/training.log"
    log_to_console: bool = True
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "log_file": self.log_file,
            "log_to_console": self.log_to_console,
            "format_string": self.format_string,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Create from dictionary."""
        logging_data = data.get("logging", {})
        return cls(
            level=logging_data.get("level", "INFO"),
            log_file=logging_data.get("log_file", "logs/training.log"),
            log_to_console=logging_data.get("log_to_console", True),
            format_string=logging_data.get("format_string", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )


@dataclass
class ConfigSchema:
    """Main configuration schema combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "logging": self.logging.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigSchema":
        """Create from dictionary."""
        return cls(
            model=ModelConfig.from_dict(data),
            data=DataConfig.from_dict(data),
            training=TrainingConfig.from_dict(data),
            logging=LoggingConfig.from_dict(data),
        )
