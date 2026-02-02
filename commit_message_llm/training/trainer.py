"""Trainer wrapper for commit message LLM training."""

import math
from typing import Any, Optional

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from commit_message_llm.config.schema import ConfigSchema, TrainingConfig
from commit_message_llm.model.setup import ModelSetup
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


class TrainingResult:
    """Container for training results."""

    def __init__(
        self,
        train_result: dict[str, Any],
        val_metrics: Optional[dict[str, float]] = None,
        test_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Initialize TrainingResult.

        Args:
            train_result: Raw training result from Trainer.
            val_metrics: Validation metrics.
            test_metrics: Test metrics.
        """
        self.train_result = train_result
        self.val_metrics = val_metrics or {}
        self.test_metrics = test_metrics or {}

    @property
    def train_runtime(self) -> float:
        """Training runtime in seconds."""
        return self.train_result.get("train_runtime", 0.0)

    @property
    def train_samples_per_second(self) -> float:
        """Training throughput in samples per second."""
        return self.train_result.get("train_samples_per_second", 0.0)

    @property
    def val_loss(self) -> Optional[float]:
        """Validation loss."""
        return self.val_metrics.get("val_loss")

    @property
    def val_perplexity(self) -> Optional[float]:
        """Validation perplexity."""
        loss = self.val_loss
        return math.exp(loss) if loss is not None else None

    @property
    def test_loss(self) -> Optional[float]:
        """Test loss."""
        return self.test_metrics.get("test_loss")

    @property
    def test_perplexity(self) -> Optional[float]:
        """Test perplexity."""
        loss = self.test_loss
        return math.exp(loss) if loss is not None else None


class TrainerWrapper:
    """
    Wrapper around Hugging Face Trainer for commit message LLM training.

    This class handles the setup and execution of training, including
    configuration, data collation, and callbacks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: ConfigSchema,
    ) -> None:
        """
        Initialize the TrainerWrapper.

        Args:
            model: The model to train.
            tokenizer: The tokenizer.
            config: Full configuration object.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_config = config.training
        self._trainer: Optional[Trainer] = None

    def _create_data_collator(self) -> DataCollatorForSeq2Seq:
        """Create the data collator for training."""
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    def _create_training_arguments(self) -> TrainingArguments:
        """Create the TrainingArguments object."""
        tc = self.training_config

        return TrainingArguments(
            output_dir=tc.output_dir,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            per_device_eval_batch_size=tc.per_device_eval_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            num_train_epochs=tc.num_train_epochs,
            lr_scheduler_type=tc.lr_scheduler_type,
            warmup_ratio=tc.warmup_ratio,
            logging_steps=tc.logging_steps,
            max_steps=tc.max_steps,
            save_steps=tc.save_steps,
            eval_steps=tc.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            fp16=tc.fp16,
            bf16=tc.bf16,
            report_to=tc.report_to,
            remove_unused_columns=False,
            dataloader_num_workers=tc.dataloader_num_workers,
            dataloader_pin_memory=tc.dataloader_pin_memory,
            gradient_checkpointing=tc.gradient_checkpointing,
            optim=tc.optim,
            max_grad_norm=tc.max_grad_norm,
            group_by_length=tc.group_by_length,
            seed=tc.seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def get_trainer(
        self,
        train_dataset,
        eval_dataset,
    ) -> Trainer:
        """
        Get or create the Trainer instance.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.

        Returns:
            Configured Trainer instance.
        """
        if self._trainer is not None:
            return self._trainer

        from commit_message_llm.training.callbacks import (
            EarlyStoppingCallback,
            LoggingCallback,
        )

        logger.info("Creating Trainer...")

        self._trainer = Trainer(
            model=self.model,
            args=self._create_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self._create_data_collator(),
            callbacks=[
                EarlyStoppingCallback(
                    patience=self.training_config.early_stopping_patience,
                    threshold=self.training_config.early_stopping_threshold,
                ),
                LoggingCallback(log_every_n_steps=self.training_config.logging_steps),
            ],
        )

        return self._trainer

    def train(
        self,
        train_dataset,
        eval_dataset,
    ) -> TrainingResult:
        """
        Run the training loop.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.

        Returns:
            TrainingResult with metrics.
        """
        trainer = self.get_trainer(train_dataset, eval_dataset)

        logger.info("Starting training...")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Evaluation samples: {len(eval_dataset)}")
        logger.info(f"  Max steps: {self.training_config.max_steps}")
        logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  Batch size: {self.training_config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.training_config.gradient_accumulation_steps}")

        train_result = trainer.train()

        logger.info("Training completed.")
        logger.info(f"  Time: {train_result.metrics['train_runtime']:.2f} seconds")
        logger.info(f"  Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

        return TrainingResult(train_result.metrics)

    def evaluate(
        self,
        eval_dataset,
        test_dataset: Optional[Any] = None,
    ) -> tuple[Optional[dict[str, float]], Optional[dict[str, float]]]:
        """
        Evaluate the model on validation and test sets.

        Args:
            eval_dataset: Validation dataset.
            test_dataset: Optional test dataset.

        Returns:
            Tuple of (val_metrics, test_metrics).
        """
        if self._trainer is None:
            logger.warning("Trainer not initialized. Call train() first.")
            return None, None

        val_metrics = None
        test_metrics = None

        # Evaluate validation set
        if eval_dataset is not None:
            logger.info("Evaluating on validation set...")
            val_metrics = self._trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="val")

            val_loss = val_metrics.get("val_loss")
            val_ppl = math.exp(val_loss) if val_loss is not None else None

            logger.info(f"  Validation loss: {val_loss:.4f}")
            logger.info(f"  Validation perplexity: {val_ppl:.2f}")

        # Evaluate test set
        if test_dataset is not None:
            logger.info("Evaluating on test set...")
            test_metrics = self._trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

            test_loss = test_metrics.get("test_loss")
            test_ppl = math.exp(test_loss) if test_loss is not None else None

            logger.info(f"  Test loss: {test_loss:.4f}")
            logger.info(f"  Test perplexity: {test_ppl:.2f}")

        return val_metrics, test_metrics

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save the trained model.

        Args:
            output_dir: Directory to save to. If None, uses config output_dir.
        """
        if self._trainer is None:
            logger.warning("Trainer not initialized. Nothing to save.")
            return

        output_dir = output_dir or self.training_config.output_dir

        logger.info(f"Saving model to {output_dir}...")
        self._trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        logger.info("Model saved.")


__all__ = [
    "TrainerWrapper",
    "TrainingResult",
]
