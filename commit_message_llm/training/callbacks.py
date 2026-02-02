"""Training callbacks for commit message LLM."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback
from transformers.trainer_utils import has_length

from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback for Hugging Face Transformers Trainer.

    Stops training when validation loss doesn't improve for a given number of evaluations.
    """

    patience: int = 5
    """Number of evaluations with no improvement to wait before stopping."""

    threshold: float = 0.0
    """Minimum change to qualify as an improvement."""

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as improvement."""

    early_stopping_metric: str = "eval_loss"
    """Metric to monitor for early stopping."""

    lower_is_better: bool = True
    """Whether lower values of the metric are better."""

    def __post_init__(self) -> None:
        """Initialize the callback state."""
        self.best_metric: float | None = None
        self.patience_counter: int = 0
        self.best_step: int = 0
        self.best_model_checkpoint: str | None = None

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Reset counters when training begins."""
        self.best_metric = None
        self.patience_counter = 0
        self.best_step = 0
        self.best_model_checkpoint = None
        logger.info(f"Early stopping enabled: patience={self.patience}, threshold={self.threshold}")

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics: Mapping[str, float],
        **kwargs,
    ) -> None:
        """
        Check for early stopping after each evaluation.

        Args:
            args: TrainingArguments.
            state: TrainerState.
            control: TrainerControl.
            metrics: Evaluation metrics.
        """
        metric = metrics.get(self.early_stopping_metric)

        if metric is None:
            logger.warning(
                f"Early stopping metric '{self.early_stopping_metric}' not found "
                f"in evaluation metrics: {list(metrics.keys())}"
            )
            return

        if self.best_metric is None:
            # First evaluation
            self.best_metric = metric
            self.best_step = state.global_step
            if state.best_model_checkpoint is not None:
                self.best_model_checkpoint = state.best_model_checkpoint
            logger.info(f"Initial {self.early_stopping_metric} = {metric:.4f}")
            return

        # Check if metric improved
        if self.lower_is_better:
            improved = metric < (self.best_metric - self.threshold)
        else:
            improved = metric > (self.best_metric + self.threshold)

        if improved:
            # Metric improved
            delta = abs(metric - self.best_metric)
            self.best_metric = metric
            self.best_step = state.global_step
            self.patience_counter = 0

            if state.best_model_checkpoint is not None:
                self.best_model_checkpoint = state.best_model_checkpoint

            logger.info(
                f"Early stopping: {self.early_stopping_metric} improved from "
                f"{self.best_metric:.4f} to {metric:.4f} (delta={delta:.4f})"
            )
        else:
            # No improvement
            self.patience_counter += 1
            logger.info(
                f"Early stopping: no improvement for {self.patience_counter}/{self.patience} evaluations. "
                f"Best {self.early_stopping_metric} = {self.best_metric:.4f} at step {self.best_step}"
            )

            if self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered! No improvement for {self.patience} evaluations. "
                    f"Best {self.early_stopping_metric} = {self.best_metric:.4f} at step {self.best_step}"
                )
                control.should_training_stop = True


@dataclass
class GPUMemoryCallback(TrainerCallback):
    """
    Callback to log GPU memory usage during training.
    """

    log_every_n_steps: int = 100

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Log GPU memory after each N steps."""
        if state.global_step % self.log_every_n_steps == 0:
            from commit_message_llm.utils.gpu_utils import get_gpu_memory_info

            info = get_gpu_memory_info(0)
            if info:
                logger.info(f"GPU memory at step {state.global_step}: {info.used_mb} MB / {info.total_mb} MB")


@dataclass
class LoggingCallback(TrainerCallback):
    """
    Enhanced logging callback for training progress.
    """

    log_every_n_steps: int = 50

    def on_log(self, args, state, control, logs: dict[str, float] | None = None, **kwargs) -> None:
        """Log training metrics."""
        if logs is None:
            return

        if state.global_step % self.log_every_n_steps == 0:
            log_items = []
            for k, v in logs.items():
                if isinstance(v, float):
                    log_items.append(f"{k}={v:.4f}")
                else:
                    log_items.append(f"{k}={v}")

            if log_items:
                logger.info(f"Step {state.global_step}: {', '.join(log_items)}")


__all__ = [
    "EarlyStoppingCallback",
    "GPUMemoryCallback",
    "LoggingCallback",
]
