"""Evaluation metrics for commit message LLM."""

import math
from collections.abc import Mapping


def perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss.

    Returns:
        Perplexity value.
    """
    try:
        return math.exp(loss)
    except (OverflowError, ValueError):
        return float("inf")


def compute_metrics(eval_preds: tuple) -> dict[str, float]:
    """
    Compute metrics for evaluation.

    Args:
        eval_preds: Tuple of (predictions, labels) from Trainer.

    Returns:
        Dictionary of metric names and values.
    """
    logits, labels = eval_preds

    # Calculate loss manually if needed
    # For now, we'll rely on the trainer's built-in loss computation
    return {}


__all__ = [
    "perplexity",
    "compute_metrics",
]
