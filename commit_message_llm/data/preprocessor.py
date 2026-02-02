"""Data preprocessing and dataset management for commit message LLM."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from commit_message_llm.config.schema import DataConfig
from commit_message_llm.data.cleaning import (
    keep_example,
    preprocess_example,
    infer_columns,
)
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Container for processed dataset splits."""

    train: Dataset
    validation: Dataset
    test: Dataset

    def __len__(self) -> int:
        """Total number of examples across all splits."""
        return len(self.train) + len(self.validation) + len(self.test)


class DataProcessor:
    """
    Handles loading, cleaning, and preparing the dataset for training.

    This class manages the entire data pipeline from raw dataset loading
    to cleaned and filtered splits ready for tokenization.
    """

    def __init__(self, config: DataConfig) -> None:
        """
        Initialize the DataProcessor.

        Args:
            config: Data configuration object.
        """
        self.config = config
        self._dataset: DatasetDict | None = None
        self._cleaned_dataset: DatasetDict | None = None
        self._diff_col: str = config.diff_column or "diff"
        self._msg_col: str = config.message_column or "message"

    def load_dataset(self) -> DatasetDict:
        """
        Load the dataset from Hugging Face or cache.

        Returns:
            Loaded DatasetDict.
        """
        if self._dataset is not None:
            return self._dataset

        logger.info(f"Loading dataset: {self.config.dataset_name}")

        load_kwargs = {}
        if self.config.dataset_cache_dir:
            load_kwargs["cache_dir"] = self.config.dataset_cache_dir

        self._dataset = load_dataset(self.config.dataset_name, **load_kwargs)

        logger.info(f"Splits: {list(self._dataset.keys())}")
        for split_name, split in self._dataset.items():
            logger.info(f"  {split_name}: {len(split)} rows, columns={split.column_names}")

        return self._dataset

    def infer_columns(self) -> tuple[str, str]:
        """
        Infer diff and message column names from the dataset.

        Returns:
            Tuple of (diff_column, message_column).
        """
        ds = self.load_dataset()

        sample_split = "train" if "train" in ds else list(ds.keys())[0]
        diff_col, msg_col = infer_columns(ds[sample_split][0])

        if diff_col is None:
            diff_col = "diff"
        if msg_col is None:
            msg_col = "message"

        self._diff_col = diff_col
        self._msg_col = msg_col

        logger.info(f"Inferred columns -> diff: {diff_col}, message: {msg_col}")
        return diff_col, msg_col

    def clean_dataset(self) -> DatasetDict:
        """
        Clean and filter the dataset.

        Returns:
            Cleaned DatasetDict with filtered examples.
        """
        if self._cleaned_dataset is not None:
            return self._cleaned_dataset

        ds = self.load_dataset()

        # Preprocess (normalize newlines)
        logger.info("Preprocessing dataset...")
        ds_clean = ds.map(
            lambda ex: preprocess_example(ex, self._diff_col, self._msg_col),
            desc="Normalizing text",
        )

        # Filter examples
        logger.info("Filtering dataset...")
        ds_clean = ds_clean.filter(
            lambda ex: keep_example(
                ex,
                diff_col=self._diff_col,
                msg_col=self._msg_col,
                min_diff_chars=self.config.min_diff_chars,
                max_diff_chars=self.config.max_diff_chars,
                min_msg_chars=self.config.min_message_chars,
                min_msg_words=self.config.min_message_words,
            ),
            desc="Filtering examples",
        )

        # Log statistics
        for split in ds.keys():
            before = len(ds[split])
            after = len(ds_clean[split])
            pct = 100.0 * after / before if before else 0.0
            logger.info(f"  {split}: {after}/{before} kept ({pct:.2f}%)")

        self._cleaned_dataset = ds_clean
        return ds_clean

    def reduce_dataset(self, ds: DatasetDict | None = None) -> DatasetDict:
        """
        Reduce dataset size by sampling.

        Args:
            ds: DatasetDict to reduce. If None, uses cleaned dataset.

        Returns:
            Reduced DatasetDict.
        """
        if ds is None:
            ds = self.clean_dataset()

        def _cap(n: int, limit: int) -> int:
            return min(n, limit)

        # Shuffle and cap train split
        if len(ds["train"]) > self.config.train_samples:
            ds["train"] = ds["train"].shuffle(seed=self.config.shuffle_seed).select(
                range(_cap(len(ds["train"]), self.config.train_samples))
            )

        # Shuffle and cap validation split
        if len(ds["validation"]) > self.config.val_samples:
            ds["validation"] = ds["validation"].shuffle(seed=self.config.shuffle_seed).select(
                range(_cap(len(ds["validation"]), self.config.val_samples))
            )

        # Shuffle and cap test split
        if len(ds["test"]) > self.config.test_samples:
            ds["test"] = ds["test"].shuffle(seed=self.config.shuffle_seed).select(
                range(_cap(len(ds["test"]), self.config.test_samples))
            )

        logger.info("Reduced dataset:")
        logger.info(f"  train: {len(ds['train'])} samples")
        logger.info(f"  validation: {len(ds['validation'])} samples")
        logger.info(f"  test: {len(ds['test'])} samples")

        return ds

    def get_splits(self) -> DatasetSplit:
        """
        Get the cleaned and reduced dataset splits.

        Returns:
            DatasetSplit containing train, validation, and test splits.
        """
        ds = self.reduce_dataset()
        return DatasetSplit(
            train=ds["train"],
            validation=ds["validation"],
            test=ds["test"],
        )

    @property
    def diff_column(self) -> str:
        """Get the diff column name."""
        return self._diff_col

    @property
    def message_column(self) -> str:
        """Get the message column name."""
        return self._msg_col


__all__ = [
    "DataProcessor",
    "DatasetSplit",
]
