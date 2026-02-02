"""Text tokenization utilities for commit message LLM."""

from typing import Any

import torch
from transformers import PreTrainedTokenizer

from commit_message_llm.config.schema import DataConfig, ModelConfig
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


class TextTokenizer:
    """
    Handles tokenization of diff and message pairs for causal LM training.

    This class creates prompts in the format:
        diff + "\n\nCommit message:\n" + message

    And creates proper labels for causal language modeling where only
    the message portion contributes to the loss.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        model_config: ModelConfig,
    ) -> None:
        """
        Initialize the TextTokenizer.

        Args:
            tokenizer: Pre-trained tokenizer instance.
            data_config: Data configuration object.
            model_config: Model configuration object.
        """
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.model_config = model_config
        self.max_length = model_config.max_length
        self.sep = data_config.separator

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._diff_col = data_config.diff_column or "diff"
        self._msg_col = data_config.message_column or "message"

    def tokenize_batch(self, batch: dict[str, list]) -> dict[str, list[list[int]]]:
        """
        Tokenize a batch of examples for causal LM training.

        Args:
            batch: Batch dict with diff and message columns.

        Returns:
            Dict with input_ids, attention_mask, and labels lists.
        """
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[list[int]] = []
        labels_list: list[list[int]] = []

        for diff, msg in zip(batch[self._diff_col], batch[self._msg_col]):
            prompt_text = diff + self.sep
            full_text = diff + self.sep + msg + self.tokenizer.eos_token

            # Tokenize prompt to find its length
            prompt_ids = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            # Tokenize full text
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            # Create labels: -100 for prompt (ignored in loss), actual ids for message
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:len(input_ids)]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    def tokenize_dataset(
        self,
        dataset,
        remove_columns: bool = True,
    ):
        """
        Tokenize an entire dataset.

        Args:
            dataset: Dataset to tokenize.
            remove_columns: Whether to remove original columns after tokenization.

        Returns:
            Tokenized dataset.
        """
        column_names = dataset.column_names if remove_columns else None

        tokenized = dataset.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing",
        )

        return tokenized

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text: str, truncation: bool = True, max_length: int | None = None) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode.
            truncation: Whether to truncate.
            max_length: Maximum length (uses self.max_length if None).

        Returns:
            List of token IDs.
        """
        if max_length is None:
            max_length = self.max_length

        return self.tokenizer.encode(text, truncation=truncation, max_length=max_length)


__all__ = [
    "TextTokenizer",
]
