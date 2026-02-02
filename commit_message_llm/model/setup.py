"""Model setup utilities for commit message LLM."""

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from commit_message_llm.config.schema import ModelConfig
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


class ModelSetup:
    """
    Handles model and tokenizer loading, quantization, and LoRA setup.

    This class provides a unified interface for loading the base model,
    configuring quantization, and preparing the model for QLoRA training.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the ModelSetup.

        Args:
            config: Model configuration object.
        """
        self.config = config
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer.

        Returns:
            Loaded tokenizer instance.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        logger.info(f"Loading tokenizer from: {self.config.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            use_fast=self.config.use_fast_tokenizer,
        )

        # Set pad token if needed
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Tokenizer vocab size: {self._tokenizer.vocab_size}")
        return self._tokenizer

    def get_quantization_config(self) -> BitsAndBytesConfig:
        """
        Get the quantization configuration.

        Returns:
            BitsAndBytesConfig instance.
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self._get_compute_dtype(),
        )

    def _get_compute_dtype(self) -> torch.dtype:
        """
        Get the compute dtype from config string.

        Returns:
            torch.dtype for computation.
        """
        dtype_str = self.config.bnb_4bit_compute_dtype.lower()
        if dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float16":
            return torch.float16
        else:
            return torch.float32

    def load_model(self) -> PreTrainedModel:
        """
        Load the model with quantization configuration.

        Returns:
            Loaded model instance.
        """
        if self._model is not None:
            return self._model

        logger.info(f"Loading model from: {self.config.model_id}")

        quant_config = self.get_quantization_config()
        dtype = self._get_compute_dtype()

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=quant_config,
            device_map="auto",
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        )

        # Set to eval mode initially
        self._model.eval()

        logger.info(f"Model loaded. Dtype: {dtype}")
        return self._model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer, loading it if necessary."""
        if self._tokenizer is None:
            return self.load_tokenizer()
        return self._tokenizer

    def get_model(self) -> PreTrainedModel:
        """Get the model, loading it if necessary."""
        if self._model is None:
            return self.load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        return self.get_tokenizer()

    @property
    def model(self) -> PreTrainedModel:
        """Get the model."""
        return self.get_model()

    def prepare_for_kbit_training(self, model: PreTrainedModel | None = None) -> PreTrainedModel:
        """
        Prepare model for k-bit training.

        Args:
            model: Model to prepare. If None, uses self.model.

        Returns:
            Prepared model instance.
        """
        from peft import prepare_model_for_kbit_training

        if model is None:
            model = self.get_model()

        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        logger.info("Model prepared for k-bit training")
        return model

    def apply_lora(self, model: PreTrainedModel | None = None) -> PreTrainedModel:
        """
        Apply LoRA adapters to the model.

        Args:
            model: Model to apply LoRA to. If None, uses self.model.

        Returns:
            Model with LoRA adapters applied.
        """
        from peft import get_peft_model

        from commit_message_llm.model.lora import create_lora_config

        if model is None:
            model = self.get_model()

        lora_config = create_lora_config(self.config)

        logger.info("Applying LoRA adapters...")
        logger.info(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}")
        logger.info(f"  target_modules={lora_config.target_modules}")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info("LoRA adapters applied")
        return model


__all__ = [
    "ModelSetup",
]
