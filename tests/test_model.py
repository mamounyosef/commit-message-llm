"""Unit tests for model setup and configuration."""

import pytest
from unittest.mock import MagicMock, patch

from commit_message_llm.config.schema import ModelConfig
from commit_message_llm.model.lora import create_lora_config
from commit_message_llm.model.setup import ModelSetup


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_id == "Qwen/Qwen2.5-Coder-0.5B"
        assert config.load_in_4bit is True
        assert config.max_length == 512
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = ModelConfig(lora_r=32)
        d = config.to_dict()
        assert d["lora_r"] == 32
        assert d["model_id"] == "Qwen/Qwen2.5-Coder-0.5B"

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "model": {
                "model_id": "test/model",
                "lora_r": 64,
            }
        }
        config = ModelConfig.from_dict(data)
        assert config.model_id == "test/model"
        assert config.lora_r == 64


class TestLoraConfig:
    """Tests for LoRA configuration creation."""

    def test_create_lora_config(self) -> None:
        """Test creating LoRA config from ModelConfig."""
        model_config = ModelConfig(lora_r=32, lora_alpha=64)
        lora_config = create_lora_config(model_config)

        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert lora_config.task_type == "CAUSAL_LM"
        assert lora_config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_custom_target_modules(self) -> None:
        """Test custom target modules."""
        model_config = ModelConfig(
            lora_target_modules=["q_proj", "v_proj"]
        )
        lora_config = create_lora_config(model_config)
        assert lora_config.target_modules == ["q_proj", "v_proj"]


class TestModelSetup:
    """Tests for ModelSetup class."""

    def test_initialization(self) -> None:
        """Test ModelSetup initialization."""
        config = ModelConfig()
        setup = ModelSetup(config)
        assert setup.config == config
        assert setup._model is None
        assert setup._tokenizer is None

    @patch("commit_message_llm.model.setup.AutoTokenizer")
    def test_load_tokenizer(self, mock_tokenizer_class) -> None:
        """Test tokenizer loading."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = ModelConfig()
        setup = ModelSetup(config)

        tokenizer = setup.load_tokenizer()

        assert tokenizer == mock_tokenizer
        mock_tokenizer_class.from_pretrained.assert_called_once()
        assert tokenizer.pad_token == tokenizer.eos_token

    def test_get_quantization_config(self) -> None:
        """Test quantization config creation."""
        config = ModelConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )
        setup = ModelSetup(config)

        quant_config = setup.get_quantization_config()

        assert quant_config.load_in_4bit is True
        assert quant_config.bnb_4bit_quant_type == "nf4"
        assert quant_config.bnb_4bit_use_double_quant is True

    def test_get_compute_dtype(self) -> None:
        """Test compute dtype conversion."""
        setup = ModelSetup(ModelConfig())

        assert setup._get_compute_dtype() == torch.bfloat16  # type: ignore

        config = ModelConfig(bnb_4bit_compute_dtype="float16")
        setup = ModelSetup(config)
        assert setup._get_compute_dtype() == torch.float16  # type: ignore

    @patch("commit_message_llm.model.setup.prepare_model_for_kbit_training")
    def test_prepare_for_kbit_training(self, mock_prepare) -> None:
        """Test model preparation for k-bit training."""
        mock_model = MagicMock()
        mock_prepare.return_value = mock_model

        config = ModelConfig()
        setup = ModelSetup(config)
        setup._model = mock_model

        result = setup.prepare_for_kbit_training()

        assert result == mock_model
        mock_prepare.assert_called_once_with(mock_model)
        assert mock_model.gradient_checkpointing_enable.called
        assert mock_model.config.use_cache is False

    @patch("commit_message_llm.model.setup.get_peft_model")
    @patch("commit_message_llm.model.setup.create_lora_config")
    def test_apply_lora(self, mock_create_lora, mock_get_peft) -> None:
        """Test applying LoRA to model."""
        mock_model = MagicMock()
        mock_lora_config = MagicMock()
        mock_create_lora.return_value = mock_lora_config
        mock_get_peft.return_value = mock_model

        config = ModelConfig()
        setup = ModelSetup(config)
        setup._model = mock_model

        result = setup.apply_lora()

        assert result == mock_model
        mock_create_lora.assert_called_once_with(config)
        mock_get_peft.assert_called_once()


# Import torch for type checking
import torch
