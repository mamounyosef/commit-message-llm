"""Unit tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from commit_message_llm.config import load_config, save_config
from commit_message_llm.config.schema import (
    ConfigSchema,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
)


class TestConfigSchema:
    """Tests for ConfigSchema dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ConfigSchema()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = ConfigSchema()
        d = config.to_dict()
        assert "model" in d
        assert "data" in d
        assert "training" in d
        assert "logging" in d

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "model": {
                "model_id": "test/model",
                "lora_r": 32,
            },
            "training": {
                "learning_rate": 1e-4,
                "max_steps": 1000,
            }
        }
        config = ConfigSchema.from_dict(data)
        assert config.model.model_id == "test/model"
        assert config.model.lora_r == 32
        assert config.training.learning_rate == 1e-4
        assert config.training.max_steps == 1000


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = ModelConfig()
        assert config.model_id == "Qwen/Qwen2.5-Coder-0.5B"
        assert config.load_in_4bit is True
        assert config.max_length == 512
        assert config.lora_r == 16

    def test_from_dict(self) -> None:
        """Test creating from dict."""
        data = {"model": {"model_id": "custom/model", "lora_r": 64}}
        config = ModelConfig.from_dict(data)
        assert config.model_id == "custom/model"
        assert config.lora_r == 64


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = DataConfig()
        assert config.dataset_name == "Maxscha/commitbench"
        assert config.min_diff_chars == 50
        assert config.train_samples == 120000

    def test_from_dict(self) -> None:
        """Test creating from dict."""
        data = {"data": {"dataset_name": "custom/dataset", "train_samples": 50000}}
        config = DataConfig.from_dict(data)
        assert config.dataset_name == "custom/dataset"
        assert config.train_samples == 50000


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = TrainingConfig()
        assert config.output_dir == "qwen2.5-coder-0.5b-qlora"
        assert config.per_device_train_batch_size == 6
        assert config.learning_rate == 1.8e-4
        assert config.max_steps == 6000

    def test_from_dict(self) -> None:
        """Test creating from dict."""
        data = {"training": {"batch_size": 4, "learning_rate": 1e-4}}
        config = TrainingConfig.from_dict(data)
        # batch_size maps to per_device_train_batch_size
        assert config.per_device_train_batch_size == 6  # default, not in dict


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.log_file == "logs/training.log"
        assert config.log_to_console is True


class TestConfigFile:
    """Tests for config file loading and saving."""

    def test_save_and_load_config(self) -> None:
        """Test saving and loading config file."""
        config = ConfigSchema()
        config.model.lora_r = 32
        config.training.max_steps = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            save_config(config, config_path)
            assert config_path.exists()

            loaded_config = load_config(config_path)
            assert loaded_config.model.lora_r == 32
            assert loaded_config.training.max_steps == 1000

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_invalid_yaml_returns_default(self) -> None:
        """Test loading invalid YAML returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty.yaml"
            config_path.write_text("")

            # Should return default config for empty file
            config = load_config(config_path)
            assert isinstance(config, ConfigSchema)

    def test_yaml_roundtrip(self) -> None:
        """Test config survives YAML roundtrip."""
        original = ConfigSchema()
        original.model.lora_r = 64
        original.data.train_samples = 50000
        original.training.early_stopping_patience = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"

            save_config(original, config_path)
            loaded = load_config(config_path)

            assert loaded.model.lora_r == 64
            assert loaded.data.train_samples == 50000
            assert loaded.training.early_stopping_patience == 10
