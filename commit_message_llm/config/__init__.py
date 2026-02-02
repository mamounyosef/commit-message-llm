"""Configuration management for commit message LLM training."""

from pathlib import Path
from typing import Any
import yaml

from commit_message_llm.config.schema import ConfigSchema, TrainingConfig, ModelConfig, DataConfig


# Default config path relative to package root
_default_config_path: Path | None = None


def set_default_config_path(path: Path | str) -> None:
    """Set the default configuration file path."""
    global _default_config_path
    _default_config_path = Path(path)


def default_config_path() -> Path | None:
    """Get the default configuration file path."""
    return _default_config_path


def load_config(config_path: Path | str | None = None) -> ConfigSchema:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses default_config_path().

    Returns:
        ConfigSchema object with all configuration values.

    Raises:
        FileNotFoundError: If config file is not found.
        ValueError: If config file is invalid.
    """
    if config_path is None:
        config_path = _default_config_path

    if config_path is None:
        # Return default config
        return ConfigSchema()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    return ConfigSchema.from_dict(config_data)


def save_config(config: ConfigSchema, config_path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: ConfigSchema object to save.
        config_path: Path to save the YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


__all__ = [
    "load_config",
    "save_config",
    "set_default_config_path",
    "default_config_path",
    "ConfigSchema",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
]
