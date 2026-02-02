"""LoRA configuration utilities for commit message LLM."""

from peft import LoraConfig

from commit_message_llm.config.schema import ModelConfig


def create_lora_config(config: ModelConfig) -> LoraConfig:
    """
    Create a LoRA configuration for fine-tuning.

    Args:
        config: Model configuration object.

    Returns:
        LoraConfig instance for use with PEFT.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )


__all__ = [
    "create_lora_config",
]
