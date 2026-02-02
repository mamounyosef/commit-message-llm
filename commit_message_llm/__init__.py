"""
Commit Message LLM - Fine-tuning Qwen2.5-Coder-0.5B for commit message generation.

This package provides tools for fine-tuning a small language model using QLoRA
to generate clear, concise commit messages from Git diffs.
"""

__version__ = "0.1.0"
__author__ = "Commit Message LLM Contributors"

from commit_message_llm.config import load_config, default_config_path
from commit_message_llm.data import DataProcessor, DatasetSplit
from commit_message_llm.model import ModelSetup, LoraConfig
from commit_message_llm.training import Trainer, EarlyStoppingCallback
from commit_message_llm.evaluation import Evaluator
from commit_message_llm.inference import CommitMessageGenerator
from commit_message_llm.utils import setup_logging, get_gpu_memory_info

__all__ = [
    "load_config",
    "default_config_path",
    "DataProcessor",
    "DatasetSplit",
    "ModelSetup",
    "LoraConfig",
    "Trainer",
    "EarlyStoppingCallback",
    "Evaluator",
    "CommitMessageGenerator",
    "setup_logging",
    "get_gpu_memory_info",
]