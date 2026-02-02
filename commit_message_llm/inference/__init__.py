"""Inference module for commit message LLM."""

from commit_message_llm.inference.generator import CommitMessageGenerator, GenerationResult

__all__ = [
    "CommitMessageGenerator",
    "GenerationResult",
]
