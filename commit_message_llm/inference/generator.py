"""Commit message generator for inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from commit_message_llm.config.schema import ConfigSchema, ModelConfig
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result of commit message generation."""

    diff: str
    """The input diff text."""

    commit_message: str
    """The generated commit message."""

    prompt: str
    """The full prompt sent to the model."""

    generation_time: float
    """Time taken for generation in seconds."""

    @property
    def truncated_diff(self) -> str:
        """Get truncated version of diff for display."""
        return self.diff[:800] + ("..." if len(self.diff) > 800 else "")


class CommitMessageGenerator:
    """
    Generates commit messages from Git diffs using a trained model.

    This class handles loading a trained model (with LoRA adapters if specified)
    and generating commit messages from raw diff text.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: ConfigSchema | None = None,
        device: str | None = None,
        load_adapter: bool = True,
    ) -> None:
        """
        Initialize the CommitMessageGenerator.

        Args:
            model_path: Path to the trained model or model ID.
            config: Configuration object. If None, uses default config.
            device: Device to run on ("cuda", "cpu", or None for auto).
            load_adapter: Whether to load LoRA adapters from the model path.
        """
        self.model_path = Path(model_path)
        self.config = config or ConfigSchema()
        self.model_config = self.config.model
        self.load_adapter = load_adapter

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Any = None
        self._tokenizer: Any = None

        logger.info(f"Initializing generator with model: {model_path}")
        logger.info(f"Device: {self.device}")

    def load(self) -> None:
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=self.model_config.use_fast_tokenizer,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        if self.load_adapter:
            # Load as PEFT model with adapters
            from peft import PeftModel

            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )

            logger.info(f"Loading LoRA adapters from {self.model_path}...")
            self._model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load as regular model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )

        if self.device == "cpu":
            self._model = self._model.to(self.device)

        self._model.eval()

        logger.info("Model loaded successfully")

    @property
    def model(self) -> Any:
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def generate(
        self,
        diff: str,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a commit message from a diff.

        Args:
            diff: The Git diff text.
            max_new_tokens: Maximum new tokens to generate.
            do_sample: Whether to use sampling (vs greedy decoding).
            **kwargs: Additional generation arguments.

        Returns:
            GenerationResult with the generated commit message.
        """
        import time

        if max_new_tokens is None:
            max_new_tokens = self.model_config.max_new_tokens
        if do_sample is None:
            do_sample = self.model_config.do_sample

        # Build prompt
        separator = self.config.data.separator
        prompt = diff + separator

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length,
        ).to(self.model.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                repetition_penalty=self.model_config.repetition_penalty,
                no_repeat_ngram_size=self.model_config.no_repeat_ngram_size,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generation_time = time.time() - start_time

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        commit_message = generated_text[len(prompt):].strip()

        return GenerationResult(
            diff=diff,
            commit_message=commit_message,
            prompt=prompt,
            generation_time=generation_time,
        )

    def generate_batch(
        self,
        diffs: list[str],
        **kwargs,
    ) -> list[GenerationResult]:
        """
        Generate commit messages for multiple diffs.

        Args:
            diffs: List of Git diff texts.
            **kwargs: Additional generation arguments.

        Returns:
            List of GenerationResult objects.
        """
        results = []

        for i, diff in enumerate(diffs):
            logger.debug(f"Generating commit message {i + 1}/{len(diffs)}...")
            result = self.generate(diff, **kwargs)
            results.append(result)

        return results

    def generate_from_git_diff(self, repo_path: str | Path | None = None, **kwargs) -> list[GenerationResult]:
        """
        Generate commit messages from the current git diff.

        Args:
            repo_path: Path to the git repository. If None, uses current directory.
            **kwargs: Additional generation arguments.

        Returns:
            List of GenerationResult objects (one per file changed).
        """
        import subprocess

        try:
            cmd = ["git", "diff"]
            if repo_path:
                cmd.extend(["-C", str(repo_path)])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            diff_text = result.stdout

            if not diff_text:
                logger.warning("No git diff found")
                return []

            # Split by file and generate for each
            # For simplicity, we'll treat the whole diff as one item
            return [self.generate(diff_text, **kwargs)]

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get git diff: {e}")
            return []
        except FileNotFoundError:
            logger.error("Git not found. Please ensure git is installed.")
            return []


__all__ = [
    "CommitMessageGenerator",
    "GenerationResult",
]
