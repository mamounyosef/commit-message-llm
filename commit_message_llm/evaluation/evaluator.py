"""Evaluator for commit message LLM model."""

import random
from typing import Any

import torch

from commit_message_llm.config.schema import ConfigSchema
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


class Evaluator:
    """
    Handles evaluation of the trained model including
    quantitative metrics and qualitative sampling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ConfigSchema,
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            model: Trained model.
            tokenizer: Tokenizer.
            config: Configuration object.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = config.model
        self.training_config = config.training

    def sample_eval(
        self,
        tokenized_test,
        n: int | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Sample and evaluate examples from the test set.

        Args:
            tokenized_test: Tokenized test dataset.
            n: Number of samples to generate.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            List of evaluation results with prompts, predictions, and ground truths.
        """
        if n is None:
            n = self.training_config.eval_samples
        if max_new_tokens is None:
            max_new_tokens = self.model_config.max_new_tokens

        logger.info(f"Sampling {n} examples for qualitative evaluation...")

        rnd = random.Random()
        idxs = rnd.sample(range(len(tokenized_test)), k=min(n, len(tokenized_test)))

        self.model.eval()
        results = []

        with torch.no_grad():
            for idx in idxs:
                ex = tokenized_test[idx]
                input_ids = ex["input_ids"]
                labels = ex["labels"]

                # Find prompt length (leading -100 labels)
                prompt_len = 0
                for l in labels:
                    if l == -100:
                        prompt_len += 1
                    else:
                        break

                if prompt_len >= len(labels):
                    logger.debug(f"Index {idx}: skipped (no target tokens)")
                    continue

                prompt_ids = input_ids[:prompt_len]
                target_ids = [l for l in labels if l != -100]

                # Generate from prompt
                inputs = {
                    "input_ids": torch.tensor([prompt_ids], device=self.model.device),
                    "attention_mask": torch.tensor([[1] * len(prompt_ids)], device=self.model.device),
                }

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.model_config.do_sample,
                    num_beams=1,
                    repetition_penalty=self.model_config.repetition_penalty,
                    no_repeat_ngram_size=self.model_config.no_repeat_ngram_size,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                gen_ids = outputs[0].tolist()
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

                pred_msg = gen_text[len(prompt_text):].strip()
                gt_msg = self.tokenizer.decode(target_ids, skip_special_tokens=True).strip()

                results.append({
                    "index": idx,
                    "prompt": prompt_text,
                    "prediction": pred_msg,
                    "ground_truth": gt_msg,
                })

                logger.info("=" * 80)
                logger.info(f"Sample {len(results)} - Index: {idx}")
                logger.info(f"Prompt (truncated):\n{prompt_text[:800]}{'...' if len(prompt_text) > 800 else ''}")
                logger.info(f"\nGround truth:\n{gt_msg}")
                logger.info(f"\nModel output:\n{pred_msg}")

        return results

    def print_summary(self, results: list[dict[str, Any]]) -> None:
        """
        Print a summary of evaluation results.

        Args:
            results: List of evaluation results.
        """
        logger.info("=" * 80)
        logger.info(f"Generated {len(results)} samples")
        logger.info("=" * 80)


__all__ = [
    "Evaluator",
]
