"""Unit tests for evaluation metrics."""

import math

import pytest

from commit_message_llm.evaluation.metrics import perplexity


class TestPerplexity:
    """Tests for perplexity calculation."""

    def test_perplexity_from_loss(self) -> None:
        """Test perplexity calculation from loss."""
        assert math.isclose(perplexity(1.0), math.e, rel_tol=1e-9)
        assert math.isclose(perplexity(0.0), 1.0, rel_tol=1e-9)

    def test_perplexity_higher_loss(self) -> None:
        """Test perplexity increases with loss."""
        ppl1 = perplexity(1.0)
        ppl2 = perplexity(2.0)
        assert ppl2 > ppl1

    def test_perplexity_negative_loss(self) -> None:
        """Test perplexity for negative loss (unlikely but possible)."""
        ppl = perplexity(-0.5)
        assert ppl < 1.0
        assert ppl > 0

    def test_perplexity_very_high_loss(self) -> None:
        """Test perplexity for very high loss."""
        # At some point, it should overflow to infinity
        ppl = perplexity(1000)
        assert ppl == float("inf")
