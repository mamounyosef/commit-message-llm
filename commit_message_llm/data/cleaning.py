"""Data cleaning utilities for commit message dataset."""

import re
from collections.abc import Callable, Mapping
from typing import Any

# Default patterns for detecting bad commit messages
BAD_EXACT = {
    "update", "updated", "fix", "fixed", "wip", ".", "..", "...", "temp", "test",
}

# Pattern for reference-only commits (e.g., "fix #123")
REF_ONLY_RE = re.compile(
    r"^\s*(fixe[sd]?|close[sd]?|resolve[sd]?|ref[s]?)\s*#?\w+.*$",
    re.IGNORECASE
)

# Pattern for placeholder tokens
PLACEHOLDER_RE = re.compile(r"<HASH>|<URL>|#<I>|\(#<I>\)")

# Patterns for detecting diff text
DIFF_PATTERNS = [
    r"^diff --git ",
    r"^@@ ",
    r"^\+\+\+ ",
    r"^--- ",
    r"^index ",
    r"^[+-]",
]


def normalize_newlines(s: str) -> str:
    """
    Normalize newlines in a string to Unix-style (\n).

    Args:
        s: Input string.

    Returns:
        String with normalized newlines and trimmed whitespace.
    """
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def looks_like_diff(text: str | None) -> bool:
    """
    Check if text looks like a Git diff.

    Args:
        text: Text to check.

    Returns:
        True if text appears to be a diff.
    """
    if not text:
        return False
    return any(re.search(p, text, flags=re.MULTILINE) for p in DIFF_PATTERNS)


def looks_like_message(text: str | None) -> bool:
    """
    Check if text looks like a commit message.

    Args:
        text: Text to check.

    Returns:
        True if text appears to be a commit message.
    """
    if not text:
        return False
    if looks_like_diff(text):
        return False
    return len(text.split()) <= 200


def is_bad_message(
    msg: str,
    min_chars: int = 3,
    min_words: int = 3,
    bad_exact: set[str] | None = None,
) -> bool:
    """
    Check if a commit message is bad (low quality).

    Args:
        msg: Commit message to check.
        min_chars: Minimum character count.
        min_words: Minimum word count.
        bad_exact: Set of exact bad messages to filter.

    Returns:
        True if the message is considered bad.
    """
    bad_exact = bad_exact or BAD_EXACT
    m = msg.strip()

    if not m:
        return True
    if m.lower() in bad_exact:
        return True
    if len(m) < min_chars:
        return True
    if len(m.split()) < min_words:
        return True
    if REF_ONLY_RE.match(m):
        return True
    if PLACEHOLDER_RE.search(m):
        return True

    return False


def keep_example(
    ex: Mapping[str, Any],
    diff_col: str = "diff",
    msg_col: str = "message",
    min_diff_chars: int = 50,
    max_diff_chars: int = 8000,
    min_msg_chars: int = 3,
    min_msg_words: int = 3,
    bad_exact: set[str] | None = None,
) -> bool:
    """
    Determine if an example should be kept in the dataset.

    Args:
        ex: Example dict with diff and message columns.
        diff_col: Name of the diff column.
        msg_col: Name of the message column.
        min_diff_chars: Minimum diff character count.
        max_diff_chars: Maximum diff character count.
        min_msg_chars: Minimum message character count.
        min_msg_words: Minimum message word count.
        bad_exact: Set of exact bad messages to filter.

    Returns:
        True if the example should be kept.
    """
    bad_exact = bad_exact or BAD_EXACT

    d = normalize_newlines(ex.get(diff_col, "") or "")
    m = normalize_newlines(ex.get(msg_col, "") or "")

    if len(d) < min_diff_chars:
        return False
    if len(d) > max_diff_chars:
        return False
    if is_bad_message(m, min_msg_chars, min_msg_words, bad_exact):
        return False

    return True


def infer_columns(example: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """
    Infer which columns are diff and message columns.

    Args:
        example: A single example from the dataset.

    Returns:
        Tuple of (diff_column, message_column) names.
    """
    scores: dict[str, tuple[int, int]] = {}

    for k, v in example.items():
        if not isinstance(v, str):
            continue

        diff_score = 0
        msg_score = 0

        if looks_like_diff(v):
            diff_score += 2
        if looks_like_message(v):
            msg_score += 1
        if len(v) > 500:
            diff_score += 1
        if len(v) < 300:
            msg_score += 1

        scores[k] = (diff_score, msg_score)

    if not scores:
        return None, None

    diff_col = max(scores, key=lambda k: scores[k][0])
    msg_col = max(scores, key=lambda k: scores[k][1])

    if diff_col == msg_col:
        msg_candidates = sorted(scores.keys(), key=lambda k: scores[k][1], reverse=True)
        if len(msg_candidates) > 1:
            msg_col = msg_candidates[1]

    return diff_col, msg_col


def preprocess_example(
    ex: Mapping[str, Any],
    diff_col: str = "diff",
    msg_col: str = "message",
) -> dict[str, Any]:
    """
    Preprocess a single example by normalizing text columns.

    Args:
        ex: Example dict with diff and message columns.
        diff_col: Name of the diff column.
        msg_col: Name of the message column.

    Returns:
        Preprocessed example dict.
    """
    result = dict(ex)
    result[diff_col] = normalize_newlines(ex.get(diff_col, "") or "")
    result[msg_col] = normalize_newlines(ex.get(msg_col, "") or "")
    return result


__all__ = [
    "normalize_newlines",
    "looks_like_diff",
    "looks_like_message",
    "is_bad_message",
    "keep_example",
    "infer_columns",
    "preprocess_example",
    "BAD_EXACT",
    "REF_ONLY_RE",
    "PLACEHOLDER_RE",
    "DIFF_PATTERNS",
]
