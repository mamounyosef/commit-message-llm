"""Unit tests for data cleaning utilities."""

import pytest

from commit_message_llm.data.cleaning import (
    normalize_newlines,
    looks_like_diff,
    looks_like_message,
    is_bad_message,
    keep_example,
    infer_columns,
    BAD_EXACT,
)


class TestNormalizeNewlines:
    """Tests for normalize_newlines function."""

    def test_windows_newlines(self) -> None:
        """Test Windows-style newlines are converted."""
        assert normalize_newlines("line1\r\nline2\r\n") == "line1\nline2"

    def test_old_mac_newlines(self) -> None:
        """Test old Mac-style newlines are converted."""
        assert normalize_newlines("line1\rline2\r") == "line1\nline2"

    def test_mixed_newlines(self) -> None:
        """Test mixed newline styles are normalized."""
        assert normalize_newlines("line1\r\nline2\rline3\n") == "line1\nline2\nline3"

    def test_whitespace_trim(self) -> None:
        """Test leading/trailing whitespace is removed."""
        assert normalize_newlines("  \n  text  \n  ") == "text"


class TestLooksLikeDiff:
    """Tests for looks_like_diff function."""

    def test_git_diff_header(self) -> None:
        """Test git diff header pattern."""
        assert looks_like_diff("diff --git a/file.py b/file.py")

    def test_hunk_header(self) -> None:
        """Test hunk header pattern."""
        assert looks_like_diff("@@ -1,3 +1,4 @@")

    def test_file_markers(self) -> None:
        """Test file addition/deletion markers."""
        assert looks_like_diff("+++ a/file.py")
        assert looks_like_diff("--- a/file.py")

    def test_index_line(self) -> None:
        """Test index line pattern."""
        assert looks_like_diff("index abc123..def456 123456")

    def test_add_remove_lines(self) -> None:
        """Test added/removed line patterns."""
        assert looks_like_diff("+ new line")
        assert looks_like_diff("- old line")

    def test_empty_string(self) -> None:
        """Test empty string returns False."""
        assert not looks_like_diff("")

    def test_none_returns_false(self) -> None:
        """Test None returns False."""
        assert not looks_like_diff(None)

    def test_regular_text(self) -> None:
        """Test regular text doesn't match."""
        assert not looks_like_diff("This is just a regular message")


class TestLooksLikeMessage:
    """Tests for looks_like_message function."""

    def test_short_text(self) -> None:
        """Test short text is considered a message."""
        assert looks_like_message("Fix bug in parser")

    def test_long_text_returns_false(self) -> None:
        """Test very long text is not considered a message."""
        long_text = "word " * 250  # More than 200 words
        assert not looks_like_message(long_text)

    def test_diff_returns_false(self) -> None:
        """Test diff returns False even if short."""
        assert not looks_like_message("diff --git a/file.py")

    def test_empty_string(self) -> None:
        """Test empty string returns False."""
        assert not looks_like_message("")

    def test_none_returns_false(self) -> None:
        """Test None returns False."""
        assert not looks_like_message(None)


class TestIsBadMessage:
    """Tests for is_bad_message function."""

    def test_empty_message(self) -> None:
        """Test empty message is bad."""
        assert is_bad_message("")

    def test_whitespace_only(self) -> None:
        """Test whitespace-only message is bad."""
        assert is_bad_message("   ")

    def test_exact_bad_messages(self) -> None:
        """Test exact bad messages are filtered."""
        assert is_bad_message("fix")
        assert is_bad_message("update")
        assert is_bad_message("wip")
        assert is_bad_message(".")

    def test_case_sensitivity(self) -> None:
        """Test lowercase comparison for exact matches."""
        assert is_bad_message("FIX")
        assert is_bad_message("UPDATE")

    def test_too_short(self) -> None:
        """Test messages below minimum character count are bad."""
        assert is_bad_message("ab", min_chars=3)

    def test_too_few_words(self) -> None:
        """Test messages with too few words are bad."""
        assert is_bad_message("fix bug", min_words=3)

    def test_reference_only(self) -> None:
        """Test reference-only commits are bad."""
        assert is_bad_message("fix #123")
        assert is_bad_message("closes #456")
        assert is_bad_message("resolves #789")

    def test_placeholder_tokens(self) -> None:
        """Test messages with placeholder tokens are bad."""
        assert is_bad_message("Fix <HASH> issue")
        assert is_bad_message("See <URL> for details")

    def test_valid_message(self) -> None:
        """Test valid message passes."""
        assert not is_bad_message("Fix bug in parser that caused crash")

    def test_custom_bad_messages(self) -> None:
        """Test custom bad message set."""
        bad_set = {"custom", "badmsg"}
        assert is_bad_message("custom", bad_exact=bad_set)
        assert not is_bad_message("valid message", bad_exact=bad_set)


class TestKeepExample:
    """Tests for keep_example function."""

    def test_valid_example(self) -> None:
        """Test valid example is kept."""
        ex = {
            "diff": "diff --git a/file.py b/file.py\n+ new line",
            "message": "Add new feature to file"
        }
        assert keep_example(ex)

    def test_diff_too_short(self) -> None:
        """Test example with diff too short is rejected."""
        ex = {
            "diff": "short",
            "message": "Add feature"
        }
        assert not keep_example(ex, min_diff_chars=50)

    def test_diff_too_long(self) -> None:
        """Test example with diff too long is rejected."""
        ex = {
            "diff": "x" * 10000,
            "message": "Add feature"
        }
        assert not keep_example(ex, max_diff_chars=8000)

    def test_bad_message(self) -> None:
        """Test example with bad message is rejected."""
        ex = {
            "diff": "diff --git a/file.py b/file.py\n+ new line",
            "message": "fix"
        }
        assert not keep_example(ex)

    def test_empty_columns(self) -> None:
        """Test example with empty columns is rejected."""
        ex = {
            "diff": "",
            "message": ""
        }
        assert not keep_example(ex)


class TestInferColumns:
    """Tests for infer_columns function."""

    def test_diff_and_message_columns(self) -> None:
        """Test inferring diff and message columns."""
        ex = {
            "diff": "diff --git a/file.py b/file.py\n+ new line",
            "message": "Add new feature",
            "other": "some other data"
        }
        diff_col, msg_col = infer_columns(ex)
        assert diff_col == "diff"
        assert msg_col == "message"

    def test_alternative_column_names(self) -> None:
        """Test inferring with alternative column names."""
        ex = {
            "patch": "diff --git a/file.py b/file.py\n+ new line",
            "commit_message": "Add new feature",
        }
        diff_col, msg_col = infer_columns(ex)
        assert diff_col == "patch"
        assert msg_col == "commit_message"

    def test_no_clear_diff_column(self) -> None:
        """Test when no clear diff column exists."""
        ex = {
            "col1": "some text",
            "col2": "some other text",
        }
        diff_col, msg_col = infer_columns(ex)
        # Should return something, even if not confident
        assert diff_col is not None
        assert msg_col is not None

    def test_empty_example(self) -> None:
        """Test empty example returns None for both."""
        diff_col, msg_col = infer_columns({})
        assert diff_col is None
        assert msg_col is None

    def test_non_string_values_ignored(self) -> None:
        """Test non-string values are ignored."""
        ex = {
            "text": "some text",
            "number": 123,
            "boolean": True,
        }
        diff_col, msg_col = infer_columns(ex)
        # text should be selected as both (only string column)
        assert diff_col == "text"
