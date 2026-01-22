import pytest
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import create_count_task, create_index_task, create_reverse_task, get_ordinal, number_to_text


class TestGetOrdinal:
    def test_ordinal_first(self):
        assert get_ordinal(1) == "first"

    def test_ordinal_second(self):
        assert get_ordinal(2) == "second"

    def test_ordinal_third(self):
        assert get_ordinal(3) == "third"

    def test_ordinal_fourth(self):
        assert get_ordinal(4) == "fourth"

    def test_ordinal_eleventh(self):
        assert get_ordinal(11) == "eleventh"

    def test_ordinal_twentieth(self):
        assert get_ordinal(20) == "twentieth"

    def test_ordinal_fallback(self):
        # Numbers > 20 fall back to numeric format
        assert get_ordinal(21) == "21th"


class TestNumberToText:
    def test_zero(self):
        assert number_to_text(0) == "zero"

    def test_one(self):
        assert number_to_text(1) == "one"

    def test_five(self):
        assert number_to_text(5) == "five"

    def test_ten(self):
        assert number_to_text(10) == "ten"

    def test_fallback(self):
        # Numbers > 10 fall back to numeric string
        assert number_to_text(11) == "11"


class TestCountTask:
    def test_simple_count(self):
        task = create_count_task("apple", force_char="p")
        assert task["target"] == "2"
        # Format: "The number of times the letter X occurs in word is " (matches training)
        assert task["input"] == "The number of times the letter P occurs in apple is "

    def test_zero_count(self):
        task = create_count_task("apple", force_char="z")
        assert task["target"] == "0"

    def test_uppercase_letter(self):
        """Letter should be uppercase by default."""
        task = create_count_task("Strawberry", force_char="r")
        assert task["target"] == "3"
        assert "strawberry" in task["input"]
        assert " R " in task["input"]  # Uppercase letter

    def test_lowercase_letter(self):
        """Letter should be lowercase when use_lowercase=True."""
        task = create_count_task("Strawberry", force_char="r", use_lowercase=True)
        assert task["target"] == "3"
        assert "strawberry" in task["input"]
        assert " r " in task["input"]  # Lowercase letter

    def test_metadata_has_token_info(self):
        task = create_count_task("apple", force_char="a")
        assert "is_single_token" in task["metadata"]
        assert "num_tokens" in task["metadata"]


class TestIndexTask:
    def test_first_letter(self):
        task = create_index_task("Python", force_idx=0)
        assert "first" in task["input"]
        assert task["input"].startswith("Q:")
        assert task["target"] == "p"  # lowercase

    def test_third_letter(self):
        task = create_index_task("Python", force_idx=2)
        assert "third" in task["input"]
        assert task["target"] == "t"  # lowercase


class TestReverseTask:
    def test_reverse_stressed(self):
        task = create_reverse_task("stressed")
        assert task["target"] == "d-e-s-s-e-r-t-s"
        # Format: "s-t-r-e-s-s-e-d reversed is"
        assert task["input"] == "s-t-r-e-s-s-e-d reversed is"

    def test_reverse_palindrome_returns_none(self):
        """Palindromes should return None to avoid trivial cases."""
        task = create_reverse_task("racecar")
        assert task is None

    def test_reverse_case_normalization(self):
        task = create_reverse_task("Hello")
        assert task["target"] == "o-l-l-e-h"  # lowercase, hyphenated
        assert task["input"] == "h-e-l-l-o reversed is"
