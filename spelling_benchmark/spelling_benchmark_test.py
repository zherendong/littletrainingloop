import pytest
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import create_count_task, create_index_task, create_reverse_task, get_ordinal


class TestGetOrdinal:
    def test_ordinal_1st(self):
        assert get_ordinal(1) == "1st"

    def test_ordinal_2nd(self):
        assert get_ordinal(2) == "2nd"

    def test_ordinal_3rd(self):
        assert get_ordinal(3) == "3rd"

    def test_ordinal_4th(self):
        assert get_ordinal(4) == "4th"

    def test_ordinal_11th(self):
        assert get_ordinal(11) == "11th"

    def test_ordinal_21st(self):
        assert get_ordinal(21) == "21st"


class TestCountTask:
    def test_simple_count(self):
        task = create_count_task("apple", force_char="p")
        assert task["target"] == "2"
        assert "apple" in task["input"]

    def test_zero_count(self):
        task = create_count_task("apple", force_char="z")
        assert task["target"] == "0"

    def test_case_insensitivity(self):
        """Word should be lowercased in output."""
        task = create_count_task("Strawberry", force_char="r")
        assert task["target"] == "3"
        assert "strawberry" in task["input"]


class TestIndexTask:
    def test_first_letter(self):
        task = create_index_task("Python", force_idx=0)
        assert "1st" in task["input"]
        assert task["target"] == "p"  # lowercase

    def test_third_letter(self):
        task = create_index_task("Python", force_idx=2)
        assert "3rd" in task["input"]
        assert task["target"] == "t"  # lowercase


class TestReverseTask:
    def test_reverse_stressed(self):
        task = create_reverse_task("stressed")
        assert task["target"] == "desserts"

    def test_reverse_palindrome(self):
        task = create_reverse_task("racecar")
        assert task["target"] == "racecar"

    def test_reverse_case_normalization(self):
        task = create_reverse_task("Hello")
        assert task["target"] == "olleh"  # lowercase
