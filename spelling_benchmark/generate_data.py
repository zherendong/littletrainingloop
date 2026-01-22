import json
import random
import urllib.request
import os
from pathlib import Path
from typing import List, Dict, Any

import language_model_dataloader

# Local asset paths (preferred)
ASSETS_DIR = Path(__file__).parent / "assets"
COMMON_WORDS_FILE = ASSETS_DIR / "common_words.txt"
FULL_WORDS_FILE = ASSETS_DIR / "full_words.txt"

# Fallback URLs if local files don't exist
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
COMMON_WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"

# Module-level tokenizer (lazy-loaded)
_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = language_model_dataloader.default_tokenizer()
    return _tokenizer


def get_ordinal(n: int) -> str:
    """Returns ordinal string (first, second, third) for a given integer."""
    ordinals = [
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"
    ]
    if 1 <= n <= 20:
        return ordinals[n - 1]
    # Fallback for numbers > 20 (shouldn't happen with max word length 10)
    return f"{n}th"


def number_to_text(n: int) -> str:
    """Returns text representation of a number (zero, one, two, etc.)."""
    numbers = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten"
    ]
    if 0 <= n <= 10:
        return numbers[n]
    # Fallback for numbers > 10 (unlikely for letter counts)
    return str(n)


def _get_token_metadata(word: str) -> Dict[str, Any]:
    """Returns tokenization metadata for a word."""
    tokenizer = _get_tokenizer()
    num_tokens = len(tokenizer.encode(word))
    return {
        "num_tokens": num_tokens,
        "is_single_token": num_tokens == 1,
    }


def create_count_task(word: str, force_char: str = None, use_lowercase: bool = False) -> Dict[str, Any]:
    """Creates a 'Count the letter' task.

    Format: "The number of times the letter X occurs in word is "
    (matches training format from strawberry_dataloader.py)
    """
    word_lower = word.lower()

    # 80% chance to pick a char in the word, 20% random char (unless forced)
    if force_char:
        char = force_char.lower() if use_lowercase else force_char.upper()
    else:
        if random.random() < 0.8:
            char = random.choice(sorted(set(word_lower)))
        else:
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
        if not use_lowercase:
            char = char.upper()

    count = word_lower.count(char.lower())
    return {
        "input": f"The number of times the letter {char} occurs in {word_lower} is ",
        "target": str(count),
        "task_type": "count",
        "metadata": {
            "word": word_lower,
            "char": char,
            **_get_token_metadata(word_lower),
        }
    }


def create_index_task(word: str, force_idx: int = None) -> Dict[str, Any]:
    """Creates a 'What is the N-th letter' task."""
    word_lower = word.lower()

    if force_idx is not None:
        idx = force_idx
    else:
        idx = random.randint(0, len(word_lower) - 1)

    ordinal = get_ordinal(idx + 1)
    return {
        "input": f"Q: What is the {ordinal} letter of the word '{word_lower}'?",
        "target": word_lower[idx],
        "task_type": "index",
        "metadata": {
            "word": word_lower,
            "index": idx,
            **_get_token_metadata(word_lower),
        }
    }


def is_palindrome(word: str) -> bool:
    """Check if a word is a palindrome."""
    return word == word[::-1]


def create_reverse_task(word: str) -> Dict[str, Any] | None:
    """Creates a 'Spell the word backwards' task.

    Format: "cat reversed is "
    Target: "tac"

    Returns None if word is a palindrome (to avoid trivial cases).
    """
    word_lower = word.lower()
    if is_palindrome(word_lower):
        return None
    return {
        "input": f"{word_lower} reversed is ",
        "target": word_lower[::-1],
        "task_type": "reverse",
        "metadata": {
            "word": word_lower,
            **_get_token_metadata(word_lower),
        }
    }


def load_word_list(local_path: Path, url: str, name: str) -> List[str]:
    """Loads word list from local file or falls back to URL.
    
    Filters to 4-10 char alphabetic words.
    """
    if local_path.exists():
        print(f"Loading {name} from {local_path}...")
        words = local_path.read_text().splitlines()
    else:
        print(f"Downloading {name} from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                words = response.read().decode('utf-8').splitlines()
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            return []
    # Filter: 4-10 chars, alphabetic only
    return [w.strip().lower() for w in words if w.isalpha() and 4 <= len(w) <= 10]


def fetch_words() -> tuple[List[str], List[str]]:
    """Fetches both common words and full word list.
    
    Loads from local assets if available, otherwise downloads from URLs.
    
    Returns:
        Tuple of (common_words, other_words) where other_words excludes common_words.
    """
    common_words = load_word_list(COMMON_WORDS_FILE, COMMON_WORDS_URL, "common words")
    full_words = load_word_list(FULL_WORDS_FILE, WORD_LIST_URL, "full word list")
    
    # Deduplicate: remove common words from full list
    common_set = set(common_words)
    other_words = [w for w in full_words if w not in common_set]
    
    print(f"Loaded {len(common_words)} common words, {len(other_words)} other words (deduped)")
    return common_words, other_words


def generate_dataset(
    output_path: str,
    num_samples: int = 1000,
    seed: int = 42,
    use_lowercase: bool = False,
    reverse_samples: int | None = None,
):
    """Generate spelling benchmark dataset.

    Args:
        output_path: Path to write JSONL output.
        num_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.
        use_lowercase: Use lowercase letters in count task.
        reverse_samples: Fixed number of reverse samples. If None, split evenly (1/3).
            Remaining samples are split evenly between count and index.
    """
    random.seed(seed)

    common_words, other_words = fetch_words()

    if not common_words or not other_words:
        print("Error: Could not load word lists")
        return

    data = []

    # Stratified Sampling: exact quotas for task types
    if reverse_samples is None:
        # Default: split evenly into thirds
        count_samples = num_samples // 3
        index_samples = num_samples // 3
        reverse_samples = num_samples - count_samples - index_samples
    else:
        # Custom: fixed reverse samples, rest split evenly between count/index
        remaining = num_samples - reverse_samples
        count_samples = remaining // 2
        index_samples = remaining - count_samples

    print(f"Task distribution: count={count_samples}, index={index_samples}, reverse={reverse_samples}")

    # Create task list
    tasks = (['count'] * count_samples) + \
            (['index'] * index_samples) + \
            (['reverse'] * reverse_samples)
    random.shuffle(tasks)
    
    single_token_count = 0
    multi_token_count = 0
    
    for task_type in tasks:
        entry = None
        while entry is None:
            # 50% common words, 50% other words
            if random.random() < 0.5:
                word = random.choice(common_words)
            else:
                word = random.choice(other_words)

            if task_type == "count":
                entry = create_count_task(word, use_lowercase=use_lowercase)
            elif task_type == "index":
                entry = create_index_task(word)
            elif task_type == "reverse":
                # Returns None for palindromes, retry with different word
                entry = create_reverse_task(word)

        # Count for statistics
        if entry["metadata"]["is_single_token"]:
            single_token_count += 1
        else:
            multi_token_count += 1

        data.append(entry)

    # Safe directory creation
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Successfully generated {len(data)} items to {output_path}")
    print(f"  Single-token words: {single_token_count} ({100*single_token_count/len(data):.1f}%)")
    print(f"  Multi-token words: {multi_token_count} ({100*multi_token_count/len(data):.1f}%)")

