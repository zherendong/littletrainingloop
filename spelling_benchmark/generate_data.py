import argparse
import json
import random
import urllib.request
import os
from typing import List, Dict, Any

import language_model_dataloader

# Full English word list (~370k words)
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
# Common English words (~3000 most frequent)
COMMON_WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"


def get_ordinal(n: int) -> str:
    """Returns ordinal string (1st, 2nd, 3rd) for a given integer."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def create_count_task(word: str, num_tokens: int, force_char: str = None) -> Dict[str, Any]:
    """Creates a 'Count the letter' task."""
    word_lower = word.lower()
    
    # 80% chance to pick a char in the word, 20% random char (unless forced)
    if force_char:
        char = force_char
    else:
        if random.random() < 0.8:
            char = random.choice(list(set(word_lower))) 
        else:
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            
    count = word_lower.count(char)
    return {
        "input": f"How many times does the letter '{char}' appear in the word '{word_lower}'?",
        "target": str(count),
        "task_type": "count",
        "metadata": {
            "word": word_lower,
            "char": char,
            "num_tokens": num_tokens,
            "is_single_token": num_tokens == 1,
        }
    }


def create_index_task(word: str, num_tokens: int, force_idx: int = None) -> Dict[str, Any]:
    """Creates a 'What is the N-th letter' task."""
    word_lower = word.lower()
    
    if force_idx is not None:
        idx = force_idx
    else:
        idx = random.randint(0, len(word_lower) - 1)
        
    ordinal = get_ordinal(idx + 1)
    return {
        "input": f"What is the {ordinal} letter of the word '{word_lower}'?",
        "target": word_lower[idx],
        "task_type": "index",
        "metadata": {
            "word": word_lower,
            "index": idx,
            "num_tokens": num_tokens,
            "is_single_token": num_tokens == 1,
        }
    }


def create_reverse_task(word: str, num_tokens: int) -> Dict[str, Any]:
    """Creates a 'Reverse the word' task."""
    word_lower = word.lower()
    return {
        "input": f"Reverse the word '{word_lower}'.",
        "target": word_lower[::-1],
        "task_type": "reverse",
        "metadata": {
            "word": word_lower,
            "num_tokens": num_tokens,
            "is_single_token": num_tokens == 1,
        }
    }


def fetch_word_list(url: str, name: str) -> List[str]:
    """Downloads and filters a word list (4-10 chars, alphabetic only)."""
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
    
    Returns:
        Tuple of (common_words, full_words) where full_words excludes common_words.
    """
    common_words = fetch_word_list(COMMON_WORDS_URL, "common words")
    full_words = fetch_word_list(WORD_LIST_URL, "full word list")
    
    # Deduplicate: remove common words from full list
    common_set = set(common_words)
    full_words_deduped = [w for w in full_words if w not in common_set]
    
    print(f"Loaded {len(common_words)} common words, {len(full_words_deduped)} other words (deduped)")
    return common_words, full_words_deduped


def generate_dataset(output_path: str, num_samples: int = 1000, seed: int = 42):
    random.seed(seed)
    
    # Load tokenizer from our repo
    tokenizer = language_model_dataloader.default_tokenizer()
    
    common_words, other_words = fetch_words()
    
    if not common_words or not other_words:
        print("Error: Could not load word lists")
        return

    data = []

    # Stratified Sampling: exact quotas for task types
    count_samples = num_samples // 3
    index_samples = num_samples // 3
    reverse_samples = num_samples - count_samples - index_samples  # Catch remainder

    # Create task list
    tasks = (['count'] * count_samples) + \
            (['index'] * index_samples) + \
            (['reverse'] * reverse_samples)
    random.shuffle(tasks)
    
    single_token_count = 0
    multi_token_count = 0
    
    for task_type in tasks:
        # 50% common words, 50% other words
        if random.random() < 0.5:
            word = random.choice(common_words)
        else:
            word = random.choice(other_words)
        
        # Calculate token count using our tokenizer
        num_tokens = len(tokenizer.encode(word))
        
        if num_tokens == 1:
            single_token_count += 1
        else:
            multi_token_count += 1
        
        if task_type == "count":
            entry = create_count_task(word, num_tokens)
        elif task_type == "index":
            entry = create_index_task(word, num_tokens)
        elif task_type == "reverse":
            entry = create_reverse_task(word, num_tokens)
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="spelling_benchmark/spelling_bee.jsonl")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output, args.samples, args.seed)
