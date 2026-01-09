import argparse
import json
import random
import urllib.request
import os
from typing import List, Dict, Any

WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

def get_ordinal(n: int) -> str:
    """Returns ordinal string (1st, 2nd, 3rd) for a given integer."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def create_count_task(word: str, force_char: str = None) -> Dict[str, str]:
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
        "metadata": {"word": word_lower, "char": char}
    }

def create_index_task(word: str, force_idx: int = None) -> Dict[str, str]:
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
        "metadata": {"word": word_lower, "index": idx}
    }

def create_reverse_task(word: str) -> Dict[str, str]:
    """Creates a 'Reverse the word' task."""
    word_lower = word.lower()
    return {
        "input": f"Reverse the word '{word_lower}'.",
        "target": word_lower[::-1],
        "task_type": "reverse",
        "metadata": {"word": word_lower}
    }

def fetch_words() -> List[str]:
    print(f"Downloading word list from {WORD_LIST_URL}...")
    try:
        with urllib.request.urlopen(WORD_LIST_URL) as response:
            words = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Error downloading words: {e}")
        return []
    # Filter: 4-10 chars, alphabetic only
    return [w.strip().lower() for w in words if w.isalpha() and 4 <= len(w) <= 10]

def generate_dataset(output_path: str, num_samples: int = 1000, seed: int = 42):
    random.seed(seed)
    words = fetch_words()
    if not words:
        return

    print(f"Loaded {len(words)} candidate words.")
    data = []

    # Stratified Sampling: exact quotas
    count_samples = num_samples // 3
    index_samples = num_samples // 3
    reverse_samples = num_samples - count_samples - index_samples  # Catch remainder

    # Create task list
    tasks = (['count'] * count_samples) + \
            (['index'] * index_samples) + \
            (['reverse'] * reverse_samples)
    random.shuffle(tasks)
    
    for task_type in tasks:
        word = random.choice(words)
        
        if task_type == "count":
            entry = create_count_task(word)
        elif task_type == "index":
            entry = create_index_task(word)
        elif task_type == "reverse":
            entry = create_reverse_task(word)
            
        data.append(entry)

    # Safe directory creation
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Successfully generated {len(data)} items to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="spelling_benchmark/spelling_bee.jsonl")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output, args.samples, args.seed)
