"""
Sampling text windows from SlimPajama data files.

Provides:
- sample_random_windows: Sample random windows from random positions
- sample_consecutive_windows: Sample consecutive windows for testing adjacency
"""

import json
import glob
import random


def sample_random_windows(
    data_dir: str,
    num_windows: int,
    window_size: int,
    tokenizer,
    seed: int,
) -> list[str]:
    """
    Sample random windows from random positions in random files.

    Texts are sampled with weight proportional to their length, so longer texts
    contribute more windows. This gives a representative sample of all positions.

    Windows can be 1 to window_size tokens, allowing sampling from the beginning
    of texts where less context is available.

    Args:
        data_dir: Path to directory containing .jsonl files
        num_windows: Number of windows to sample
        window_size: Maximum size of each window in tokens
        tokenizer: Tokenizer to use for encoding/decoding
        seed: Random seed for reproducibility

    Returns:
        List of window text strings
    """
    random.seed(seed)

    files = glob.glob(f"{data_dir}/*.jsonl")
    if not files:
        raise ValueError(f"No .jsonl files found in {data_dir}")

    random.shuffle(files)
    windows = []

    for file in files:
        if len(windows) >= num_windows:
            break

        try:
            with open(file, "r") as f:
                texts = [json.loads(line)["text"] for line in f]
        except Exception:
            continue

        texts = [t for t in texts if t]
        if not texts:
            continue
        weights = [len(t) for t in texts]

        windows_needed = num_windows - len(windows)
        sampled_texts = random.choices(texts, weights=weights, k=windows_needed)

        for text in sampled_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) == 0:
                continue

            pos = random.randint(1, len(tokens))
            start = max(0, pos - window_size)
            window_text = tokenizer.decode(tokens[start:pos])
            windows.append(window_text)

    return windows


def sample_consecutive_windows(
    data_dir: str,
    num_texts: int,
    windows_per_text: int,
    window_size: int,
    tokenizer,
    seed: int,
) -> list[tuple[int, int, str]]:
    """
    Sample consecutive windows from diverse texts.

    Returns windows with their (text_idx, window_idx) for tracking.
    Texts are sampled with weight proportional to their length.

    Windows start from position 0 (empty context) and grow up to window_size tokens,
    allowing testing of positions at the beginning of texts.

    Args:
        data_dir: Path to directory containing .jsonl files
        num_texts: Number of texts to sample from
        windows_per_text: Number of consecutive windows per text
        window_size: Maximum size of each window in tokens
        tokenizer: Tokenizer for encoding/decoding
        seed: Random seed

    Returns:
        List of (text_idx, window_idx, window_text) tuples
    """
    random.seed(seed)

    files = glob.glob(f"{data_dir}/*.jsonl")
    if not files:
        raise ValueError(f"No .jsonl files found in {data_dir}")

    random.shuffle(files)
    all_windows = []
    text_idx = 0

    for file in files:
        if text_idx >= num_texts:
            break

        try:
            with open(file, "r") as f:
                file_texts = [json.loads(line)["text"] for line in f]
        except Exception:
            continue

        valid_texts = [t for t in file_texts if t and len(t) >= windows_per_text]
        if not valid_texts:
            continue
        weights = [len(t) for t in valid_texts]

        texts_needed = num_texts - text_idx
        sampled_texts = random.choices(valid_texts, weights=weights, k=texts_needed)

        for text in sampled_texts:
            if text_idx >= num_texts:
                break

            tokens = tokenizer.encode(text)
            if len(tokens) < windows_per_text:
                continue

            for pos in range(1, windows_per_text + 1):
                start = max(0, pos - window_size)
                window_text = tokenizer.decode(tokens[start:pos])
                all_windows.append((text_idx, pos - 1, window_text))

            text_idx += 1

    return all_windows
