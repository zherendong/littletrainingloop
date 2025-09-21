"""
Data loader for SlimPajama dataset turning SlimPajama dictionaries into text to train on.
"""


def extract(row: dict) -> str:
    """Extract text from a row."""
    return row["text"]
