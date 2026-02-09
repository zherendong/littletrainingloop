"""
TinyShakespeare dataloader for quick experiments.
"""

import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

import language_model_basics
import training_loop


def download_tiny_shakespeare(data_dir: str = "./data") -> str:
    """Download TinyShakespeare if needed, return the text."""
    path = Path(data_dir) / "shakespeare.txt"
    path.parent.mkdir(exist_ok=True)
    
    if not path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare to {path}...")
        urllib.request.urlretrieve(url, path)
    
    with open(path, "r") as f:
        return f.read()


class TinyShakespeareVocab:
    """Character-level vocabulary for TinyShakespeare."""
    
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
    
    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[t] for t in tokens)


# Global vocab instance (initialized lazily)
_vocab: TinyShakespeareVocab | None = None
_data: np.ndarray | None = None


def get_vocab_and_data() -> tuple[TinyShakespeareVocab, np.ndarray]:
    """Get vocab and encoded data, downloading/encoding if needed."""
    global _vocab, _data
    if _vocab is None:
        text = download_tiny_shakespeare()
        _vocab = TinyShakespeareVocab(text)
        _data = np.array(_vocab.encode(text), dtype=np.int32)
        print(f"TinyShakespeare: {len(_data):,} chars, vocab size {_vocab.vocab_size}")
    return _vocab, _data


class TinyShakespeareDataLoader(training_loop.DataProvider[language_model_basics.LMData]):
    """Simple dataloader for TinyShakespeare."""
    
    def __init__(
        self,
        config: language_model_basics.LanguageModelTrainingConfig,
        split: str = "train",
    ):
        self.config = config
        self.split = split
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.seed = config.seed
        
        vocab, data = get_vocab_and_data()
        self.vocab = vocab
        
        # Split data 90/10
        n = int(0.9 * len(data))
        if split == "train":
            self.data = data[:n]
        else:
            self.data = data[n:]
    
    def generate(self) -> Iterable[language_model_basics.LMData]:
        """Generate batches of data."""
        rng = np.random.default_rng(self.seed)
        
        while True:
            # Random starting positions
            max_start = len(self.data) - self.sequence_length - 1
            starts = rng.integers(0, max_start, size=self.batch_size)
            
            inputs = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
            targets = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
            loss_mask = np.ones((self.batch_size, self.sequence_length), dtype=np.float32)
            
            for i, start in enumerate(starts):
                inputs[i] = self.data[start:start + self.sequence_length]
                targets[i] = self.data[start + 1:start + self.sequence_length + 1]
            
            yield language_model_basics.LMData(
                inputs=inputs,
                targets=targets,
                loss_mask=loss_mask,
            )
    
    def get_name(self) -> str:
        return f"TinyShakespeare ({self.split})"


def create_tiny_shakespeare_dataloader(
    config: language_model_basics.LanguageModelTrainingConfig,
    split: str = "train",
) -> TinyShakespeareDataLoader:
    """Create a TinyShakespeare dataloader."""
    return TinyShakespeareDataLoader(config, split)


def get_vocab_size() -> int:
    """Get the vocabulary size for TinyShakespeare."""
    vocab, _ = get_vocab_and_data()
    return vocab.vocab_size