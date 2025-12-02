"""Embedding module that encodes spelling of tokens.

- Same interface as nn.Embedding.
- In addition to having an embedding per token, it has an embedding per character.
- The embedding for a token is the sum of the embeddings for its characters, rotated
  by an angle corresponding to the characters position in the token.

Both, token embeddings and character embeddings are learned. The rotation angles are
not learned, and instead we use rotary position embeddings.

Characters of each token are stored in the embedding module. We fix a limit of 32
characters per token.
"""

import random
import math

import torch
import torch.nn as nn
import torchtune

import initialization
import fp32norm


class SpellingBeeEmbedding(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        vocab: list[str],
        separate_token_embedding: bool,
        weight_dtype: torch.dtype,
        max_characters: int = 16,
        character_norm: bool = True,
        char_init_scale: float = 1.0,
        apply_rotary: bool = True,
        scale: float = 1.0,
        spelling_type: str = "full",  # one of "full", "dummy", "shuffled", "double", "static_emb"
        rotary_base: int = 10000,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.max_characters_per_token = max_characters
        self.separate_token_embedding = separate_token_embedding
        vocab_bytes = [token.encode("utf-8") for token in vocab]
        self.character_norm = character_norm
        self.char_init_scale = char_init_scale
        self.apply_rotary = apply_rotary
        self.scale = scale
        self.spelling_type = spelling_type
        self.rotary_base = rotary_base

        self.vocab_character_table = self._vocab_character_table(vocab_bytes)

        if separate_token_embedding:
            self.token_embedding = nn.Embedding(
                num_tokens, embedding_dim, dtype=weight_dtype
            )
        if self.spelling_type == "double":
            self.token_embedding_2 = nn.Embedding(
                num_tokens, embedding_dim, dtype=weight_dtype
            )
        if self.spelling_type == "static_emb":
            # just a single embedding vector
            self.aux_embedding = nn.Parameter(
                torch.empty(1, embedding_dim, dtype=weight_dtype)
            )
        self.character_embedding = nn.Embedding(256, embedding_dim, dtype=weight_dtype)
        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(
            dim=embedding_dim, max_seq_len=max_characters, base=self.rotary_base
        )
        if character_norm:
            self.char_emb_norm = fp32norm.FP32LayerNorm(embedding_dim)

    def init_weights(self):
        """Initialize weights with proper variance-preserving initialization."""

        # The variance of independent random variables is additive. For correlated
        # random variables it's even worse. So we scale down the correlated variables
        # (the character embeddings) significantly.
        if self.separate_token_embedding:
            initialization.init_embedding(self.token_embedding, scaling_factor=1.0)
        initialization.init_embedding(
            self.character_embedding,
            scaling_factor=self.char_init_scale,  # / self.max_characters_per_token,
        )
        if self.spelling_type == "double":
            initialization.init_embedding(self.token_embedding_2, scaling_factor=1.0)
        if self.spelling_type == "aux_embedding":
            # manual initialization
            with torch.no_grad():
                self.aux_embedding.normal_(0, 1.0 / math.sqrt(self.embedding_dim))
        if self.character_norm:
            self.char_emb_norm.init_weights(init_val=1.0)

    def _vocab_character_table(self, vocab_bytes: list[bytes]) -> torch.Tensor:
        """Compute the character table for the vocabulary."""
        assert self.spelling_type in [
            "full",
            "dummy",
            "shuffled",
            "double",
            "static_emb",
        ]
        # raise ValueError(f"Unknown spelling type: {self.spelling_type}")
        if self.spelling_type in ["dummy", "double"]:
            return torch.zeros(
                (self.num_tokens, self.max_characters_per_token), dtype=torch.int32
            )
        if self.spelling_type == "shuffled":
            # random permutation of the tokens
            random.seed(4253217)
            vocab_bytes = random.sample(vocab_bytes, len(vocab_bytes))
        character_table = torch.zeros(
            (self.num_tokens, self.max_characters_per_token), dtype=torch.int32
        )
        num_tokens_clipped = 0
        for token_idx, token_bytes in enumerate(vocab_bytes):
            token_characters = []
            for char_idx, char_byte in enumerate(token_bytes):
                if char_idx >= self.max_characters_per_token:
                    num_tokens_clipped += 1
                    break
                token_characters.append(char_byte)
            character_table[token_idx, : len(token_characters)] = torch.tensor(
                token_characters, dtype=torch.int32
            )
        if num_tokens_clipped > 0:
            print(
                f"Warning: clipped {num_tokens_clipped} tokens to {self.max_characters_per_token} characters"
            )
        return character_table

    def get_character_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        """input is a tensor of shape (batch_size, sequence_length) of token ids."""
        # Use advanced indexing to gather all characters at once
        # input shape: (batch_size, sequence_length)
        # vocab_character_table shape: (num_tokens, max_characters)
        batch_size, sequence_length = input.shape
        characters = self.vocab_character_table[input]
        assert characters.shape == (
            batch_size,
            sequence_length,
            self.max_characters_per_token,
        )
        embeddings = self.character_embedding(characters)
        assert embeddings.shape == (
            batch_size,
            sequence_length,
            self.max_characters_per_token,
            self.embedding_dim,
        )
        if self.apply_rotary:
            embeddings = self._apply_rotary(embeddings)
        embeddings = embeddings.mean(dim=-2)
        if self.character_norm:
            embeddings = self.char_emb_norm(embeddings)
        if self.scale != 1.0:
            embeddings = embeddings * self.scale
        assert embeddings.shape == (batch_size, sequence_length, self.embedding_dim)
        return embeddings

    def _apply_rotary(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_characters, embedding_dim = embeddings.shape
        orig_shape = embeddings.shape
        # We need to pretend that we have the shape (batch, sequence, num_heads, head_dim),
        # but we want the rotation to happen along the position of the character in the
        # token. We also don't have multiple heads.
        embeddings = embeddings.view(
            batch_size * sequence_length, max_characters, 1, embedding_dim
        )
        embeddings = self.rotary_emb(embeddings)
        embeddings = embeddings.view(*orig_shape)
        return embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """input is a tensor of shape (batch_size, sequence_length) of token ids."""
        if self.spelling_type == "double":
            embeddings = self.token_embedding_2(input)
        elif self.spelling_type == "static_emb":
            embeddings = self.aux_embedding
        else:
            embeddings = self.get_character_embeddings(input)
        if self.separate_token_embedding:
            token_embeddings = self.token_embedding(input)
            embeddings = 0.5 * embeddings + 0.5 * token_embeddings
        return embeddings
