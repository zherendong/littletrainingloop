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

import torch
import torch.nn as nn
import torchtune

import initialization


class SpellingBeeEmbedding(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        vocab: list[str],
        separate_token_embedding: bool,
        max_characters: int = 16,
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.max_characters_per_token = max_characters
        self.separate_token_embedding = separate_token_embedding
        vocab_bytes = [token.encode("utf-8") for token in vocab]
        self.vocab_character_table = self._vocab_character_table(vocab_bytes)

        if separate_token_embedding:
            self.token_embedding = nn.Embedding(
                num_tokens, embedding_dim, dtype=weight_dtype
            )
        self.character_embedding = nn.Embedding(256, embedding_dim, dtype=weight_dtype)
        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(
            dim=embedding_dim, max_seq_len=max_characters
        )

    def init_weights(self):
        print("Initializing spelling bee")
        if self.separate_token_embedding:
            initialization.init_embedding(self.token_embedding)
        initialization.init_embedding(self.character_embedding)
        self.character_embedding.weight.data[0] = 0

    def _vocab_character_table(self, vocab_bytes: list[bytes]) -> torch.Tensor:
        """Compute the character table for the vocabulary."""
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
        embeddings = self.get_character_embeddings(input)
        embeddings = self._apply_rotary(embeddings)
        embeddings = embeddings.sum(dim=-2)
        if self.separate_token_embedding:
            token_embeddings = self.token_embedding(input)
            embeddings += token_embeddings
        # if torch.isnan(embeddings).any():
        #     raise ValueError("NaN in character embeddings")
        # print(
        #     f"Character embeddings: mean {embeddings.mean()}, std {embeddings.std()}, max {embeddings.max()}, min {embeddings.min()}"
        # )
        return embeddings
