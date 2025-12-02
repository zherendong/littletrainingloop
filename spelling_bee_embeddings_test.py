"""
Test script for the spelling_bee_embeddings.py module.
"""

import torch
import spelling_bee_embeddings


def test_spelling_bee_embedding_shape():
    vocab = ["a", "b", "c"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=True,
        weight_dtype=torch.float32,
    )
    input = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32)
    output = embedding(input)
    assert output.shape == (2, 3, 8)


def test_equal_spelling_means_equal_embedding():
    vocab = ["a", "a"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=False,
        weight_dtype=torch.float32,
    )
    embedding.init_weights()
    input = torch.tensor([[0], [1]], dtype=torch.int32)
    output = embedding(input)
    assert output.shape == (2, 1, 8)
    assert torch.allclose(output[0], output[1])


def test_different_spelling_means_different_embedding():
    vocab = ["a", "b"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=False,
        weight_dtype=torch.float32,
    )
    input = torch.tensor([[0], [1]], dtype=torch.int32)
    output = embedding(input)
    assert not torch.allclose(output[0], output[1])


def test_separate_token_embedding():
    """If we have a separate token embedding, the output should be different, even if the spelling is the same."""
    vocab = ["a", "a"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=True,
        weight_dtype=torch.float32,
    )
    input = torch.tensor([[0], [1]], dtype=torch.int32)
    output = embedding(input)
    assert output.shape == (2, 1, 8)
    assert not torch.allclose(output[0], output[1])


def test_get_character_embeddings():
    vocab = ["a", "b", "c"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=True,
        weight_dtype=torch.float32,
    )
    input = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32)
    character_embeddings = embedding.get_character_embeddings(input)
    assert character_embeddings.shape == (2, 3, 8)


def test_character_table():
    vocab = ["abc", "123"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=True,
        max_characters=8,
        weight_dtype=torch.float32,
    )
    assert embedding.vocab_character_table.shape == (2, 8)
    assert embedding.vocab_character_table[0, 0] == ord("a")
    assert embedding.vocab_character_table[0, 1] == ord("b")
    assert embedding.vocab_character_table[0, 2] == ord("c")
    assert embedding.vocab_character_table[0, 3] == 0
    assert embedding.vocab_character_table[1, 0] == ord("1")
    assert embedding.vocab_character_table[1, 1] == ord("2")
    assert embedding.vocab_character_table[1, 2] == ord("3")
    assert embedding.vocab_character_table[1, 3] == 0


def test_dummy_mode():
    vocab = ["abc", "123"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab),
        embedding_dim=8,
        vocab=vocab,
        separate_token_embedding=True,
        max_characters=8,
        weight_dtype=torch.float32,
        spelling_type="dummy",
    )
    assert embedding.vocab_character_table.shape == (2, 8)
    assert embedding.vocab_character_table[0, 0] == 0
    assert embedding.vocab_character_table[0, 1] == 0
    assert embedding.vocab_character_table[0, 2] == 0
    assert embedding.vocab_character_table[0, 3] == 0
    assert embedding.vocab_character_table[1, 0] == 0
