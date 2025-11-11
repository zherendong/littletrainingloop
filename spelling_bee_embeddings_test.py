"""
Test script for the spelling_bee_embeddings.py module.
"""

import torch
import spelling_bee_embeddings


def test_spelling_bee_embedding_shape():
    vocab = ["a", "b", "c"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab), embedding_dim=8, vocab=vocab
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
    )
    input = torch.tensor([[0], [1]], dtype=torch.int32)
    output = embedding(input)
    assert output.shape == (2, 1, 8)
    assert not torch.allclose(output[0], output[1])


def test_get_character_embeddings():
    vocab = ["a", "b", "c"]
    embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
        num_tokens=len(vocab), embedding_dim=8, vocab=vocab
    )
    input = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32)
    character_embeddings = embedding.get_character_embeddings(input)
    assert character_embeddings.shape == (2, 3, 16, 8)
