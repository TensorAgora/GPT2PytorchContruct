import pytest
import torch

from gpt2_pytorch.model import GPT2Embeddings, PositionEmbedding, TokenEmbedding


def test_embedding_is_token_plus_position(tiny_config, tiny_ids):
    emb = GPT2Embeddings(tiny_config).eval()
    hidden, debug = emb(tiny_ids, return_debug=True)
    assert hidden.shape == (2, 6, 32)
    assert debug["token.output"].shape == (2, 6, 32) and debug["position.output"].shape == (1, 6, 32)
    expected = emb.token_embedding.weight[tiny_ids] + emb.position_embedding.weight[:6]
    assert torch.equal(hidden, expected)


def test_module_types(tiny_config):
    emb = GPT2Embeddings(tiny_config)
    assert isinstance(emb.token_embedding, TokenEmbedding) and isinstance(emb.position_embedding, PositionEmbedding)
    assert emb.token_embedding.weight.shape == (100, 32) and emb.position_embedding.weight.shape == (16, 32)


def test_explicit_position_ids(tiny_config, tiny_ids):
    emb = GPT2Embeddings(tiny_config).eval()
    shifted = (torch.arange(6) + 3).unsqueeze(0)
    assert not torch.equal(emb(tiny_ids), emb(tiny_ids, position_ids=shifted))
    assert torch.equal(emb(tiny_ids), emb(tiny_ids, position_ids=torch.arange(6).unsqueeze(0)))


def test_sequence_longer_than_max_positions_fails(tiny_config):
    emb = GPT2Embeddings(tiny_config).eval()
    with pytest.raises(AssertionError, match="max_position_embeddings"):
        emb(torch.zeros(1, 17, dtype=torch.long))


def test_embedding_dropout_only_in_train_mode(tiny_config, tiny_ids):
    emb = GPT2Embeddings(tiny_config)
    emb.eval()
    assert torch.equal(emb(tiny_ids), emb(tiny_ids))
