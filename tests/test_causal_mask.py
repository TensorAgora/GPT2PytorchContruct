import torch

from conftest import needs_checkpoint
from gpt2_pytorch.model import CausalSelfAttention


def test_mask_is_lower_triangular(tiny_config):
    attn = CausalSelfAttention(tiny_config).eval()
    _, d = attn(torch.randn(1, 5, 32), return_debug=True)
    assert torch.equal(d["causal_mask"], torch.tril(torch.ones(5, 5, dtype=torch.bool)))


def test_future_probabilities_are_zero_and_rows_sum_to_one(tiny_config):
    attn = CausalSelfAttention(tiny_config).eval()
    _, d = attn(torch.randn(3, 7, 32), return_debug=True)
    probs = d["attention_probs"]  # [B, H, T, T]
    future = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    assert probs[..., future].abs().max().item() == 0.0
    torch.testing.assert_close(probs.sum(-1), torch.ones(3, 4, 7))
    assert (probs[..., ~future] >= 0).all()


def test_changing_future_tokens_does_not_change_past_outputs(tiny_model, tiny_ids):
    changed = tiny_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 13) % 100
    with torch.no_grad():
        a, b = tiny_model(tiny_ids), tiny_model(changed)
    assert torch.equal(a[:, :4], b[:, :4])  # positions 0..3 never saw tokens 4..
    assert not torch.equal(a[:, 4:], b[:, 4:])


@needs_checkpoint
def test_real_model_every_block_and_head_is_causal(real_model):
    ids = torch.tensor([[464, 2068, 7586, 21831, 18045, 625, 262, 16931]])
    with torch.no_grad():
        debug = real_model(ids, return_debug=True).debug
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    for block in range(6):
        probs = debug[f"block.{block}.attention.probs"]
        assert probs[..., future].max().item() == 0.0
        torch.testing.assert_close(probs.sum(-1), torch.ones(1, 12, 8))
