import math

import torch
import torch.nn.functional as F
from torch import nn

import gpt2_pytorch
from gpt2_pytorch.model import CausalSelfAttention


def make_attention(config):
    torch.manual_seed(1)
    return CausalSelfAttention(config).eval(), torch.randn(2, 6, config.hidden_size)


def test_debug_tensors_and_shapes(tiny_config):
    attn, x = make_attention(tiny_config)
    output, debug = attn(x, return_debug=True)
    B, T, C, H, D = 2, 6, 32, 4, 8
    shapes = {k: tuple(v.shape) for k, v in debug.items()}
    assert shapes == {
        "qkv": (B, T, 3 * C), "query": (B, H, T, D), "key": (B, H, T, D), "value": (B, H, T, D),
        "attention_scores": (B, H, T, T), "causal_mask": (T, T), "attention_probs": (B, H, T, T), "context": (B, H, T, D),
    }
    assert output.shape == (B, T, C)


def test_qkv_split_order_and_scaling(tiny_config):
    attn, x = make_attention(tiny_config)
    _, d = attn(x, return_debug=True)
    qkv = attn.qkv_projection(x)
    q, k, v = qkv.chunk(3, dim=-1)  # Q first, then K, then V
    split = lambda t: t.reshape(2, 6, 4, 8).permute(0, 2, 1, 3)
    assert torch.equal(d["query"], split(q)) and torch.equal(d["key"], split(k)) and torch.equal(d["value"], split(v))
    assert torch.allclose(d["attention_scores"], d["query"] @ d["key"].transpose(-2, -1) / math.sqrt(8))


def test_matches_torch_sdpa_reference(tiny_config):
    # test-only cross-check: the library never calls SDPA
    attn, x = make_attention(tiny_config)
    _, d = attn(x, return_debug=True)
    reference = F.scaled_dot_product_attention(d["query"], d["key"], d["value"], is_causal=True)
    torch.testing.assert_close(d["context"], reference, rtol=1e-5, atol=1e-6)


def test_output_projection_of_merged_heads(tiny_config):
    attn, x = make_attention(tiny_config)
    output, d = attn(x, return_debug=True)
    merged = d["context"].permute(0, 2, 1, 3).reshape(2, 6, 32)
    assert torch.allclose(output, attn.output_projection(merged), atol=1e-6)


def test_debug_sink_receives_same_tensors_and_plain_call_unchanged(tiny_config):
    attn, x = make_attention(tiny_config)
    got = []
    attn.debug_sink = got.append
    plain = attn(x)
    assert torch.is_tensor(plain) and len(got) == 1 and set(got[0]) >= {"query", "attention_probs"}
    attn.debug_sink = None
    assert torch.equal(plain, attn(x))


def test_no_multihead_attention_or_sdpa_in_library():
    from pathlib import Path

    src = Path(gpt2_pytorch.__file__).parent
    text = "\n".join(p.read_text() for p in src.rglob("*.py"))
    assert "scaled_dot_product_attention(" not in text and "MultiheadAttention(" not in text  # docstrings may name them
