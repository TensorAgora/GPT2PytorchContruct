import torch

from gpt2_pytorch.model import TransformerBlock


def test_residual_structure(tiny_config):
    torch.manual_seed(0)
    block = TransformerBlock(tiny_config).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        after_attention = x + block.attention(block.ln_1(x))
        expected = after_attention + block.mlp(block.ln_2(after_attention))
        output, d = block(x, return_debug=True)
    assert torch.equal(output, expected) and torch.equal(block(x), expected)
    assert torch.equal(d["attention_residual.output"], after_attention)
    assert torch.equal(d["input"], x) and torch.equal(d["output"], output)


def test_block_debug_names(tiny_config):
    block = TransformerBlock(tiny_config).eval()
    _, d = block(torch.randn(1, 4, 32), return_debug=True)
    assert set(d) == {
        "input", "ln_1.output", "attention.qkv", "attention.q", "attention.k", "attention.v", "attention.scores",
        "attention.mask", "attention.probs", "attention.context", "attention.output", "attention_residual.output",
        "ln_2.output", "mlp.fc.output", "mlp.activation.output", "mlp.output", "output",
    }
