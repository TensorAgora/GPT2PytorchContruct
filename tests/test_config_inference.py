import pytest
import torch

from conftest import needs_checkpoint
from gpt2_pytorch import GPT2Config
from gpt2_pytorch.checkpoint import infer_config


@needs_checkpoint
def test_checkpoint_confirms_distilgpt2(checkpoint_state):
    config = infer_config(checkpoint_state)
    assert (config.vocab_size, config.max_position_embeddings, config.hidden_size) == (50257, 1024, 768)
    assert (config.num_layers, config.num_heads, config.intermediate_size) == (6, 12, 3072)
    assert config.head_dim == 64
    assert config == GPT2Config()


def test_inference_from_synthetic_shapes():
    c, layers = 128, 3
    state = {
        "transformer.wte.weight": torch.empty(500, c),
        "transformer.wpe.weight": torch.empty(32, c),
        "transformer.ln_f.weight": torch.empty(c),
    }
    for i in range(layers):
        p = f"transformer.h.{i}."
        state[p + "attn.c_attn.weight"] = torch.empty(c, 3 * c)
        state[p + "attn.c_proj.weight"] = torch.empty(c, c)
        state[p + "mlp.c_fc.weight"] = torch.empty(c, 4 * c)
        state[p + "mlp.c_proj.weight"] = torch.empty(4 * c, c)
    config = infer_config(state, head_dim=32)
    assert (config.vocab_size, config.max_position_embeddings, config.hidden_size) == (500, 32, 128)
    assert (config.num_layers, config.num_heads, config.intermediate_size) == (3, 4, 512)


def test_inconsistent_shapes_fail_loudly():
    state = {
        "transformer.wte.weight": torch.empty(10, 64),
        "transformer.wpe.weight": torch.empty(8, 64),
        "transformer.ln_f.weight": torch.empty(64),
        "transformer.h.0.attn.c_attn.weight": torch.empty(64, 100),  # not 3C
        "transformer.h.0.mlp.c_fc.weight": torch.empty(64, 256),
    }
    with pytest.raises(AssertionError):
        infer_config(state)


def test_config_rejects_bad_head_split():
    with pytest.raises(ValueError):
        GPT2Config(hidden_size=100, num_heads=12)
