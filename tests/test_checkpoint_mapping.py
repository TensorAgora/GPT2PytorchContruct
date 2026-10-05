import pytest
import torch

from conftest import needs_checkpoint
from gpt2_pytorch import DistilGPT2LMHeadModel
from gpt2_pytorch.checkpoint import CheckpointError, map_key
from gpt2_pytorch.checkpoint.loader import load_state_dict_into


def hf_state_for(config, seed=0):
    """Random state dict in Hugging Face GPT-2 layout (Conv1D weights are [in, out])."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g)
    c, f, n = config.hidden_size, config.intermediate_size, config.max_position_embeddings
    state = {"transformer.wte.weight": r(config.vocab_size, c), "transformer.wpe.weight": r(n, c)}
    for i in range(config.num_layers):
        p = f"transformer.h.{i}."
        state |= {
            p + "ln_1.weight": r(c), p + "ln_1.bias": r(c),
            p + "attn.bias": torch.tril(torch.ones(1, 1, n, n)),
            p + "attn.c_attn.weight": r(c, 3 * c), p + "attn.c_attn.bias": r(3 * c),
            p + "attn.c_proj.weight": r(c, c), p + "attn.c_proj.bias": r(c),
            p + "ln_2.weight": r(c), p + "ln_2.bias": r(c),
            p + "mlp.c_fc.weight": r(c, f), p + "mlp.c_fc.bias": r(f),
            p + "mlp.c_proj.weight": r(f, c), p + "mlp.c_proj.bias": r(c),
        }
    state |= {"transformer.ln_f.weight": r(c), "transformer.ln_f.bias": r(c)}
    state["lm_head.weight"] = state["transformer.wte.weight"]  # tied, same tensor
    return state


@pytest.mark.parametrize(
    "key,action,target",
    [
        ("transformer.wte.weight", "load", "transformer.embeddings.token_embedding.weight"),
        ("transformer.wpe.weight", "load", "transformer.embeddings.position_embedding.weight"),
        ("transformer.h.3.ln_1.weight", "load", "transformer.blocks.3.ln_1.weight"),
        ("transformer.h.3.attn.c_attn.weight", "transpose", "transformer.blocks.3.attention.qkv_projection.weight"),
        ("transformer.h.3.attn.c_attn.bias", "load", "transformer.blocks.3.attention.qkv_projection.bias"),
        ("transformer.h.3.attn.c_proj.weight", "transpose", "transformer.blocks.3.attention.output_projection.weight"),
        ("transformer.h.3.ln_2.bias", "load", "transformer.blocks.3.ln_2.bias"),
        ("transformer.h.3.mlp.c_fc.weight", "transpose", "transformer.blocks.3.mlp.fc.weight"),
        ("transformer.h.3.mlp.c_proj.weight", "transpose", "transformer.blocks.3.mlp.projection.weight"),
        ("transformer.h.3.mlp.c_proj.bias", "load", "transformer.blocks.3.mlp.projection.bias"),
        ("transformer.ln_f.weight", "load", "transformer.final_layer_norm.weight"),
        ("lm_head.weight", "tie", "lm_head.weight"),
        ("transformer.h.3.attn.bias", "ignore", "transformer.blocks.3.attention.causal_mask"),
        ("transformer.h.3.attn.masked_bias", "ignore", None),
    ],
)
def test_map_key(key, action, target):
    mapping = map_key(key)
    assert (mapping.action, mapping.target) == (action, target)


@pytest.mark.parametrize("key", ["transformer.h.0.attn.c_attn.extra", "transformer.h.0.foo.weight", "something.else", "transformer.wte.bias"])
def test_unknown_keys_are_unmapped(key):
    assert map_key(key) is None


def test_exactly_four_projection_weights_transpose(tiny_config):
    state = hf_state_for(tiny_config)
    transposed = {k for k in state if (m := map_key(k)) and m.action == "transpose"}
    assert len(transposed) == 4 * tiny_config.num_layers
    assert all(k.endswith(".weight") and ("c_attn" in k or "c_proj" in k or "c_fc" in k) for k in transposed)


def test_load_transposes_conv1d_and_nothing_else(tiny_config):
    state = hf_state_for(tiny_config)
    model = DistilGPT2LMHeadModel(tiny_config)
    report = load_state_dict_into(model, state)
    block = model.transformer.blocks[1]
    assert torch.equal(block.attention.qkv_projection.weight, state["transformer.h.1.attn.c_attn.weight"].T)
    assert torch.equal(block.mlp.projection.weight, state["transformer.h.1.mlp.c_proj.weight"].T)
    assert torch.equal(block.ln_1.weight, state["transformer.h.1.ln_1.weight"])  # never transposed
    assert torch.equal(model.transformer.token_embedding.weight, state["transformer.wte.weight"])
    n = tiny_config.num_layers
    assert (len(report.loaded), len(report.transposed), len(report.tied), len(report.ignored)) == (2 + 12 * n + 2, 4 * n, 1, n)
    assert report.missing == [] and report.unexpected == []


def test_unexpected_key_fails_loudly(tiny_config):
    state = hf_state_for(tiny_config) | {"transformer.h.0.attn.rotary.weight": torch.zeros(1)}
    with pytest.raises(CheckpointError, match="rotary"):
        load_state_dict_into(DistilGPT2LMHeadModel(tiny_config), state)


def test_missing_key_fails_loudly(tiny_config):
    state = hf_state_for(tiny_config)
    del state["transformer.h.0.mlp.c_fc.bias"]
    with pytest.raises(CheckpointError, match="mlp.fc.bias"):
        load_state_dict_into(DistilGPT2LMHeadModel(tiny_config), state)


def test_wrong_shape_fails_loudly(tiny_config):
    state = hf_state_for(tiny_config)
    state["transformer.h.0.attn.c_attn.weight"] = state["transformer.h.0.attn.c_attn.weight"].T.contiguous()  # already [out, in]
    with pytest.raises(CheckpointError, match="shape"):
        load_state_dict_into(DistilGPT2LMHeadModel(tiny_config), state)


def test_non_causal_attn_bias_is_not_silently_ignored(tiny_config):
    state = hf_state_for(tiny_config)
    state["transformer.h.0.attn.bias"] = torch.ones_like(state["transformer.h.0.attn.bias"])
    with pytest.raises(CheckpointError, match="causal mask"):
        load_state_dict_into(DistilGPT2LMHeadModel(tiny_config), state)


def test_untied_lm_head_is_rejected(tiny_config):
    state = hf_state_for(tiny_config)
    state["lm_head.weight"] = state["transformer.wte.weight"] + 1
    with pytest.raises(CheckpointError, match="lm_head"):
        load_state_dict_into(DistilGPT2LMHeadModel(tiny_config), state)


def test_missing_lm_head_is_reconstructed_by_tying(tiny_config):
    state = hf_state_for(tiny_config)
    del state["lm_head.weight"]  # what a safetensors file looks like
    model = DistilGPT2LMHeadModel(tiny_config)
    report = load_state_dict_into(model, state)
    assert report.tied == ["lm_head.weight -> transformer.embeddings.token_embedding.weight (reconstructed)"]
    assert torch.equal(model.lm_head.weight, state["transformer.wte.weight"])


@needs_checkpoint
def test_real_checkpoint_report(pretrained):
    _, _, report = pretrained
    assert (len(report.loaded), len(report.transposed), len(report.tied), len(report.ignored)) == (76, 24, 1, 6)
    assert (report.missing, report.unexpected) == ([], [])
