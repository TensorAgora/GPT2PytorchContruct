import torch

from conftest import needs_checkpoint
from gpt2_pytorch.checkpoint.inspect import checkpoint_tree, inspect_state_dict, summarize

pytestmark = needs_checkpoint


def test_tensor_inventory_of_checkpoint(checkpoint_state):
    infos = inspect_state_dict(checkpoint_state)
    s = summarize(infos)
    assert s["tensor_count"] == 83  # 2 embeddings + 6 blocks * 13 + ln_f(2) + lm_head
    assert {i.dtype for i in infos} == {"torch.float32"}
    assert s["logical_numel"] == 126_801_408
    # unique storage = 81,912,576 parameters + 6 * 1024^2 causal-mask buffers
    assert s["unique_numel"] == 81_912_576 + 6 * 1024 * 1024
    assert s["unique_bytes"] == 4 * s["unique_numel"]


def test_lm_head_shares_storage_with_wte(checkpoint_state):
    s = summarize(inspect_state_dict(checkpoint_state))
    assert s["tied"] == {"transformer.wte.weight": ["lm_head.weight"], "lm_head.weight": ["transformer.wte.weight"]}


def test_conv1d_shapes_are_in_by_out(checkpoint_state):
    shapes = {k: tuple(v.shape) for k, v in checkpoint_state.items()}
    assert shapes["transformer.h.0.attn.c_attn.weight"] == (768, 2304)
    assert shapes["transformer.h.0.attn.c_proj.weight"] == (768, 768)
    assert shapes["transformer.h.0.mlp.c_fc.weight"] == (768, 3072)
    assert shapes["transformer.h.0.mlp.c_proj.weight"] == (3072, 768)
    assert shapes["transformer.h.0.attn.bias"] == (1, 1, 1024, 1024)


def test_tree_collapses_identical_blocks(checkpoint_state):
    tree = checkpoint_tree(inspect_state_dict(checkpoint_state))
    assert "wte" in tree and "c_attn" in tree and "same structure as 0" in tree
    assert "weight [50257,768]" in tree
