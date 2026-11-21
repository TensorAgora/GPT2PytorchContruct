import pytest

from gpt2_pytorch.registry.descriptor import DescriptorError, evaluate, hyper_from_config, layer_blocks, load


def test_first_k():
    assert layer_blocks({"first_k": "first_k_dense_replace", "first": "dense", "then": "moe"},
                        {"first_k_dense_replace": 3}, 6) == ["dense"] * 3 + ["moe"] * 3


def test_uniform_list_period_pattern():
    assert layer_blocks({"uniform": "gqa"}, {}, 2) == ["gqa", "gqa"]
    cfg = {"layer_types": ["linear_attention", "full_attention"] * 2, "p": 4, "pat": "M*M-"}
    assert layer_blocks({"list": "layer_types", "map": {"linear_attention": "delta_rule", "full_attention": "gqa"}}, cfg, 4) == \
        ["delta_rule", "gqa", "delta_rule", "gqa"]
    assert layer_blocks({"period": "p", "phase": 1, "on": "attn", "off": "mamba"}, cfg, 6) == \
        ["mamba", "attn", "mamba", "mamba", "mamba", "attn"]
    assert layer_blocks({"pattern": "pat", "map": {"M": "mamba", "*": "attn", "-": "mlp"}}, cfg, 4) == ["mamba", "attn", "mamba", "mlp"]


def test_index_sets_per_layer_by_presence():
    assert layer_blocks({"index_sets": {"gqa": "full"}, "default": "swa"}, {"full": [1, 3]}, 4) == ["swa", "gqa", "swa", "gqa"]
    cfg = {"layers": [{"kind": "a"}, {"kind": "b"}]}
    assert layer_blocks({"per_layer_config": "layers", "field": "kind", "map": {"a": "gqa", "b": "mla"}}, cfg, 2) == ["gqa", "mla"]
    names = ["model.layers.1.mlp.experts.0.up_proj.weight"]
    assert layer_blocks({"by_presence": {"moe": "model.layers.{i}.mlp.experts.0.up_proj.weight"}, "default": "dense"}, {}, 3, names) == \
        ["dense", "moe", "dense"]


def test_evaluate_is_arithmetic_only():
    assert evaluate("D // Hq", {"D": 3584, "Hq": 28}) == 128
    with pytest.raises(DescriptorError):
        evaluate("__import__('os')", {})


def test_qwen2_descriptor_and_hyper():
    d = load("qwen2")
    assert d["recipe"]["flags"]["qkv_bias"] and d["schemes"]["hf"]["embed"] == "model.embed_tokens.weight"
    h = hyper_from_config(d, {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128, "vocab_size": 10})
    assert h["Hkv"] == (4, "default:Hq") and h["Dh"] == (16, "default:D // Hq") and h["D"] == (64, "config:hidden_size")


def test_selector_else_chain_reads_layer_types_then_presence():
    from gpt2_pytorch.registry.descriptor import layer_blocks
    sel = {"index_sets": {"gqa": "full_attn_idxs"}, "default": "conv",
           "else": {"list": "layer_types", "map": {"conv": "conv", "full_attention": "gqa"},
                    "else": {"by_presence": {"gqa": ["a.{i}.q", "blk.{i}.attn_q.weight"]}, "default": "conv"}}}
    assert layer_blocks(sel, {"full_attn_idxs": [1]}, 3) == ["conv", "gqa", "conv"]
    assert layer_blocks(sel, {"layer_types": ["full_attention", "conv", "conv"]}, 3) == ["gqa", "conv", "conv"]
    assert layer_blocks(sel, {}, 3, names=["blk.2.attn_q.weight"]) == ["conv", "conv", "gqa"]


def test_depthwise_conv_layouts():
    from gpt2_pytorch.registry.bind import conv_layout
    assert "GGUF" in conv_layout([64, 1, 3], [64, 3]) and "MLX" in conv_layout([64, 1, 3], [64, 3, 1])
    assert conv_layout([64, 1, 3], [3, 64]) is None and conv_layout([64, 2, 3], [64, 3, 2]) is None


def test_mlx_stacked_experts_bind_one_tensor():
    from gpt2_pytorch.registry.bind import stacked_key, verify
    view = "stack of model.layers.0.mlp.experts.{0…3}.gate_proj.weight"
    sk = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    assert stacked_key(view) == sk
    e = [{"name": "Wge_0", "checkpoint_key": None, "view": view, "shape": [4, 8, 16], "role": "parameter"}]
    m = {"tensors": [{"key": sk, "dtype": "BF16", "shape": [4, 8, 16], "file": "m.safetensors", "numel": 512}], "config": {}}
    b = verify(e, m, {"ignored": []})
    assert b.ok and b.keys == {"Wge_0": sk} and b.partition["bound"] == [sk]
