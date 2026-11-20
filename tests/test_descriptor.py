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
