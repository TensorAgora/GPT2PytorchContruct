import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import read_manifest  # noqa: E402
from gpt2_pytorch.registry.descriptor import load  # noqa: E402
from gpt2_pytorch.registry.namemaps import c3, gguf_scheme, hf_tensor, transform_of  # noqa: E402
from gpt2_pytorch.registry.resolve import resolve_and_bind  # noqa: E402


def test_llama_q_maps_both_ways_with_rope_permute():
    hf = load("llama")["schemes"]["hf"]
    assert gguf_scheme(hf)["q"].format(i=0, what="weight") == "blk.0.attn_q.weight"
    assert "ATTN_Q" in hf_tensor(hf["q"]) and hf["q"].format(i=0, what="weight") == "model.layers.0.self_attn.q_proj.weight"
    assert transform_of("blk.0.attn_q.weight", "gguf", "llama") == ["rope_permute"]
    assert transform_of("blk.0.attn_norm.weight", "gguf", "gemma2") == ["offset"]


def test_c3():
    for d in ("llama", "mixtral", "qwen2", "deepseek_v3"):
        assert c3(load(d)["schemes"]["hf"]) == [], d


@pytest.mark.skipif(find("microsoft/Phi-3.5-mini-instruct") is None, reason="fixture missing")
def test_phi3_fused_storage_binds_as_slices():
    c = FixtureClient(find("microsoft/Phi-3.5-mini-instruct"))
    r = resolve_and_bind(read_manifest(c.job(), c, fetched_at="t")[1])
    assert r.status == "ok" and r.binding.ok
    assert r.overrides["Wq_0"] == {"checkpoint_key": "model.layers.0.self_attn.qkv_proj.weight", "view": "[0:3072,:]"}
    assert r.overrides["Wu_0"]["view"] == "[8192:16384,:]"
    assert {t["transform"] for t in r.transforms} == {"split"}


def test_qwen35_gguf_spelled_config_and_names():
    from gpt2_pytorch.registry.build import spell_qwen3_5
    from gpt2_pytorch.registry.namemaps import gguf_rename
    kv = {f"qwen35.{k}": v for k, v in {"block_count": 5, "nextn_predict_layers": 1, "full_attention_interval": 4, "embedding_length": 64,
          "attention.head_count": 4, "attention.head_count_kv": 2, "attention.key_length": 16, "feed_forward_length": 128,
          "attention.layer_norm_rms_epsilon": 1e-6, "ssm.group_count": 2, "ssm.state_size": 16, "ssm.time_step_rank": 4,
          "ssm.inner_size": 64, "ssm.conv_kernel": 4, "rope.freq_base": 1e7, "rope.dimension_count": 4}.items()}
    cfg = spell_qwen3_5(kv, {"token_embd.weight": [100, 64]})
    assert cfg["num_hidden_layers"] == 4 and cfg["layer_types"][3] == "full_attention" and cfg["tie_word_embeddings"]
    stored = {"blk.0.post_attention_norm.weight", "blk.0.ssm_dt.bias", "blk.0.ssm_a", "token_embd.weight", "output_norm.weight"}
    pre = "model.language_model."
    exp = [{"name": n, "checkpoint_key": pre + k} for n, k in [("a", "layers.0.post_attention_layernorm.weight"), ("b", "layers.0.linear_attn.dt_bias"),
                                                                ("c", "layers.0.linear_attn.A_log"), ("d", "embed_tokens.weight"), ("e", "norm.weight")]]
    out = gguf_rename({"tensors": []}, exp, "qwen35", stored, pre)
    assert {e["checkpoint_key"] for e in out} == stored
