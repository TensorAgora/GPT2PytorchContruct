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
