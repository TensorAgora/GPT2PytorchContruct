import copy
from functools import cache

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import read_manifest  # noqa: E402
from gpt2_pytorch.registry.resolve import candidates, resolve_and_bind  # noqa: E402


def synthetic(cfg, small=None, fmt="safetensors"):
    return {"model": {"id": "o/m", "format": fmt, "category": "decoder_moe"}, "config": {"config.json": cfg, **(small or {})},
            "tensors": [], "gguf": None}


@cache
def fixture(repo):
    c = FixtureClient(find(repo))
    return read_manifest(c.job(), c, fetched_at="t")[1]


def test_kimi_k2_resolves_by_architectures_to_deepseek_v3():
    cfg = {"model_type": "kimi_k2", "architectures": ["DeepseekV3ForCausalLM"], "auto_map": {"AutoModel": "modeling_deepseek.X"}}
    cands, status = candidates(synthetic(cfg))
    assert status is None and cands[0].descriptor == "deepseek_v3" and cands[0].rung == "R2"


def test_remote_code_without_matching_architecture_is_skipped():
    cfg = {"model_type": "retnet", "architectures": ["RetNetForCausalLM"], "auto_map": {"AutoModel": "modeling_retnet.X"}}
    assert candidates(synthetic(cfg)) == ([], "skipped:remote_code")


def test_gguf_architecture_survives_a_remote_code_flag():  # D-24
    m = synthetic({}, fmt="gguf") | {"gguf": {"kv": {"general.architecture": "qwen3"}}}
    cands, status = candidates(m, {"trust_remote_code": "true"})
    assert status is None and [(c.descriptor, c.rung) for c in cands] == [("qwen3", "R1")]


def test_diffusers_and_timm_give_library_classes():
    cands, _ = candidates(synthetic({}, {"model_index.json": {"_class_name": "StableDiffusionPipeline"},
                                         "unet/config.json": {"_class_name": "UNet2DConditionModel"}}))
    assert "diffusers:UNet2DConditionModel" in {c.library_class for c in cands}
    cands, _ = candidates(synthetic({"architecture": "resnet50", "pretrained_cfg": {}}))
    assert [c.library_class for c in cands] == ["timm:resnet50"]


@pytest.mark.skipif(find("distilbert/distilgpt2") is None, reason="fixture missing")
def test_fallthrough_records_near_miss():
    m = copy.deepcopy(fixture("distilbert/distilgpt2"))
    m["config"]["config.json"]["architectures"] = ["LlamaForCausalLM", "GPT2LMHeadModel"]
    r = resolve_and_bind(m)
    assert r.status == "ok" and r.descriptor["id"] == "gpt2"
    assert r.near_misses and r.near_misses[0]["descriptor"].startswith("llama@")


@pytest.mark.skipif(find("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF") is None, reason="fixture missing")
def test_gguf_mixtral_binds_through_r1():
    r = resolve_and_bind(fixture("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF"))
    assert r.status == "ok" and r.descriptor["id"] == "mixtral" and r.candidate.rung == "R1" and r.binding.ok


def test_bert_self_output_owns_the_projection():  # TEL hosts each region in the smallest module holding it: no ties
    r = resolve_and_bind(fixture("google-bert/bert-base-uncased"))
    assert r.status == "ok"

    def groups(n):
        yield n
        for c in n["children"]:
            yield from groups(c)
    by = {g["id"]: g for g in groups(r.graph["groups"])}
    so, ln = by["bert.encoder.layer.0.attention.output"], by["bert.encoder.layer.0.attention.output.LayerNorm"]
    assert so["module_type"] == "SelfOutput" and len(so["members"]) > len(ln["members"])
