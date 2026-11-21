"""Path B (master R-12): library classes traced on the meta device bind real headers; C2 against path A."""
import json

import pytest

pytest.importorskip("timm")
pytest.importorskip("diffusers")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import load_json  # noqa: E402
from gpt2_pytorch.hub.pipeline import load_graph, run_one  # noqa: E402
from tools.graph_check import check  # noqa: E402


@pytest.mark.parametrize("repo, status, bound", [("timm/test_convnext.r160_in1k", "ok", 101), ("stabilityai/sd-turbo", "partial", 686)])
def test_tiny_timm_and_unet_bind(repo, status, bound, tmp_path):
    fx = find(repo)
    if fx is None:
        pytest.skip(f"{repo} fixture missing")
    c = FixtureClient(fx)
    r = run_one(c.job(), tmp_path, c, fetched_at="t")
    rep = json.loads((r.dir / "report.json").read_text())
    assert r.status == status and rep["winner"]["path"] == "B"
    assert rep["binding"]["bound"] == bound and rep["binding"]["unclaimed"] == 0
    g = load_graph(r.dir)
    assert g["provenance"]["path"] == "B" and any(o["primitive"] == "Opaque" for o in g["ops"])
    if repo.startswith("timm"):  # every parameter of the traced graph names a stored tensor of the same shape
        assert check(g, load_json(r.dir / "manifest.json")) == []
    else:
        assert check(g) == []


@pytest.mark.parametrize("descriptor", ["llama", "gpt2", "qwen3_moe", "jamba", "t5", "bert"])
def test_c2_path_a_equals_path_b(descriptor):
    from tools.conformance import c2
    assert c2(descriptor) == "PASS"


@pytest.mark.parametrize("repo", ["FacebookAI/roberta-base", "google-t5/t5-small", "distilbert/distilgpt2", "facebook/bart-base", "Qwen/Qwen3-0.6B"])
def test_path_b_binds_transformers_checkpoints_without_descriptors(repo, tmp_path, monkeypatch):
    """saved class first, wrapper prefix aligned, ties and the class's ignore-on-load keys classed (descriptors hidden)"""
    from gpt2_pytorch.registry import resolve
    fx = find(repo)
    if fx is None:
        pytest.skip(f"{repo} fixture missing")
    orig = resolve.candidates
    monkeypatch.setattr(resolve, "candidates", lambda m, r: ([c for c in orig(m, r)[0] if c.descriptor is None], orig(m, r)[1]))
    c = FixtureClient(fx)
    r = run_one(c.job(), tmp_path, c, fetched_at="t")
    rep = json.loads((r.dir / "report.json").read_text())
    assert r.status == "ok" and rep["winner"]["path"] == "B", rep.get("near_misses")
    assert check(load_graph(r.dir), load_json(r.dir / "manifest.json")) == []


def test_denoiser_inputs_cover_sdxl_and_dit_classes():
    from gpt2_pytorch.registry.trace import denoiser_inputs
    sdxl = {"in_channels": 4, "sample_size": 128, "cross_attention_dim": 2048, "addition_embed_type": "text_time",
            "projection_class_embeddings_input_dim": 2816, "addition_time_embed_dim": 256}
    _, kw = denoiser_inputs("UNet2DConditionModel", sdxl)
    assert list(kw["added_cond_kwargs"]["text_embeds"].shape) == [1, 1280] and list(kw["added_cond_kwargs"]["time_ids"].shape) == [1, 6]
    flux = {"in_channels": 64, "joint_attention_dim": 4096, "pooled_projection_dim": 768, "guidance_embeds": True}
    args, kw = denoiser_inputs("FluxTransformer2DModel", flux)
    assert list(args[0].shape) == [1, 16, 64] and "guidance" in kw and list(kw["img_ids"].shape) == [16, 3]
    assert denoiser_inputs("MiniMaxH3Transformer3DModel", {}) is None
