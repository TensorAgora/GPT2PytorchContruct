import json
from pathlib import Path

import pytest

from gpt2_pytorch.hub.catalog import Job
from gpt2_pytorch.hub.discover import discover, small_files
from gpt2_pytorch.hub.http import HttpError

CASES = json.loads(Path("tests/fixtures/model_info/cases.json").read_text())


class FakeClient:
    def __init__(self, case, range_status=206):
        self.case, self.range_status, self.ranges = case, range_status, []

    def model_info(self, repo, revision=None, refresh=False):
        return self.case["info"]

    def get_small(self, repo, sha, path):
        return json.dumps(self.case.get("files", {}).get(path, {})).encode()

    def file_url(self, repo, sha, path):
        return f"https://hf/{repo}/resolve/{sha}/{path}"

    def get_range(self, url, start, end):
        self.ranges.append(url)
        if self.range_status != 206:
            raise HttpError(self.range_status, url)
        return b"\0" * (end - start + 1)


def run(name, fmt=None, **kw):
    case = CASES[name]
    job = Job("org/" + name, "decoder_dense", fmt or case.get("job_format", "safetensors"))
    return discover(job, FakeClient(case, **kw))


def paths(d):
    return [w.path for w in d.weights]


def test_sharded_reads_index_shards_only():
    d = run("sharded")
    assert d.status is None and d.sha == "a" * 40
    assert paths(d) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert "consolidated.safetensors" in d.variants


def test_single_root_safetensors():
    d = run("single")
    assert paths(d) == ["model.safetensors"] and d.weights[0].size == 350_000_000


def test_diffusers_components_prefer_plain_file():
    d = run("diffusers")
    assert paths(d) == ["text_encoder/model-00001-of-00001.safetensors", "unet/diffusion_pytorch_model.safetensors",
                        "vae/diffusion_pytorch_model.fp16.safetensors"]
    assert {w.component for w in d.weights} == {"text_encoder", "unet", "vae"}
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in d.variants and "sd_xl_base_1.0.safetensors" in d.variants


def test_gguf_picks_q4_k_m_and_lists_others():
    d = run("gguf_multi_quant")
    assert paths(d) == ["model.Q4_K_M.gguf"] and d.variants == ["model.Q2_K.gguf", "model.Q8_0.gguf"]
    assert d.components == ["mmproj-model-f16.gguf"]


def test_gguf_multi_part_is_one_logical_file_smallest_wins():
    d = run("gguf_multi_part")
    assert paths(d) == ["big-Q8_0.gguf"] and d.weights[0].parts == ["big-Q8_0-00001-of-00002.gguf", "big-Q8_0-00002-of-00002.gguf"]
    assert d.weights[0].size == 100 and d.variants == ["small-IQ2_XS.gguf"]


def test_pytorch_only_has_no_weights_and_deferred_format_makes_no_request():
    assert run("pytorch_only").status == "skipped:no_weights"
    assert run("pytorch_only", fmt="pytorch").status == "skipped:deferred_format"


def test_small_files():
    d = run("diffusers")
    got = small_files(d, FakeClient(CASES["diffusers"]))
    assert sorted(got) == ["model_index.json", "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
                           "unet/config.json", "vae/config.json"]


def test_gated(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert run("gated").status == "skipped:gated"
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert run("gated", range_status=403).status == "skipped:gated_not_accepted"
    assert run("gated").status is None


def test_component_folders_without_model_index():
    d = run("components_no_model_index")
    assert paths(d) == ["vae/diffusion_pytorch_model.safetensors"] and d.weights[0].component == "vae"
