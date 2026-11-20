import copy

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("gguf")
from gpt2_pytorch.hub.catalog import Job  # noqa: E402
from gpt2_pytorch.hub.discover import Discovery, Weight, discover, small_files  # noqa: E402
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.headers import read_header  # noqa: E402
from gpt2_pytorch.hub.manifest import build_manifest, dump, validate_manifest  # noqa: E402

FIX = find("distilbert/distilgpt2")


def distilgpt2(fetched_at):
    c = FixtureClient(FIX)
    job = c.job()
    d = discover(job, c)
    headers = {p: read_header(c, d.repo, d.sha, p) for w in d.weights for p in w.parts}
    return build_manifest(job, d, small_files(d, c), headers, fetched_at)


@pytest.mark.skipif(FIX is None, reason="distilgpt2 header fixture not captured")
def test_fixture_manifest_validates_and_is_deterministic():
    a, b = distilgpt2("2026-01-01T00:00:00+00:00"), distilgpt2("2026-02-02T00:00:00+00:00")
    assert validate_manifest(a) == []
    assert a["stats"]["payload_bytes_read"] == 0 and a["stats"]["tensor_count"] == len(a["tensors"]) > 70
    assert a["model"]["revision_sha"] in FIX.name and a["config"]["config.json"]["model_type"] == "gpt2"
    for m in (a, b):
        m["provenance"].pop("fetched_at")
    assert dump(a) == dump(b)


def test_gguf_shape_is_ne_reversed_and_names_kept():
    rec = {"kind": "gguf", "version": 3, "alignment": 32, "kv": {"general.architecture": "llama"}, "tensor_count": 1,
           "header_bytes": 1000, "bytes_read": 1000,
           "tensors": [{"name": "blk.0.attn_k.weight", "ne": [4096, 1024], "type": "Q4_K", "offset": 0}]}
    d = Discovery("o/m-GGUF", "a" * 40, [{"path": "m.Q4_K_M.gguf", "size": 10 ** 7}], False,
                  [Weight("m.Q4_K_M.gguf", ["m.Q4_K_M.gguf"], "gguf", 10 ** 7)])
    m = build_manifest(Job("o/m-GGUF", "decoder_moe", "gguf"), d, {}, {"m.Q4_K_M.gguf": rec}, "t")
    assert validate_manifest(m) == []
    t = m["tensors"][0]
    assert t["key"] == "blk.0.attn_k.weight" and t["ne"] == [4096, 1024] and t["shape"] == [1024, 4096]
    assert t["offset_begin"] == 1024 and t["nbytes"] == 4096 * 1024 // 256 * 144  # data aligned to 32; Q4_K blocks
    assert m["gguf"]["kv"]["general.architecture"] == "llama"


def test_payload_bytes_are_refused():
    rec = {"kind": "safetensors", "header": {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
           "header_bytes": 100, "bytes_read": 108}
    d = Discovery("o/m", "b" * 40, [], False, [Weight("model.safetensors", ["model.safetensors"], "safetensors", 108)])
    with pytest.raises(AssertionError):
        build_manifest(Job("o/m", "decoder_dense", "safetensors"), d, {}, {"model.safetensors": copy.deepcopy(rec)}, "t")
