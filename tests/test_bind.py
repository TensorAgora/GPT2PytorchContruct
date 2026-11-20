import copy
from functools import cache

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import read_manifest  # noqa: E402
from gpt2_pytorch.registry.bind import stack_members, verify  # noqa: E402
from gpt2_pytorch.registry.build import build  # noqa: E402
from gpt2_pytorch.registry.descriptor import load  # noqa: E402
from gpt2_pytorch.registry.hyper import resolve_hyper  # noqa: E402

pytestmark = pytest.mark.skipif(find("distilbert/distilgpt2") is None, reason="fixture missing")


@cache
def setup():
    c = FixtureClient(find("distilbert/distilgpt2"))
    m = read_manifest(c.job(), c, fetched_at="t")[1]
    d = load("gpt2")
    g, expected = build(d, resolve_hyper(d, m))
    return m, d, g, expected


def test_distilgpt2_binds_fully():
    m, d, g, expected = setup()
    b = verify(expected, m, d, g["aliases"])
    assert b.ok and len(b.partition["bound"]) == 82


def test_removed_misshaped_and_extra_tensors_are_named():
    m, d, g, expected = setup()
    bad = copy.deepcopy(m)
    bad["tensors"] = [t for t in bad["tensors"] if t["key"] != "transformer.h.0.attn.c_attn.weight"]
    next(t for t in bad["tensors"] if t["key"] == "transformer.wte.weight")["shape"] = [50000, 768]
    bad["tensors"].append({"key": "foo.weight", "shape": [1], "dtype": "F32"})
    b = verify(expected, bad, d, g["aliases"])
    assert not b.ok
    assert [k for _, k in b.missing] == ["transformer.h.0.attn.c_attn.weight"]
    assert [x[1] for x in b.misshaped] == ["transformer.wte.weight"] and b.unclaimed == ["foo.weight"]


def test_stack_views_and_fp8_scales():
    assert stack_members("stack of model.layers.3.mlp.experts.{0…2}.up_proj.weight (FP8, 128 × 128 block scales)") == \
        [f"model.layers.3.mlp.experts.{j}.up_proj.weight" for j in range(3)]
    d = {"ignored": [], "components": [{"prefix": "mtp."}]}
    expected = [{"name": "W", "role": "parameter", "shape": [256, 256], "checkpoint_key": "w.weight", "view": None},
                {"name": "S", "role": "parameter", "shape": [2, 4, 8], "checkpoint_key": None, "view": "stack of e.{0…1}.weight"}]
    m = {"tensors": [{"key": "w.weight", "shape": [256, 256], "dtype": "F8_E4M3"}, {"key": "w.weight_scale_inv", "shape": [2, 2], "dtype": "F32"},
                     {"key": "e.0.weight", "shape": [4, 8], "dtype": "BF16"}, {"key": "e.1.weight", "shape": [4, 8], "dtype": "BF16"},
                     {"key": "mtp.x", "shape": [1], "dtype": "BF16"}]}
    b = verify(expected, m, d)
    assert b.ok and b.partition["aux"] == ["w.weight_scale_inv"] and b.partition["component"] == ["mtp.x"]
    assert b.views == {"W": "dequantized: F8_E4M3 x w.weight_scale_inv [2, 2] (block-wise)"}
