"""Every golden whose descriptor exists rebuilds from the config it was spelled from (tests/fixtures/spelled)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
from gpt2_pytorch.registry.build import build, dump_js  # noqa: E402
from gpt2_pytorch.registry.descriptor import hyper_from_config, ids, load  # noqa: E402
from gpt2_pytorch.registry.namemaps import gguf_scheme  # noqa: E402
from tools.graph_check import check, compare  # noqa: E402

GOLDENS = {"distilgpt2": "gpt2", "qwen2.5-7b-instruct": "qwen2", "mistral-7b-instruct-v0.3": "mistral",
           "phi-3.5-mini-instruct": "phi3", "gemma-2-2b-it": "gemma2", "mixtral-8x7b-instruct-q4km": "mixtral",
           "deepseek-v3": "deepseek_v3", "qwen3.8-27b-fp8": "qwen3_5"}
IGNORE = {"qwen3.8-27b-fp8": ("view",)}  # FP8 *_scale_inv views come from the header (D-16); full equality in 08a
SCHEME = {"mixtral-8x7b-instruct-q4km": "gguf"}


def rebuild(golden):
    d = load(GOLDENS[golden])
    if SCHEME.get(golden) == "gguf":
        d = {**d, "schemes": {**d["schemes"], "gguf": gguf_scheme(d["schemes"]["hf"])}}
    cfg = json.loads(Path("tests/fixtures/spelled", f"{golden}.config.json").read_text())
    return build(d, hyper_from_config(d, cfg), SCHEME.get(golden, "hf"), name=golden, cfg=cfg)[0]


@pytest.mark.parametrize("golden", GOLDENS)
def test_rebuilds_from_spelled_config(golden):
    if GOLDENS[golden] not in ids():
        pytest.skip(f"descriptor {GOLDENS[golden]} not written yet")
    g = rebuild(golden)
    gold = json.loads(Path("json", f"{golden}.json").read_text())
    assert check(g) == []
    assert compare(g, gold, IGNORE.get(golden, ())) == []
    if golden == "distilgpt2":
        assert dump_js(g).encode() == Path("json/distilgpt2.json").read_bytes()  # R-29
