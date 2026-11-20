import json
from functools import cache
from pathlib import Path

from gpt2_pytorch.registry.build import build
from gpt2_pytorch.registry.descriptor import hyper_from_config, load


@cache
def golden(name):
    return json.loads(Path("json", f"{name}.json").read_text())


def layer_ops(g, i, root):
    t = {x["name"]: x for x in g["tensors"]}
    return [(o["primitive"], o["fn"], o["out"], t[o["out"]]["shape"]) for o in g["ops"] if o["module"].startswith(f"{root}.layers.{i}.") or o["module"] == f"{root}.layers.{i}"]


def test_llama_layer_from_blocks_matches_golden_layer0():
    cfg = json.loads(Path("tests/fixtures/spelled/qwen2.5-7b-instruct.config.json").read_text()) | {"num_hidden_layers": 1}
    d = load("qwen2")
    g, expected = build(d, hyper_from_config(d, cfg), cfg=cfg)
    assert layer_ops(g, 0, "qwen2") == layer_ops(golden("qwen2.5-7b-instruct"), 0, "qwen2")
    assert {a["name"]: a["symbol"] for a in g["axes"]}["T"] == "T"  # R-52
    assert {e["checkpoint_key"] for e in expected} >= {"model.layers.0.self_attn.q_proj.bias", "lm_head.weight"}
