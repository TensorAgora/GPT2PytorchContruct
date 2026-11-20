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


def test_mamba_scan_on_a_tiny_jamba():
    import re

    from gpt2_pytorch.hub.fixtures import FixtureClient, find
    from gpt2_pytorch.hub.manifest import read_manifest
    from gpt2_pytorch.registry.hyper import resolve_hyper

    fx = find("ai21labs/Jamba-tiny-random")
    if fx is None:
        import pytest
        pytest.skip("Jamba fixture missing")
    c = FixtureClient(fx)
    m = read_manifest(c.job(), c, fetched_at="t")[1]
    d = load("jamba")
    g, _ = build(d, resolve_hyper(d, m), cfg=m["config"]["config.json"], names=[t["key"] for t in m["tensors"]])
    steps = [o for o in g["ops"] if o["out"].startswith("Ss_0.t")]
    per_step = {}
    for o in steps:
        per_step.setdefault(int(re.match(r"Ss_0\.t(\d+)\.", o["out"])[1]), []).append(o)
    assert sorted(per_step) == list(range(8)) and all(o["execution"] == {"kind": "scan", "axis": "T", "step": t, "body": f"{o['module']}:Ss_0"}
                                                     for t, ops in per_step.items() for o in ops)
    assert len(per_step[1]) == len(per_step[7]) == len(per_step[0]) + 6  # step 0: no dtu, dtA, dA, decayed, h, acc
    t = {x["name"]: x for x in g["tensors"]}
    assert t["Ss_0.t3.h"]["shape"] == [1, 256, 16] and t["Ym_0"]["shape"] == [1, 8, 256]
