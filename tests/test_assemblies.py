"""Assemblies (design doc §7.3): the text tower binds, every other stored tensor is a component, nothing is dropped."""
import json

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.pipeline import load_graph, run_one  # noqa: E402


@pytest.mark.parametrize("repo, components", [
    ("Qwen/Qwen3.8-27B-FP8", {"visual": 333, "mtp": 22}),
    ("Qwen/Qwen3-VL-2B-Instruct", {"visual": 315}),
    ("unsloth/gemma-3-4b-it", {"vision_tower": 437, "multi_modal_projector": 2}),
    ("Qwen/Qwen2.5-VL-3B-Instruct", {"visual": 390}),
])
def test_assembly_classes_every_tensor(repo, components, tmp_path):
    fx = find(repo)
    if fx is None:
        pytest.skip(f"{repo} fixture missing")
    c = FixtureClient(fx)
    r = run_one(c.job(), tmp_path, c, fetched_at="t")
    rep = json.loads((r.dir / "report.json").read_text())
    assert r.status == "partial" and rep["binding"]["unclaimed"] == 0
    assert {x["name"]: x["tensors"] for x in rep["components"]} == components
    assert all(x["state"] == "declared_without_graph" for x in load_graph(r.dir)["components"])
