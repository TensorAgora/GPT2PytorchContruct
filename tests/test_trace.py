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
