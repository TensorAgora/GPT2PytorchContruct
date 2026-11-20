import json
import shutil

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import load_json  # noqa: E402
from gpt2_pytorch.hub.pipeline import load_graph, run_one  # noqa: E402
from tools.graph_check import check  # noqa: E402

FIX = find("distilbert/distilgpt2")
pytestmark = pytest.mark.skipif(FIX is None, reason="distilgpt2 fixture missing")


def copy_fixture(tmp_path, repo, config_patch=None):
    """The distilgpt2 fixture under another repo id, optionally with a changed config.json."""
    dst = tmp_path / "fx" / f"{repo.replace('/', '__')}@{FIX.name.split('@')[1]}"
    shutil.copytree(FIX, dst)
    prov = json.loads((dst / "provenance.json").read_text())
    (dst / "provenance.json").write_text(json.dumps(prov | {"repo": repo}))
    if config_patch:
        cfg = json.loads((dst / "config.json").read_text())
        (dst / "config.json").write_text(json.dumps(cfg | config_patch))
    return dst


def run(folder, out):
    c = FixtureClient(folder)
    return run_one(c.job(), out, c, fetched_at="2026-01-01T00:00:00+00:00")


def test_distilgpt2_ok_and_graph_valid(tmp_path):
    r = run(FIX, tmp_path / "out")
    assert r.status == "ok"
    rep = json.loads((r.dir / "report.json").read_text())
    assert rep["winner"]["descriptor"] == "gpt2@1" and rep["bytes"]["payload_bytes_read"] == 0
    assert all(v["source"] for v in rep["dims"].values())  # R-09: every dim has a source
    g = load_graph(r.dir)
    manifest = load_json(r.dir / "manifest.json")
    assert check(g, manifest) == [] and g["provenance"]["descriptor"] == "gpt2@1" and g["checkpoint"]["bound"] == 82


def test_unknown_model_type_is_no_graph_with_prefix_tree(tmp_path):
    folder = copy_fixture(tmp_path, "o/mystery", {"model_type": "mystery", "architectures": ["MysteryModel"]})
    r = run(folder, tmp_path / "out")
    rep = json.loads((r.dir / "report.json").read_text())
    assert r.status == "no_graph" and rep["inventory"]["prefix_tree"]["transformer.h.N"] == 78  # 6 layers x 13 (attn.bias buffers included)
    assert not (r.dir / "graph.ref").exists()


def test_shared_graph(tmp_path):
    out = tmp_path / "out"
    a, b = run(FIX, out), run(copy_fixture(tmp_path, "o/distilgpt2-copy"), out)
    assert a.content_key == b.content_key and len(list((out / "graphs").glob("*.json.gz"))) == 1
    assert (a.dir / "graph.ref").exists() and (b.dir / "graph.ref").exists()
    assert load_graph(b.dir)["model"]["name"] == "distilgpt2-copy"


def test_deterministic(tmp_path):
    def files(out):
        r = run(FIX, out)
        return {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}, r
    (x, _), (y, _) = files(tmp_path / "a"), files(tmp_path / "b")
    assert x == y  # fetched_at is fixed here; nothing else varies


def test_path_b_timm_fixture(tmp_path):
    """no descriptor covers convnext: the timm class traced on meta binds every stored tensor (path B)"""
    pytest.importorskip("timm")
    fx = find("timm/test_convnext.r160_in1k")
    if fx is None:
        pytest.skip("timm fixture missing")
    c = FixtureClient(fx)
    r = run_one(c.job(), tmp_path, c, fetched_at="t")
    rep = json.loads((r.dir / "report.json").read_text())
    assert r.status == "ok" and rep["winner"]["path"] == "B" and rep["binding"]["unclaimed"] == 0
