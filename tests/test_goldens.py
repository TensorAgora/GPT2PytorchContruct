"""Master §8.1 / R-11: each golden repo, read from its pinned header fixture through the whole pipeline, rebuilds
structurally equal to json/<golden>.json; a remaining difference is allowed only when EXPECTED names it with its cause
(.plan/hf-graph-pipeline/08a-goldens-pinned/DESIGN.md)."""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.catalog import Job  # noqa: E402
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import load_json  # noqa: E402
from gpt2_pytorch.hub.pipeline import load_graph, run_one  # noqa: E402
from tools.graph_check import check, compare  # noqa: E402

GOLDENS = list(csv.DictReader(open("csv/goldens.csv")))
EXPECTED = {
    # the arch.mjs golden names FP8 scales only inside its expert views; the pipeline names every stored form (§5.3)
    "deepseek-v3": {"status": "partial", "ignore": ("view",)},
    "qwen3.8-27b-fp8": {"status": "partial"},
    # the checkpoint stores fused qkv_proj / gate_up_proj; the overlay binds q/k/v and gate/up as row slices of them,
    # so those parameters name the fused key and a slice view where the golden names separate keys
    "phi-3.5-mini-instruct": {"ignore": ("checkpoint_key", "view")},
}


@pytest.mark.parametrize("row", GOLDENS, ids=lambda r: Path(r["golden"]).stem)
def test_golden_from_pinned_header(row, tmp_path):
    name = Path(row["golden"]).stem
    fx = find(row["repo"])
    if fx is None:
        pytest.skip(f"{row['repo']}: no header fixture (gated: needs HF_TOKEN with accepted terms)")
    assert fx.name.endswith("@" + row["revision"]), f"fixture {fx.name} is not at the pinned revision"
    want = EXPECTED.get(name, {})
    c = FixtureClient(fx)
    r = run_one(Job(row["repo"], row["category"], row["format"], row), tmp_path, c, row["revision"], fetched_at="t")
    assert r.status == want.get("status", "ok")
    if r.status == "no_graph":
        rep = json.loads((r.dir / "report.json").read_text())
        assert rep["near_misses"], "a no_graph golden must say why"
        return
    g = load_graph(r.dir)
    assert check(g, load_json(r.dir / "manifest.json")) == []
    assert compare(g, json.loads(Path(row["golden"]).read_text()), want.get("ignore", ())) == []
