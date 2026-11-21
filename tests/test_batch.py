import csv
import json
import shutil

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.batch import coverage, read_ledger, run_all  # noqa: E402
from gpt2_pytorch.hub.catalog import Job  # noqa: E402
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.pipeline import run_one  # noqa: E402

FIX = find("distilbert/distilgpt2")
pytestmark = pytest.mark.skipif(FIX is None, reason="distilgpt2 fixture missing")


def fixtures(tmp_path, n, unknown=()):
    """n copies of the distilgpt2 fixture as o/m0..o/m{n-1}; those in `unknown` get a model_type no descriptor knows."""
    out = {}
    for i in range(n):
        repo = f"o/m{i}"
        dst = tmp_path / "fx" / f"o__m{i}@{FIX.name.split('@')[1]}"
        shutil.copytree(FIX, dst)
        prov = json.loads((dst / "provenance.json").read_text())
        (dst / "provenance.json").write_text(json.dumps(prov | {"repo": repo}))
        if i in unknown:
            cfg = json.loads((dst / "config.json").read_text()) | {"model_type": "mystery", "architectures": ["MysteryModel"]}
            (dst / "config.json").write_text(json.dumps(cfg))
        out[repo] = dst
    return out


def runner(fx, out):
    return lambda job: run_one(job, out, FixtureClient(fx[job.repo_id]), fetched_at="t")


def jobs_for(fx):
    return [Job(r, "decoder_dense", "safetensors", {"model_type": "gpt2"}) for r in sorted(fx)]


def test_restart_skips_completed_rows(tmp_path, monkeypatch):
    secret = "hf_SECRET999"
    monkeypatch.setenv("HF_TOKEN", secret)
    fx, out = fixtures(tmp_path, 5), tmp_path / "out"
    first = run_all(jobs_for(fx), out, runner(fx, out), concurrency=2, limit=3)  # "killed" after 3 rows
    second = run_all(jobs_for(fx), out, runner(fx, out), concurrency=2)
    assert len(first) == 3 and len(second) == 2
    assert {r["repo"] for r in first} | {r["repo"] for r in second} == set(fx)
    assert run_all(jobs_for(fx), out, runner(fx, out)) == []  # nothing left: no-op
    boom = lambda job: (_ for _ in ()).throw(RuntimeError(f"bad token {secret}"))  # noqa: E731
    run_all([Job("o/x", "decoder_dense", "safetensors")], out, boom)
    text = (out / "ledger.jsonl").read_text()
    assert secret not in text and read_ledger(out)["o/x"]["status"] == "failed:exception_RuntimeError"


def test_coverage(tmp_path):
    fx, out = fixtures(tmp_path, 4, unknown={1, 3}), tmp_path / "out"
    run_all(jobs_for(fx), out, runner(fx, out))
    coverage(out, jobs_for(fx))
    ledger_rows = len(read_ledger(out))
    for name in ("coverage.csv", "coverage_model_type.csv"):
        with open(out / name) as f:
            assert sum(int(r["rows"]) for r in csv.DictReader(f)) == ledger_rows == 4


def test_backlog(tmp_path):
    fx, out = fixtures(tmp_path, 4, unknown={1, 3}), tmp_path / "out"
    run_all(jobs_for(fx), out, runner(fx, out))
    coverage(out, jobs_for(fx))
    with open(out / "descriptor_backlog.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["key"], int(r["rows_unlocked"])) for r in rows] == [("mystery", 2)]
    with open(out / "coverage.csv") as f:
        assert sum(int(r["rows"]) for r in csv.DictReader(f) if r["status"] == "no_graph") == 2


def test_registry_change_requeues_partial_but_not_ok():
    from gpt2_pytorch.hub.batch import needs_run
    assert needs_run({"status": "partial", "registry": "a"}, "b") and not needs_run({"status": "partial", "registry": "b"}, "b")
    assert not needs_run({"status": "ok", "registry": "a"}, "b")
