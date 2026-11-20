"""python -m tools.hf_graph <huggingface_url> [--revision R] [--out output/categories]
python -m tools.hf_graph --fixture tests/fixtures/headers/<org>__<repo>@<sha> [--out DIR]     (offline)
python -m tools.hf_graph --materialize <repo output dir> [--to graph.json]
python -m tools.hf_graph --all [--limit N] [--category C] [--concurrency K] [--refresh]   (resumable batch, ledger.jsonl)
python -m tools.hf_graph --coverage                                               (coverage, near-miss and backlog CSVs)

Hugging Face URL -> tensor manifest -> computation graph (master R-00): reads only headers and the §3 small files, never
weights, never repository code. Writes manifest.json, report.json and graph.ref (over output/categories/graphs/<key>.json)
under <out>/<category>/<format>/<org>__<repo>/. --materialize writes the full graph of one repo folder.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from gpt2_pytorch.hub.batch import coverage, prune_graphs, run_all, throughput
from gpt2_pytorch.hub.catalog import Job, load_jobs, stratified
from gpt2_pytorch.hub.http import Client
from gpt2_pytorch.hub.fixtures import FixtureClient, job_for
from gpt2_pytorch.hub.manifest import dump
from gpt2_pytorch.hub.pipeline import load_graph, run_one

OUT = Path("output/categories")


def repo_from_url(url: str) -> tuple[str, str | None]:
    m = re.fullmatch(r"(?:https?://huggingface\.co/)?([\w.-]+/[\w.-]+)(?:/(?:tree|blob|resolve)/([^/]+).*)?/?", url.strip())
    if not m:
        raise SystemExit(f"not a Hugging Face model URL: {url}")
    return m.group(1), m.group(2)


def report(result) -> None:
    print(f"{result.status}")
    for p in result.paths:
        print(f"  {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", nargs="?")
    ap.add_argument("--revision")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--materialize", type=Path)
    ap.add_argument("--to", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, help="with --all: the first N jobs of a category x format round-robin")
    ap.add_argument("--category")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--refresh", action="store_true", help="with --all: re-run repos whose sha changed")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--readme", action="store_true", help="with --coverage: also write <out>/<category>/README.md")
    a = ap.parse_args(argv)
    if a.all or a.coverage:
        jobs = load_jobs()
        if a.all:
            todo = [j for j in jobs if not a.category or j.category == a.category]
            todo = stratified(todo) if a.limit else todo
            client = Client(concurrency=2 * a.concurrency)  # requests in flight; workers = --concurrency
            t0 = time.monotonic()
            new = run_all(todo, a.out, lambda j: run_one(j, a.out, client), a.concurrency, a.limit, a.refresh,
                          lambda j: client.model_info(j.repo_id, refresh=True)["sha"])
            print(json.dumps(throughput(new, time.monotonic() - t0) | {"pruned_graphs": prune_graphs(a.out)}))
        if a.coverage:
            print(json.dumps(coverage(a.out, jobs, a.readme)))
        return 0
    if a.materialize:
        g = load_graph(a.materialize)
        to = a.to or a.materialize / "graph.full.json"
        to.write_bytes(dump(g))
        print(to)
        return 0
    if a.fixture:
        client = FixtureClient(a.fixture)
        p = client.provenance
        base = job_for(p["repo"], p["format"], p["category"])
        result = run_one(Job(base.repo_id, base.category, base.format, base.row, base.source), a.out, client, p["sha"])
    elif a.url:
        repo, rev = repo_from_url(a.url)
        result = run_one(job_for(repo), a.out, revision=a.revision or rev)
    else:
        ap.error("give a URL, --fixture or --materialize")
    report(result)
    return 0 if result.status.startswith(("ok", "partial", "no_graph", "skipped:")) else 1


if __name__ == "__main__":
    sys.exit(main())
