"""python -m tools.hf_graph <huggingface_url> [--revision R] [--out output/categories]
python -m tools.hf_graph --fixture tests/fixtures/headers/<org>__<repo>@<sha> [--out DIR]     (offline)
python -m tools.hf_graph --materialize <repo output dir> [--to graph.json]

Hugging Face URL -> tensor manifest -> computation graph (master R-00): reads only headers and the §3 small files, never
weights, never repository code. Writes manifest.json, report.json and graph.ref (over output/categories/graphs/<key>.json)
under <out>/<category>/<format>/<org>__<repo>/. --materialize writes the full graph of one repo folder.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from gpt2_pytorch.hub.catalog import Job
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
    a = ap.parse_args(argv)
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
