"""Offline header fixtures (master §8.3, D-15): small JSON only, never weight bytes.

python -m gpt2_pytorch.hub.fixtures REPO [--revision SHA] [--format safetensors|gguf] [--category C]

writes tests/fixtures/headers/<org>__<repo>@<sha>/: model_info.json, the §3 small files at their repo paths,
headers/<part>.json (the header record; .json.gz when over 256 KiB) and provenance.json. FixtureClient(dir) answers
model_info, get_small and header from that folder, so discovery and the pipeline run offline.
"""
import argparse
import datetime
import gzip
import json
import struct
from pathlib import Path

from .catalog import ROOT, Job, load_jobs
from .discover import discover, small_files
from .headers import read_header
from .http import Client, NotAllowed, allowed

FIXTURES = ROOT / "tests/fixtures/headers"
GZIP_OVER = 256 << 10
INFO_KEYS = ("sha", "gated", "siblings", "library_name", "pipeline_tag", "tags")


def fixture_dir(repo: str, sha: str, root=FIXTURES) -> Path:
    return Path(root) / f"{repo.replace('/', '__')}@{sha}"


def find(repo: str, root=FIXTURES) -> Path | None:
    hits = sorted(Path(root).glob(f"{repo.replace('/', '__')}@*"))
    return hits[-1] if hits else None


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    if len(raw) > GZIP_OVER:
        path.with_name(path.name + ".gz").write_bytes(gzip.compress(raw, mtime=0))
    else:
        path.write_bytes(raw)


def _load(path: Path):
    gz = path.with_name(path.name + ".gz")
    return json.loads(gzip.decompress(gz.read_bytes()) if gz.exists() else path.read_bytes())


def job_for(repo: str, fmt: str | None = None, category: str | None = None) -> Job:
    for j in load_jobs():
        if j.repo_id == repo:
            return Job(repo, category or j.category, fmt or j.format, j.row, j.source)
    return Job(repo, category or "unknown", fmt or ("gguf" if "gguf" in repo.lower() else "safetensors"))


def capture(job: Job, client: Client | None = None, revision: str | None = None, root=FIXTURES) -> Path:
    client = client or Client()
    d = discover(job, client, revision)
    if d.status:
        raise SystemExit(f"{job.repo_id}: {d.status}; no fixture written")
    out = fixture_dir(job.repo_id, d.sha, root)
    info = client.model_info(job.repo_id, revision)
    _dump(out / "model_info.json", {k: info.get(k) for k in INFO_KEYS})
    for path in small_files(d, client):
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(client.get_small(job.repo_id, d.sha, path))
    for w in d.weights:
        for part in w.parts:
            _dump(out / "headers" / f"{part}.json", read_header(client, job.repo_id, d.sha, part))
    _dump(out / "provenance.json", {"repo": job.repo_id, "sha": d.sha, "category": job.category, "format": job.format,
                                     "captured_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                                     "tool": "gpt2_pytorch.hub.fixtures", "requests": client.requests,
                                     "header_bytes_read": client.bytes_read})
    return out


class FixtureClient:
    """The Client interface served from a fixture folder; makes no request."""

    requests = 0
    bytes_read = 0

    def __init__(self, folder):
        self.dir = Path(folder)
        self.provenance = _load(self.dir / "provenance.json")

    def model_info(self, repo, revision=None, refresh=False):
        return _load(self.dir / "model_info.json")

    def get_small(self, repo, sha, path):
        if not allowed(path):
            raise NotAllowed(path)
        return (self.dir / path).read_bytes()

    def file_url(self, repo, sha, path):
        return f"https://huggingface.co/{repo}/resolve/{sha}/{path}"

    def header(self, part):
        return _load(self.dir / "headers" / f"{part}.json")

    def get_range(self, url, start, end):
        raise RuntimeError("a fixture serves header records, not byte ranges")

    def job(self) -> Job:
        p = self.provenance
        return Job(p["repo"], p["category"], p["format"])


def fixture_reader(folder, part):
    """get_range over safetensors header bytes rebuilt from a fixture's header record."""
    body = json.dumps(FixtureClient(folder).header(part)["header"]).encode()
    data = struct.pack("<Q", len(body)) + body
    return lambda start, end: data[start:end + 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--revision")
    ap.add_argument("--format")
    ap.add_argument("--category")
    a = ap.parse_args(argv)
    client = Client()
    out = capture(job_for(a.repo, a.format, a.category), client, a.revision)
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"{out}: {len(files)} files, {sum(p.stat().st_size for p in files)} bytes, "
          f"{client.requests} requests, {client.bytes_read} header bytes read")


if __name__ == "__main__":
    main()
