"""Catalog rows -> jobs (master R-01): one Job per row of csv/categories/<category>/<format>.csv, plus golden repos from
csv/goldens.csv that the catalog lacks."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Job:
    repo_id: str
    category: str
    format: str
    row: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    source: str = "catalog"


def repo_id(row: dict) -> str:
    rid = row.get("huggingface_model_id") or row["huggingface_url"].split("huggingface.co/", 1)[1]
    return rid.strip("/")


def load_jobs(root=ROOT / "csv/categories", goldens=ROOT / "csv/goldens.csv") -> list[Job]:
    csv.field_size_limit(1 << 24)
    jobs = []
    for path in sorted(Path(root).glob("*/*.csv")):
        with open(path, newline="") as f:
            jobs += [Job(repo_id(r), path.parent.name, path.stem, r) for r in csv.DictReader(f)]
    jobs.sort(key=lambda j: (j.category, j.format, j.repo_id))
    seen = {j.repo_id for j in jobs}
    if goldens and Path(goldens).exists():
        with open(goldens, newline="") as f:
            jobs += [Job(r["repo"], r["category"], r["format"], r, "golden") for r in csv.DictReader(f) if r["repo"] not in seen]
    return jobs


def stratified(jobs: list[Job]) -> list[Job]:
    """Round-robin over (category, format) in sorted order, so any prefix samples every group."""
    groups = {}
    for j in jobs:
        groups.setdefault((j.category, j.format), []).append(j)
    keys, out, i = sorted(groups), [], 0
    while any(i < len(groups[k]) for k in keys):
        out += [groups[k][i] for k in keys if i < len(groups[k])]
        i += 1
    return out
