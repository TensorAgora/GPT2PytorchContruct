"""S12 batch, ledger and coverage (master R-14, R-15, §5.4).

The ledger output/categories/ledger.jsonl is append-only: one line per finished row {repo, sha, category, format, stage,
status, code, duration_s, header_bytes, requests}. A restart skips every repo that already has a terminal line (no
request is made for it); `refresh` re-discovers and re-runs only repos whose sha changed. Coverage files are derived
from the ledger's latest line per repo plus each report.json.
"""
import csv
import json
import os
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .http import redact

TERMINAL = ("ok", "ok_assumed", "partial", "no_graph", "skipped:", "failed:")
# re-run on restart: outages are not facts about the repo
TRANSIENT = ("failed:http_0", "failed:http_429", "failed:http_5", "failed:exception_")
# re-run when the registry changed: a new descriptor may now resolve them
REGISTRY_BOUND = ("no_graph", "ok_assumed", "partial", "skipped:remote_code")  # partial: a better graph may exist (D-20)


def needs_run(rec: dict | None, registry: str) -> bool:
    if rec is None or not rec["status"].startswith(TERMINAL) or rec["status"].startswith(TRANSIENT):
        return True
    return rec["status"] in REGISTRY_BOUND and rec.get("registry") != registry


def prune_graphs(out) -> int:
    """Delete shared graphs no graph.ref points at (a descriptor change leaves the old ones behind)."""
    out = Path(out)
    live = {json.loads(p.read_text())["content_key"] for p in out.rglob("graph.ref")}
    dead = [p for p in (out / "graphs").glob("*.json*") if p.name.split(".")[0] not in live]
    for p in dead:
        p.unlink()
    return len(dead)


def stage_of(status: str) -> str:
    if status.startswith("skipped:") and status != "skipped:remote_code":
        return "discover"
    if status.startswith("failed:"):
        return "headers" if not status.startswith("failed:http") else "discover"
    return "resolve" if status in ("no_graph", "skipped:remote_code") else "write"


def read_ledger(out) -> dict:
    """{repo: latest ledger line}."""
    path = Path(out) / "ledger.jsonl"
    latest = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # the last line of a killed run
                    continue
                latest[rec["repo"]] = rec
    return latest


class Ledger:
    def __init__(self, out):
        self.path = Path(out) / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, rec: dict) -> None:
        line = redact(json.dumps(rec, sort_keys=True, separators=(",", ":"))) + "\n"
        with self.lock, open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def run_all(jobs, out, run, concurrency=4, limit=None, refresh=False, sha_of=None, log=print, registry=None) -> list[dict]:
    """Run `run(job) -> Result` over jobs that need it (needs_run, or a changed sha with refresh). Returns the new
    ledger lines."""
    from ..registry.descriptor import registry_version

    out, registry = Path(out), registry or registry_version()
    done = read_ledger(out)
    todo = [j for j in jobs if needs_run(done.get(j.repo_id), registry)
            or (refresh and sha_of and done[j.repo_id].get("sha") and sha_of(j) != done[j.repo_id]["sha"])]
    todo = todo[:limit] if limit is not None else todo
    ledger, new = Ledger(out), []

    def one(job):
        t0 = time.monotonic()
        try:
            r = run(job)
            status = r.status
            rep = json.loads((r.dir / "report.json").read_text()) if r.dir and (r.dir / "report.json").exists() else {}
        except Exception as e:  # one bad repo never stops the batch
            status, rep = f"failed:exception_{type(e).__name__}", {"message": redact(str(e))[:300]}
        rec = {"repo": job.repo_id, "sha": rep.get("sha"), "category": job.category, "format": job.format, "registry": registry,
               "stage": stage_of(status), "status": status, "code": status.split(":", 1)[1] if ":" in status else None,
               "duration_s": round(time.monotonic() - t0, 3),
               "header_bytes": (rep.get("bytes") or {}).get("header_bytes_read", 0), "requests": (rep.get("bytes") or {}).get("requests", 0)}
        ledger.append(rec)
        new.append(rec)
        if len(new) % 50 == 0:
            log(f"{len(new)}/{len(todo)} rows")
        return rec

    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        list(pool.map(one, todo))
    return new


def throughput(records: list[dict], wall_s: float) -> dict:
    hb = sorted(r["header_bytes"] for r in records if r["header_bytes"])
    q = lambda p: hb[min(len(hb) - 1, int(p * len(hb)))] if hb else 0  # noqa: E731
    return {"rows": len(records), "wall_s": round(wall_s, 1), "rows_per_min": round(60 * len(records) / wall_s, 1) if wall_s else None,
            "requests_per_row": round(statistics.mean(r["requests"] for r in records), 2) if records else 0,
            "header_bytes_p50": q(0.5), "header_bytes_p95": q(0.95), "header_bytes_max": hb[-1] if hb else 0}


# ---- coverage ----------------------------------------------------------------------------------------------------
def catalog_key(row: dict) -> str:
    for col in ("model_type", "gguf_architecture", "backbone_class", "timm_module"):
        if row.get(col):
            return row[col]
    return "(unknown)"


def fetched_key(report: dict, row: dict) -> str:
    k = report.get("keys") or {}
    lib = k.get("library_class") or []
    return k.get("model_type") or k.get("gguf_architecture") or (lib[-1] if lib else None) or catalog_key(row)


def _report(out: Path, rec: dict) -> dict:
    p = out / rec["category"] / rec["format"] / rec["repo"].replace("/", "__") / "report.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def coverage(out, jobs, readme=False) -> dict:
    """coverage.csv, coverage_model_type.csv, near_misses.csv, descriptor_backlog.csv, category_keys.csv; with readme,
    <category>/README.md (status counts, top keys, top near-misses)."""
    out = Path(out)
    latest = read_ledger(out)
    cats = defaultdict(lambda: (Counter(), Counter(), Counter()))  # category -> statuses, (key, status), near-misses
    rows = {j.repo_id: j.row for j in jobs}
    by_cfs, by_key = Counter(), Counter()
    near, backlog, keys_seen = defaultdict(lambda: [0, ""]), Counter(), {}
    for repo, rec in sorted(latest.items()):
        rep = _report(out, rec) if rec["status"] in ("no_graph", "ok", "ok_assumed", "partial") else {}
        key = fetched_key(rep, rows.get(repo, {}))
        keys_seen[repo] = key
        by_cfs[(rec["category"], rec["format"], rec["status"])] += 1
        by_key[(key, rec["status"])] += 1
        cats[rec["category"]][0][rec["status"]] += 1
        cats[rec["category"]][1][(key, rec["status"].split(":")[0])] += 1
        for m in rep.get("near_misses", []) if rec["status"] == "no_graph" else []:
            cats[rec["category"]][2][f"{key}: {m['descriptor']}: {m['reason'][:80]}"] += 1
        if rec["status"] == "no_graph":
            backlog[(key, rec["category"])] += 1
            for m in rep.get("near_misses", []):
                cell = near[(key, m["descriptor"], m["reason"][:80])]
                cell[0] += 1
                cell[1] = cell[1] or "; ".join(k for _, k in (m.get("missing") or [])[:3])
    unresolved_catalog = Counter()
    for j in jobs:  # prediction for rows not yet resolved: their catalog key (or the fetched one when known)
        rec = latest.get(j.repo_id)
        if rec is None or rec["status"] == "no_graph":
            unresolved_catalog[keys_seen.get(j.repo_id) or catalog_key(j.row)] += 1
    _csv(out / "coverage.csv", ["category", "format", "status", "rows"], [[*k, v] for k, v in sorted(by_cfs.items())])
    _csv(out / "coverage_model_type.csv", ["model_type", "status", "rows"], [[*k, v] for k, v in sorted(by_key.items())])
    _csv(out / "near_misses.csv", ["key", "descriptor", "reason", "missing_example", "rows_unlockable"],
         sorted(([*k, v[1], v[0]] for k, v in near.items()), key=lambda r: (-r[4], r[0])))
    _csv(out / "descriptor_backlog.csv", ["key", "category", "rows", "rows_unlocked", "catalog_rows"],
         sorted(([k, c, n, n, unresolved_catalog.get(k, 0)] for (k, c), n in backlog.items()), key=lambda r: (-r[4], -r[3], r[0])))
    per_cat = defaultdict(Counter)
    for j in jobs:
        if j.format in ("safetensors", "gguf") and j.category != "classic_nlp":
            per_cat[j.category][keys_seen.get(j.repo_id) or catalog_key(j.row)] += 1
    _csv(out / "category_keys.csv", ["category", "rows_in_scope", "distinct_keys", "top_keys"],
         sorted(([c, sum(k.values()), len(k), "; ".join(f"{a} {n}" for a, n in k.most_common(8))] for c, k in per_cat.items()),
                key=lambda r: -r[1]))
    for c, (st, ks, nm) in cats.items() if readme else ():
        _readme(out / c / "README.md", c, st, ks, nm)
    return {"ledger_rows": len(latest), "coverage_total": sum(by_cfs.values()), "no_graph": sum(backlog.values())}


def _readme(path: Path, category: str, statuses: Counter, keys: Counter, near: Counter):
    total = sum(statuses.values())
    per_key = defaultdict(Counter)
    for (k, s), n in keys.items():
        per_key[k][s] += n
    top = sorted(per_key.items(), key=lambda kv: -sum(kv[1].values()))[:15]
    lines = [f"# {category}", "", f"{total} rows (coverage.csv). Generated by `python -m tools.hf_graph --coverage --readme`.", "",
             "| status | rows |", "| --- | --: |", *[f"| {s} | {n} |" for s, n in statuses.most_common()], "",
             "## Top keys", "", "| key | rows | ok | partial | no_graph | failed | skipped |", "| --- | --: | --: | --: | --: | --: | --: |",
             *[f"| {k} | {sum(c.values())} | {c['ok']} | {c['partial']} | {c['no_graph']} | {c['failed']} | {c['skipped']} |" for k, c in top], "",
             "## Top near-misses", "", *([f"- {n} × {m}" for m, n in near.most_common(10)] or ["none"]), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
