"""One repo end to end (master §5, S1-S12): discover, small files, headers, manifest, resolution, hyperparameters,
binding, build, and the outputs of §5.4:

  <out>/<category>/<format>/<org>__<repo>/manifest.json   Observed facts
  <out>/<category>/<format>/<org>__<repo>/report.json     status, candidates, near-misses, binding, hyper and dim sources
  <out>/<category>/<format>/<org>__<repo>/graph.ref       {content_key, model, overrides, checkpoint, provenance, components}
  <out>/graphs/<content-key>.json                         the shared graph (sha256 of its own canonical bytes, D9)

load_graph(repo_dir) materializes graph.ref over its shared graph. All files are canonical JSON; only the manifest's
provenance.fetched_at changes between runs.
"""
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Job
from .headers import HeaderError
from .http import Client, HttpError
from .manifest import dump, read_manifest

REPORT_FORMAT = "tensormorph-hf-graph-report"
CERTAINTY = {"R1": None, "R2": None, "R3": "assumed", "R4": "assumed"}


@dataclass
class Result:
    status: str
    dir: Path | None = None
    paths: list = field(default_factory=list)
    content_key: str | None = None


def repo_dir(out: Path, job: Job) -> Path:
    return Path(out) / job.category / job.format / job.repo_id.replace("/", "__")


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump(obj))
    return path


def prefix_tree(keys, depth=3) -> dict:
    """{prefix: count} over dotted names with layer indices collapsed to N, `depth` segments deep."""
    out = Counter()
    for k in keys:
        parts = ["N" if p.isdigit() else p for p in k.split(".")]
        for n in range(1, min(depth, len(parts)) + 1):
            out[".".join(parts[:n])] += 1
    return dict(sorted(out.items()))


def _components(d: dict, manifest: dict) -> list[dict]:
    out = []
    for c in d.get("components", []):
        ts = [t for t in manifest["tensors"] if t["key"].startswith(c["prefix"])]
        if ts:
            out.append({"name": c["name"], "state": c.get("state", "declared_without_graph"), "tensors": len(ts),
                        "params": sum(t["numel"] for t in ts)})
    return out


def keys_of(manifest: dict) -> dict:
    """The architecture keys the fetched files state (master §2: they correct the catalog's)."""
    from ..registry.hyper import NESTED
    small = manifest.get("config") or {}
    cfg = small.get("config.json") or {}
    nested = [cfg[n] for n in NESTED if isinstance(cfg.get(n), dict)]
    classes = sorted({c["_class_name"] for c in small.values() if isinstance(c, dict) and c.get("_class_name")})
    return {"model_type": cfg.get("model_type") or next((c.get("model_type") for c in nested if c.get("model_type")), None),
            "architectures": cfg.get("architectures") or next((c.get("architectures") for c in nested if c.get("architectures")), None),
            "gguf_architecture": ((manifest.get("gguf") or {}).get("kv") or {}).get("general.architecture"),
            "library_class": classes or ([f"timm:{cfg['architecture']}"] if cfg.get("architecture") and "pretrained_cfg" in cfg else None)}


def load_graph(folder) -> dict:
    """The full graph of a repo folder: graph.json, or graph.ref over its shared graph."""
    folder = Path(folder)
    if (folder / "graph.json").exists():
        return json.loads((folder / "graph.json").read_text())
    ref = json.loads((folder / "graph.ref").read_text())
    root = next(p for p in [folder, *folder.parents] if (p / "graphs").is_dir())
    g = json.loads((root / "graphs" / f"{ref['content_key']}.json").read_text())
    g["model"] = ref["model"]
    g["groups"]["label"] = ref["model"]["name"]
    tensors = {t["name"]: t for t in g["tensors"]}
    for name, fields in ref["overrides"].items():
        tensors[name].update(fields)
    for k in ("checkpoint", "components"):
        if ref.get(k):
            g[k] = ref[k]
    g["provenance"] = {**g.get("provenance", {}), **ref["provenance"]}
    return g


def run_one(job: Job, out, client=None, revision: str | None = None, fetched_at: str | None = None) -> Result:
    from ..registry.hyper import dim_sources
    from ..registry.resolve import resolve_and_bind

    client = client or Client()
    client = client.scoped() if hasattr(client, "scoped") else client  # per-job counters (FixtureClient has none)
    out, folder = Path(out), repo_dir(out, job)
    report = {"format": REPORT_FORMAT, "version": 1, "repo": job.repo_id, "category": job.category, "weights_format": job.format}
    requests0 = client.requests
    try:
        d, manifest = read_manifest(job, client, revision, fetched_at)
    except HeaderError as e:
        return _finish(folder, report | {"status": f"failed:{e.code}", "message": str(e)}, [])
    except HttpError as e:
        return _finish(folder, report | {"status": f"failed:http_{e.status}", "message": str(e)}, [])
    report["sha"] = d.sha
    if manifest is None:
        return _finish(folder, report | {"status": d.status}, [])
    paths = [_write(folder / "manifest.json", manifest)]
    report["bytes"] = {"header_bytes_read": manifest["stats"]["header_bytes_read"], "payload_bytes_read": 0,
                       "requests": client.requests - requests0}
    report["keys"] = keys_of(manifest)
    r = resolve_and_bind(manifest, job.row, job.repo_id.split("/")[-1].lower())
    report["candidates"] = [{"descriptor": c.descriptor, "library_class": c.library_class, "rung": c.rung,
                             "evidence": c.evidence, "score": c.score} for c in r.candidates]
    report["near_misses"] = r.near_misses
    if r.graph is None:
        keys = [t["key"] for t in manifest["tensors"]]
        report["inventory"] = {"tensor_count": len(keys), "params_by_dtype": manifest["stats"]["params_by_dtype"],
                               "prefix_tree": prefix_tree(keys)}
        return _finish(folder, report | {"status": r.status}, paths)

    d_id = f"{r.descriptor['id']}@{r.descriptor['version']}"
    b, g = r.binding, r.graph
    components = _components(r.descriptor, manifest)
    status = "partial" if r.status == "ok" and any(c["state"] == "declared_without_graph" for c in components) else r.status
    shared = copy.deepcopy(g)
    shared["model"] = {"name": r.descriptor["id"]}
    shared["groups"]["label"] = r.descriptor["id"]  # viz_graph labels the root group with the model name
    shared["provenance"] = {"certainty": r.descriptor["provenance"].get("certainty", "template"),
                            "evidence": f"descriptor {d_id} over resolved hyperparameters", "origin": "gpt2_pytorch.registry",
                            "descriptor": d_id, "path": "A"}
    raw = dump(shared)
    key = hashlib.sha256(raw).hexdigest()[:24]
    gpath = out / "graphs" / f"{key}.json"
    if not gpath.exists():
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gpath.write_bytes(raw)
    hyper_sources = {k: v[1] for k, v in r.hyper.items()}
    ref = {"content_key": key, "model": {"name": job.repo_id.split("/")[-1].lower()},
           "overrides": {name: {"view": v} for name, v in sorted(b.views.items())},
           "checkpoint": b.checkpoint(manifest), "components": components,
           "provenance": {"certainty": CERTAINTY[r.candidate.rung] or shared["provenance"]["certainty"],
                          "evidence": r.candidate.evidence, "rung": r.candidate.rung, "hyper_sources": hyper_sources}}
    paths.append(_write(folder / "graph.ref", ref))
    report |= {"status": status, "content_key": key, "graph": str(gpath.relative_to(out)),
               "winner": {"descriptor": d_id, "rung": r.candidate.rung, "evidence": r.candidate.evidence, "path": "A"},
               "hyper": {k: {"value": v, "source": s} for k, (v, s) in r.hyper.items()},
               "dims": {k: {"value": g["dims"][k], "source": s} for k, s in dim_sources(r.descriptor, r.hyper, g).items()},
               "binding": {c: len(v) for c, v in b.partition.items()} | {"unclaimed": len(b.unclaimed)},
               "components": components,
               "sidecars": [{"tensor": t, "principal": t.removesuffix("_scale_inv"), "scheme": "fp8_block"} for t in b.partition.get("aux", [])],
               "totals": {"ops": g["totals"]["ops"], "contraction_flops": g["totals"]["contraction_flops"],
                          "params": sum(math.prod(e["shape"]) for e in r.expected if e["role"] == "parameter")}}
    return _finish(folder, report, paths, key)


def _finish(folder: Path, report: dict, paths: list, key=None) -> Result:
    paths.append(_write(folder / "report.json", report))
    return Result(report["status"], folder, paths, key)
