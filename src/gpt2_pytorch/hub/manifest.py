"""S4 tensor manifest v1 (master §5.2): Observed facts only — what the headers and small files state, nothing that
depends on a chosen architecture. Canonical JSON; `provenance.fetched_at` is the only volatile field."""
import datetime
import json
import math
from functools import cache
from pathlib import Path

TOOL_VERSION = "hf-graph 0.1"
SCHEMA = Path(__file__).resolve().parents[1] / "schemas/tensor-manifest-v1.schema.json"
ENDPOINT = "https://huggingface.co"


def dump(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _folder(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _kind(path: str) -> str:
    if path.endswith(".gguf"):
        return "gguf"
    if path.endswith(".safetensors"):
        return "safetensors"
    if path.endswith(".safetensors.index.json"):
        return "index"
    return "model_index" if path == "model_index.json" else "config"


def _gguf_tensors(part: str, rec: dict) -> list[dict]:
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    align = rec["alignment"]
    data_start = -(-rec["header_bytes"] // align) * align
    out = []
    for t in rec["tensors"]:
        numel = math.prod(t["ne"])
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[t["type"]]]
        nbytes = numel // block * size
        begin = data_start + t["offset"]
        out.append({"key": t["name"], "file": part, "dtype": t["type"], "shape": t["ne"][::-1], "ne": t["ne"],
                    "numel": numel, "nbytes": nbytes, "offset_begin": begin, "offset_end": begin + nbytes})
    return out


def _safetensors_tensors(part: str, rec: dict) -> list[dict]:
    base = rec["header_bytes"]
    return [{"key": k, "file": part, "dtype": v["dtype"], "shape": v["shape"], "numel": math.prod(v["shape"]),
             "nbytes": v["data_offsets"][1] - v["data_offsets"][0],
             "offset_begin": base + v["data_offsets"][0], "offset_end": base + v["data_offsets"][1]}
            for k, v in rec["header"].items() if k != "__metadata__"]


def build_manifest(job, discovery, small: dict, headers: dict, fetched_at: str | None = None) -> dict:
    """job: catalog Job; discovery: discover.Discovery; small: {path: JSON}; headers: {part: header record}."""
    d, row = discovery, job.row
    size = {f["path"]: f["size"] for f in d.files}
    url = lambda p: f"{ENDPOINT}/{d.repo}/resolve/{d.sha}/{p}"  # noqa: E731
    indexes = {p: set(c.get("weight_map", {}).values()) for p, c in small.items() if p.endswith(".safetensors.index.json") and c}
    diagnostics, tensors, files = [], [], []
    for w in d.weights:
        for part in w.parts:
            rec = headers[part]
            shard_of = next((p for p, shards in indexes.items() if _folder(p) == _folder(part) and _name(part) in shards), None)
            if w.kind == "gguf" and len(w.parts) > 1:
                shard_of = w.path
            files.append({"path": part, "size": size.get(part, 0), "kind": w.kind, "header_bytes": rec["header_bytes"], "shard_of": shard_of})
            tensors += _gguf_tensors(part, rec) if rec["kind"] == "gguf" else _safetensors_tensors(part, rec)
    files += [{"path": p, "size": size.get(p, 0), "kind": _kind(p), "header_bytes": None, "shard_of": None} for p in small]
    files.sort(key=lambda f: f["path"])
    tensors.sort(key=lambda t: (t["file"], t["key"]))

    seen = {}
    for t in tensors:
        if t["key"] in seen:
            diagnostics.append({"code": "duplicate_tensor", "message": f"also in {seen[t['key']]}", "tensor": t["key"]})
        seen.setdefault(t["key"], t["file"])
    for p in indexes:
        mapped = set(small[p].get("weight_map", {}))
        stored = {t["key"] for t in tensors if _folder(t["file"]) == _folder(p)}
        if mapped != stored:
            diagnostics.append({"code": "index_mismatch", "message": f"{p}: weight_map {len(mapped)} keys, headers {len(stored)}"})

    gguf = next((headers[p] for w in d.weights if w.kind == "gguf" for p in w.parts[:1]), None)
    params = {}
    for t in tensors:
        params[t["dtype"]] = params.get(t["dtype"], 0) + t["numel"]
    payload = sum(max(0, r["bytes_read"] - r["header_bytes"]) for r in headers.values())
    assert payload == 0, f"{d.repo}: {payload} payload bytes read"
    metadata = {p: r["header"]["__metadata__"] for p, r in headers.items() if r["kind"] == "safetensors" and "__metadata__" in r["header"]}
    return {
        "format": "tensormorph-tensor-manifest", "version": 1,
        "model": {"id": d.repo, "url": f"{ENDPOINT}/{d.repo}", "revision_sha": d.sha, "category": job.category,
                  "format": d.weights[0].kind, "license": row.get("license") or None, "gated": d.gated,
                  "library_name": d.info.get("library_name") or row.get("library_name") or None,
                  "base_model": row.get("base_model") or None, "catalog_row": dict(row)},
        "files": files, "config": small,
        "gguf": {"version": gguf["version"], "kv": gguf["kv"], "alignment": gguf["alignment"]} if gguf else None,
        "tensors": tensors, "safetensors_metadata": metadata,
        "stats": {"tensor_count": len(tensors), "params_by_dtype": params, "weight_bytes": sum(t["nbytes"] for t in tensors),
                  "header_bytes_read": sum(r["bytes_read"] for r in headers.values()), "payload_bytes_read": payload},
        "provenance": {"fetched_at": fetched_at or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                       "tool_version": TOOL_VERSION, "urls": sorted(url(f["path"]) for f in files),
                       "catalog_sources": [s for s in (row.get("sources") or "").split("|") if s]},
        "diagnostics": diagnostics,
    }


@cache
def _validator():
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def validate_manifest(m: dict) -> list[str]:
    errs = [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in _validator().iter_errors(m)]
    if not errs:
        if m["stats"]["tensor_count"] != len(m["tensors"]):
            errs.append("stats.tensor_count != len(tensors)")
        if [(t["file"], t["key"]) for t in m["tensors"]] != sorted((t["file"], t["key"]) for t in m["tensors"]):
            errs.append("tensors not sorted by (file, key)")
    return errs


def read_manifest(job, client, revision=None, fetched_at=None):
    """S1-S4 for one job: (discovery, manifest), manifest None when discovery ended in a skipped:/failed: status."""
    from .discover import discover, small_files
    from .headers import read_header

    d = discover(job, client, revision)
    if d.status:
        return d, None
    small = small_files(d, client)
    headers = {p: read_header(client, d.repo, d.sha, p) for w in d.weights for p in w.parts}
    return d, build_manifest(job, d, small, headers, fetched_at)
