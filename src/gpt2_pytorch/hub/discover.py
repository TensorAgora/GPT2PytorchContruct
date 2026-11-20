"""S1 discover and S2 small files (master §5.1): pinned sha, file list with sizes, gated, and which weight files to read.

python -m gpt2_pytorch.hub.discover --smoke N [--concurrency K]   # N jobs round-robin over category x format
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .catalog import Job, load_jobs
from .http import Client, HttpError, allowed, token

DEFERRED_FORMATS = ("pytorch", "onnx", "other")
PART = re.compile(r"^(?P<base>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")


@dataclass
class Weight:
    path: str  # logical name: the file, or for multi-part GGUF `<base>.gguf`
    parts: list[str]
    kind: str  # safetensors | gguf
    size: int
    component: str = ""  # diffusers component folder, else ""


@dataclass
class Discovery:
    repo: str
    sha: str | None = None
    files: list[dict] = field(default_factory=list)  # [{path, size}]
    gated: bool | str = False
    weights: list[Weight] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)  # files declared but not read (mmproj GGUF)
    info: dict = field(default_factory=dict)  # library_name, pipeline_tag, tags
    status: str | None = None  # a terminal skipped:/failed: status, else None


def _folder(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _variant(path: str) -> bool:
    """`x.fp16.safetensors`, `x.ema.safetensors.index.json`: a dot left in the stem marks a variant."""
    stem = path.rsplit("/", 1)[-1].removesuffix(".index.json").removesuffix(".safetensors")
    return "." in stem


def _safetensors(files: dict, client: Client, repo: str, sha: str):
    """Index shards; else every root *.safetensors; diffusers: per component folder, plain file preferred over a
    variant. Returns (weights, variants)."""
    diffusers = "model_index.json" in files
    folders = sorted({_folder(p) for p in files if p.endswith(".safetensors")})
    components = [f for f in folders if f and "/" not in f and f"{f}/config.json" in files]
    folders = components if diffusers or "" not in folders else [""]  # no model_index.json and no root file: component folders
    weights, variants = [], []
    for folder in folders:
        tensors = sorted(p for p in files if _folder(p) == folder and p.endswith(".safetensors"))
        indexes = sorted(p for p in files if _folder(p) == folder and p.endswith(".safetensors.index.json"))
        if indexes:
            pick = min(indexes, key=lambda p: (_variant(p), p))
            shards = sorted(set(json.loads(client.get_small(repo, sha, pick))["weight_map"].values()))
            chosen = [f"{folder}/{x}" if folder else x for x in shards]
            variants += [p for p in indexes if p != pick]
        elif folder == "":
            chosen = tensors
        else:
            chosen = [p for p in tensors if not _variant(p)] or tensors[:1]
        weights += [Weight(p, [p], "safetensors", files.get(p, 0), folder) for p in chosen]
        variants += [p for p in tensors if p not in chosen]
    if diffusers:
        variants += sorted(p for p in files if p.endswith(".safetensors") and _folder(p) not in folders)
    return weights, variants


def _gguf(files: dict):
    """One representative quant (Q4_K_M, else the smallest); multi-part files are one logical file (D2)."""
    groups, components = {}, []
    for p in sorted(files):
        if not p.endswith(".gguf"):
            continue
        if "mmproj" in p.lower():
            components.append(p)
            continue
        m = PART.match(p)
        groups.setdefault(m["base"] + ".gguf" if m else p, []).append(p)
    if not groups:
        return [], [], components
    size = {k: sum(files[p] for p in v) for k, v in groups.items()}
    q4 = [k for k in groups if "q4_k_m" in k.lower()]
    pick = min(q4 or groups, key=lambda k: (size[k], k))
    return [Weight(pick, groups[pick], "gguf", size[pick])], sorted(k for k in groups if k != pick), components


def discover(job: Job, client: Client, revision: str | None = None, refresh: bool = False) -> Discovery:
    d = Discovery(job.repo_id)
    if job.format in DEFERRED_FORMATS or job.category == "classic_nlp":
        d.status = "skipped:deferred_format"
        return d
    try:
        info = client.model_info(job.repo_id, revision or (job.row.get("revision") or None), refresh=refresh)
    except HttpError as e:
        d.status = f"failed:http_{e.status}"
        return d
    d.sha, d.gated = info["sha"], info.get("gated") or False
    d.info = {k: info.get(k) for k in ("library_name", "pipeline_tag", "tags")}
    files = {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}
    d.files = [{"path": p, "size": s} for p, s in sorted(files.items())]
    try:
        st, st_variants = _safetensors(files, client, job.repo_id, d.sha)
    except (HttpError, KeyError, ValueError) as e:
        d.status = f"failed:index_{getattr(e, 'status', type(e).__name__)}"
        return d
    gg, gg_variants, d.components = _gguf(files)
    if job.format == "gguf" and gg or not st:
        d.weights, d.variants = gg, gg_variants + [w.path for w in st]
    else:
        d.weights, d.variants = st, st_variants + [w.path for w in gg] + gg_variants
    if not d.weights:
        d.status = "skipped:no_weights"
    elif d.gated:
        if not token():
            d.status = "skipped:gated"
        else:
            try:
                client.get_range(client.file_url(job.repo_id, d.sha, d.weights[0].parts[0]), 0, 7)
            except HttpError as e:
                if e.status in (401, 403):
                    d.status = "skipped:gated_not_accepted"
                else:
                    raise
    return d


def small_files(d: Discovery, client: Client) -> dict:
    """S2: the §3 small files this repo has, parsed: {path: JSON}. Index files come from the discover cache."""
    have = {f["path"] for f in d.files}
    wanted = {"config.json", "params.json", "model_index.json"}
    wanted |= {f"{w.component}/config.json" for w in d.weights if w.component}
    wanted |= {p for p in have if p.endswith(".safetensors.index.json") and _folder(p) in {w.component for w in d.weights}}
    out = {}
    for p in sorted(wanted & have):
        if allowed(p):
            try:
                out[p] = json.loads(client.get_small(d.repo, d.sha, p))
            except ValueError:
                out[p] = None
    return out


def smoke_jobs(n: int) -> list[Job]:
    groups = {}
    for j in load_jobs():
        groups.setdefault((j.category, j.format), []).append(j)
    keys, picked, i = sorted(groups), [], 0
    while len(picked) < n and any(i < len(groups[k]) for k in keys):
        picked += [groups[k][i] for k in keys if i < len(groups[k])]
        i += 1
    return picked[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args(argv)
    client = Client(concurrency=a.concurrency)

    def one(job):
        try:
            d = discover(job, client)
            if d.status is None:
                small_files(d, client)
            return job, d.sha, d.status or "discovered"
        except HttpError as e:
            return job, None, f"failed:http_{e.status}"

    with ThreadPoolExecutor(a.concurrency) as pool:
        results = list(pool.map(one, smoke_jobs(a.smoke)))
    for job, sha, status in results:
        print(f"{job.category}/{job.format}\t{job.repo_id}\t{sha or '-'}\t{status}")
    print(dict(Counter(s for *_, s in results)))
    print(f"rows: {len(results)}, with sha or terminal skipped/failed: {sum(bool(sha) or s.startswith(('skipped:', 'failed:')) for _, sha, s in results)}")
    print(f"network requests: {client.requests}")


if __name__ == "__main__":
    main()
