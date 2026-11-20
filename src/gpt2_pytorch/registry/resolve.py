"""S6 architecture resolution (master R-08, design doc §6): rungs R1 (GGUF general.architecture), R2 (architectures,
then model_type, also in nested text/llm/language/thinker configs; diffusers and timm classes), R3 (tensor-name
fingerprint) and R4 (name hint) only propose candidates; the binding verifier decides. Repository code never runs.

python -m gpt2_pytorch.registry.resolve tests/fixtures/headers/<org>__<repo>@<sha>
"""
import re
import sys
from dataclasses import dataclass, field

from .bind import verify
from .build import build
from .descriptor import DescriptorError, ids, load
from .hyper import NESTED, HyperError, dim_sources, resolve_hyper
from .namemaps import apply_overlay, gguf_scheme

RANK = {"R1": 0, "R2": 1, "R3": 2, "R4": 3}


@dataclass
class Candidate:
    descriptor: str | None
    rung: str
    evidence: str
    score: float
    library_class: str | None = None


@dataclass
class Resolution:
    status: str
    candidate: Candidate | None = None
    descriptor: dict | None = None
    hyper: dict = field(default_factory=dict)
    graph: dict | None = None
    expected: list = field(default_factory=list)
    binding: object = None
    near_misses: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)  # param -> {checkpoint_key?, view} the binding sets
    transforms: list = field(default_factory=list)  # storage transforms of the overlay that apply


def config_of(manifest: dict) -> dict:
    return (manifest.get("config") or {}).get("config.json") or {}


def needs_remote_code(cfg: dict, row: dict) -> bool:
    flags = (row.get("graph_flags") or "").split("|")
    return "auto_map" in cfg or "trust_remote_code" in flags or str(row.get("trust_remote_code", "")).lower() in ("true", "1")


def _stems(names) -> set:
    return {re.sub(r"\.\d+\.", ".N.", n) for n in names}


def candidates(manifest: dict, row: dict | None = None) -> tuple[list[Candidate], str | None]:
    """Ranked candidates and a terminal status (skipped:remote_code) when nothing may be built."""
    row = row or {}
    regs = {i: load(i) for i in ids()}
    cfg = config_of(manifest)
    kv = (manifest.get("gguf") or {}).get("kv") or {}
    out = []
    if arch := kv.get("general.architecture"):
        out += [Candidate(i, "R1", f"general.architecture={arch}", 1.0) for i, d in regs.items() if arch in d["match"].get("gguf_architecture", [])]
    configs = [("", cfg)] + [(f"{n}.", cfg[n]) for n in NESTED if isinstance(cfg.get(n), dict)]
    for where, c in configs:
        for a in c.get("architectures") or []:
            out += [Candidate(i, "R2", f"{where}architectures={a}", 0.9) for i, d in regs.items() if a in d["match"].get("architectures", [])]
    by_arch = {c.descriptor for c in out if "architectures=" in c.evidence}
    for where, c in configs:
        if mt := c.get("model_type"):
            out += [Candidate(i, "R2", f"{where}model_type={mt}", 0.8) for i, d in regs.items() if mt in d["match"].get("model_type", [])]
    small = manifest.get("config") or {}
    for path, c in sorted(small.items()):
        if isinstance(c, dict) and c.get("_class_name"):
            out.append(Candidate(None, "R2", f"{path} _class_name", 0.7, f"diffusers:{c['_class_name']}"))
    if cfg.get("architecture") and "pretrained_cfg" in cfg:
        out.append(Candidate(None, "R2", "timm config.json architecture", 0.7, f"timm:{cfg['architecture']}"))
    for col, lib in (("backbone_class", "diffusers"), ("timm_module", "timm")):
        if row.get(col):
            out.append(Candidate(None, "R2", f"catalog {col}", 0.5, f"{lib}:{row[col]}"))
    if needs_remote_code(cfg, row):
        out = [c for c in out if c.descriptor in by_arch and "architectures=" in c.evidence]
        if not out:
            return [], "skipped:remote_code"
    if not any(c.descriptor for c in out):
        names = _stems(t["key"] for t in manifest.get("tensors", []))
        scheme = "gguf" if (manifest.get("model") or {}).get("format") == "gguf" else "hf"
        for i, d in regs.items():
            keys = {v.replace("{i}", "N").replace("{what}", "weight") for v in d["schemes"].get(scheme, {}).values() if isinstance(v, str)}
            if len(keys) >= 4 and (share := len(keys & names) / len(keys)) >= 0.8:
                out.append(Candidate(i, "R3", f"{share:.0%} of {scheme} names present", share))
        repo = re.sub(r"[^a-z0-9]", "", ((manifest.get("model") or {}).get("id") or "").lower())
        for i, d in regs.items():
            hits = [mt for mt in d["match"].get("model_type", []) if re.sub(r"[^a-z0-9]", "", mt) in repo]
            if hits and not any(c.descriptor == i for c in out):
                out.append(Candidate(i, "R4", f"repo name contains {hits[0]}", 0.3))
    seen, ranked = set(), []
    for c in sorted(out, key=lambda c: (RANK[c.rung], -c.score)):
        k = c.descriptor or c.library_class
        if k not in seen:
            seen.add(k)
            ranked.append(c)
    return ranked, None


def resolve_and_bind(manifest: dict, row: dict | None = None, name: str | None = None, only: str | None = None) -> Resolution:
    """`only` binds that one descriptor, skipping the ladder (conformance C1)."""
    row = row or {}
    cands, status = ([Candidate(only, "R2", "forced (conformance)", 1.0)], None) if only else candidates(manifest, row)
    res = Resolution(status or "no_graph", candidates=cands)
    if status:
        return res
    scheme = "gguf" if manifest["model"]["format"] == "gguf" else "hf"
    arch = ((manifest.get("gguf") or {}).get("kv") or {}).get("general.architecture")
    cfg = config_of(manifest)
    names = [t["key"] for t in manifest["tensors"]]
    name = name or manifest["model"]["id"].split("/")[-1].lower()
    for c in cands:
        if not c.descriptor:
            continue
        d = load(c.descriptor)
        if scheme == "gguf" and d["recipe"]["template"] == "decoder" and "gguf" not in d["schemes"]:
            d = {**d, "schemes": {**d["schemes"], "gguf": gguf_scheme(d["schemes"]["hf"])}}  # generated (06d)
        miss = {"descriptor": f"{d['id']}@{d['version']}", "rung": c.rung, "evidence": c.evidence}
        if scheme not in d["schemes"]:
            res.near_misses.append(miss | {"reason": f"no {scheme} naming scheme"})
            continue
        try:
            hyper = resolve_hyper(d, manifest, row)
            graph, expected = build(d, hyper, scheme, name, cfg, names)
        except (HyperError, DescriptorError, KeyError, ValueError, AssertionError, ZeroDivisionError) as e:
            res.near_misses.append(miss | {"reason": f"{type(e).__name__}: {e}"[:300]})
            continue
        fmt = {k: v for k, (v, _) in hyper.items()} | graph["dims"]
        expected, overrides, transforms = apply_overlay(expected, manifest, d, scheme, d["schemes"][scheme], fmt, arch)
        binding = verify(expected, manifest, d, graph.get("aliases"), fmt)
        if not binding.ok:
            res.near_misses.append(miss | {"reason": "binding failed", **binding.near_miss()})
            continue
        res.status = "ok" if c.rung in ("R1", "R2") else "ok_assumed"
        res.candidate, res.descriptor, res.hyper, res.graph, res.expected, res.binding = c, d, hyper, graph, expected, binding
        res.overrides = {**overrides, **{n: {**overrides.get(n, {}), "view": v} for n, v in binding.views.items()}}
        res.transforms = transforms
        return res
    return res


def main(argv=None):
    from ..hub.fixtures import FixtureClient
    from ..hub.manifest import read_manifest

    folder = (argv or sys.argv[1:])[0]
    client = FixtureClient(folder)
    job = client.job()
    _, manifest = read_manifest(job, client)
    r = resolve_and_bind(manifest, job.row)
    print(f"{job.repo_id}: {r.status}" + (f" via {r.descriptor['id']}@{r.descriptor['version']} ({r.candidate.rung}: {r.candidate.evidence})" if r.descriptor else ""))
    if r.binding:
        p = r.binding.partition
        print("binding: " + ", ".join(f"{k} {len(v)}" for k, v in p.items()) + f", unbound: {len(r.binding.unclaimed)}")
        dims = dim_sources(r.descriptor, r.hyper, r.graph)
        print(f"dims: {len(dims)}, without source: {sum(1 for v in dims.values() if not v)}")
        for k, v in r.hyper.items():
            print(f"  {k} = {v[0]!r:<12} {v[1]}")
    for m in r.near_misses:
        print(f"near-miss {m['descriptor']} ({m['rung']}): {m['reason']} {m.get('counts', '')}")


if __name__ == "__main__":
    main()
