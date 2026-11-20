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
    components: list | None = None  # path B diffusers: the other pipeline folders
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
    return res if only else _path_b(manifest, row, name, cfg, cands, res)  # a forced descriptor never falls back


def _trace_descriptor(lib: str, cls: str) -> dict:
    import importlib.metadata as md
    version = md.version({"timm": "timm", "diffusers": "diffusers", "transformers": "transformers"}[lib])
    return {"id": f"trace_{lib}_{cls}", "version": version, "match": {}, "hyperparameters": {}, "schemes": {"hf": {}},
            "recipe": {"template": "trace"}, "ties": {}, "ignored": ["*.num_batches_tracked", "*.position_ids"], "components": [],
            "provenance": {"certainty": "traced", "path": "B", "ported_from": f"{lib} {version} class {cls}, torch.export on meta"}}


def _path_b(manifest, row, name, cfg, cands, res):
    """Path B (06f): when no descriptor binds, trace a library class the files name; the denoiser first for diffusers."""
    from .trace import TraceError, trace, transformers_has
    libs = [c for c in cands if c.library_class and c.library_class.split(":", 1)[0] in ("timm", "diffusers")]
    libs.sort(key=lambda c: (not any(w in c.library_class for w in ("UNet", "Transformer")), c.library_class))
    mt = cfg.get("model_type")
    if not libs and transformers_has(mt) and not needs_remote_code(cfg, row):
        libs.append(Candidate(None, "R2", f"model_type={mt} (transformers class)", 0.4, f"transformers:{mt}"))
    small = manifest.get("config") or {}
    for c in libs:
        lib, cls = c.library_class.split(":", 1)
        sub, ccfg, comps = manifest, cfg, None
        if lib == "diffusers":
            path = next((p for p, v in small.items() if isinstance(v, dict) and v.get("_class_name") == cls and "/" in p), None)
            if path is None:
                continue  # a pipeline class names no weights of its own
            folder = path.rsplit("/", 1)[0] + "/"
            ccfg = small[path]
            sub = {**manifest, "tensors": [t for t in manifest["tensors"] if t["file"].startswith(folder)]}
            comps = []
            for f in sorted({t["file"].split("/", 1)[0] for t in manifest["tensors"] if "/" in t["file"]} - {folder.rstrip("/")}):
                ts = [t for t in manifest["tensors"] if t["file"].startswith(f + "/")]
                comps.append({"name": f, "state": "declared_without_graph", "tensors": len(ts), "params": sum(t["numel"] for t in ts)})
        d = _trace_descriptor(lib, cls)
        miss = {"descriptor": f"{d['id']}@{d['version']}", "rung": c.rung, "evidence": c.evidence}
        stored = {t["key"] for t in sub["tensors"]}
        # transformers: the saved class first, then a causal LM, then the bare model (a head the checkpoint lacks)
        for kind in (("arch", "auto", "base") if lib == "transformers" else ("auto",)):
            try:
                graph, expected, info = trace(lib, cls, ccfg, name, kind)
            except (TraceError, Exception) as e:  # noqa: BLE001 - a class that cannot be built or exported is a near-miss
                res.near_misses.append(miss | {"reason": f"path B: {type(e).__name__}: {e}"[:300]})
                continue
            expected, fix = _align_prefix(graph, expected, stored)
            aliases = _ties(graph, expected, info["ties"], fix, stored)
            dropped = [k for k in stored if any(re.search(rx, k) for rx in info["ignore_re"])]
            d = {**d, "ignored": d["ignored"] + dropped + [k for u in info["unused"] if (k := fix(u)) in stored and k not in aliases]}
            binding = verify(expected, sub, d, {**(graph.get("aliases") or {}), **aliases})
            if not binding.ok:
                res.near_misses.append(miss | {"reason": f"path B ({kind}) binding failed", **binding.near_miss()})
                continue
            res.status, res.candidate, res.descriptor, res.graph, res.expected, res.binding = "ok", c, d, graph, expected, binding
            res.components = comps
            return res
    return res


def _ties(graph: dict, expected: list, ties: dict, fix, stored: set) -> dict:
    """A tied parameter binds to whichever of its names is stored; its other stored names are `tied` copies.
    Edits expected (and the graph's checkpoint keys) in place; returns {stored copy: parameter name}."""
    aliases, keyed = {}, {t["name"]: t for t in graph["tensors"]}
    for e in expected:
        names = [fix(n) for n in ties.get(e["state_dict_key"], [])] if "state_dict_key" in e else []
        here = [n for n in names if n in stored]
        if not here:
            continue
        if e["checkpoint_key"] not in stored:
            e["checkpoint_key"] = keyed[e["name"]]["checkpoint_key"] = here[0]
        aliases.update({n: e["name"] for n in here if n != e["checkpoint_key"]})
    return aliases


def _align_prefix(graph: dict, expected: list, stored: set) -> list:
    """Checkpoints saved from a task wrapper store `bert.embeddings...` where the bare class says `embeddings...` (and
    the other way round): one prefix, added or stripped, that makes the most expected keys stored. (expected, fix)"""
    keys = [e["checkpoint_key"] for e in expected]
    hit = sum(k in stored for k in keys)
    best, fix = hit, None
    for k in keys[:20]:
        if k not in stored:
            pre = next((s[: -len(k)] for s in stored if s.endswith("." + k)), None)
            for f in ((lambda x, pre=pre: pre + x) if pre else None, lambda x: x.split(".", 1)[-1]):
                if f and (n := sum(f(x) in stored for x in keys)) > best:
                    best, fix = n, f
            break
    if fix is None:
        return expected, lambda k: k
    by = {k: fix(k) if fix(k) in stored else k for k in keys}
    for t in graph["tensors"]:
        if t.get("checkpoint_key") in by:
            t["checkpoint_key"] = by[t["checkpoint_key"]]
    return [e | {"checkpoint_key": by[e["checkpoint_key"]]} for e in expected], fix


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
