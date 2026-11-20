"""python -m tools.conformance DESCRIPTOR... | --all

Per-descriptor conformance (master §8.3, design C1-C6), offline from tests/fixtures:
  C1  binds the real header fixture named in its provenance (`fixture`), every stored tensor in a class
  C3  its HF scheme names map through gguf-py to the MODEL_TENSOR of each role (registry/namemaps.py)
  C4  contraction count and FLOPs equal its golden (`provenance.golden`, rebuilt from tests/fixtures/spelled), or the
      closed-form count of the decoder template (validated on the llama-family goldens) for the fixture's config
  C5  hostile input fails cleanly: half the tensors gone, one shape wrong, an absurd layer count
  C6  two builds are byte-identical
Prints one line per descriptor; n/a when a check has no subject (no fixture, not a decoder). Exits 1 on any FAIL.
"""
import copy
import json
import sys
from pathlib import Path

from gpt2_pytorch.hub.fixtures import FixtureClient, find
from gpt2_pytorch.hub.manifest import dump, read_manifest
from gpt2_pytorch.registry.build import build
from gpt2_pytorch.registry.descriptor import DescriptorError, hyper_from_config, ids, load
from gpt2_pytorch.registry.hyper import resolve_hyper
from gpt2_pytorch.registry.namemaps import c3, gguf_scheme, split_roles
from gpt2_pytorch.registry.resolve import resolve_and_bind

SPELLED = Path("tests/fixtures/spelled")


def decoder_counts(dims: dict, layers: int) -> dict:
    """Closed-form contractions and FLOPs of the gqa + swiglu decoder (equal to the qwen2.5, mistral, phi-3.5 and gemma-2 goldens)."""
    B, T, D, Dq, Dkv, Dh, F, V, Hkv = (dims[k] for k in ("B", "T", "D", "Dq", "Dkv", "Dh", "F", "V", "Hkv"))
    Hq = dims["H"] * Hkv
    per_layer = B * T * (2 * D * Dq + 2 * D * Dkv) + B * T * Dh * Dh * (Hq + Hkv) + 2 * B * Hq * T * T * Dh + 3 * B * T * D * F
    return {"contractions": 11 * layers + 1, "contraction_flops": 2 * (layers * per_layer + B * T * D * V)}


def _with_gguf(d: dict) -> dict:
    return {**d, "schemes": {**d["schemes"], "gguf": gguf_scheme(d["schemes"]["hf"])}}


def _blocks(d: dict) -> set:
    """Every block name a descriptor's layer selectors can choose."""
    out = set()
    for sel in d["recipe"].get("layers", {}).values():
        out |= {sel[k] for k in ("uniform", "first", "then", "on", "off", "default") if k in sel}
        out |= set((sel.get("map") or {}).values()) | set(sel.get("index_sets") or {}) | set(sel.get("by_presence") or {})
    return out


def _fixture(d):
    repo = d["provenance"].get("fixture")
    folder = find(repo) if repo else None
    if not folder:
        return None, None, None
    c = FixtureClient(folder)
    job = c.job()
    return job, read_manifest(job, c, fetched_at="t")[1], c


def check(did: str) -> dict:
    d, out = load(did), {}
    job, m, _ = _fixture(d)
    scheme = "gguf" if m and m["model"]["format"] == "gguf" else "hf"
    # C1
    if m is None:
        out["C1"] = "n/a"
    else:
        r = resolve_and_bind(m, job.row, only=did)
        out["C1"] = "PASS" if r.binding and r.binding.ok else f"FAIL {r.near_misses[:1]}"
    # C3
    bad3 = c3(d["schemes"].get("hf", {}), skip=split_roles(did))
    out["C3"] = "PASS" if not bad3 else f"FAIL {bad3[:2]}"
    # C4
    golden = d["provenance"].get("golden")
    if golden and (SPELLED / f"{Path(golden).stem}.config.json").exists():
        cfg = json.loads((SPELLED / f"{Path(golden).stem}.config.json").read_text())
        sch = d["provenance"].get("golden_scheme", "hf")
        g, _ = build(_with_gguf(d) if sch == "gguf" else d, hyper_from_config(d, cfg), sch, cfg=cfg)
        want = json.loads(Path(golden).read_text())["totals"]
        keys = ("contractions", "contraction_flops", "edges", "ops")
        out["C4"] = "PASS" if all(g["totals"][k] == want[k] for k in keys) else f"FAIL {[(k, g['totals'][k], want[k]) for k in keys]}"
    elif m is not None and d["recipe"]["template"] == "decoder" and _blocks(d) <= {"gqa", "gqa_local", "swiglu"}:
        h = resolve_hyper(d, m, job.row)
        g, _ = build(_with_gguf(d) if scheme == "gguf" else d, h, scheme, cfg=m["config"].get("config.json") or {})
        want = decoder_counts(g["dims"], h["L"][0])
        out["C4"] = "PASS" if all(g["totals"][k] == v for k, v in want.items()) else f"FAIL {want} vs {g['totals']}"
    else:
        out["C4"] = "n/a"
    # C5
    if m is None:
        out["C5"] = "n/a"
    else:
        bad = []
        half = copy.deepcopy(m)
        half["tensors"] = half["tensors"][: len(half["tensors"]) // 2]
        shape = copy.deepcopy(m)
        big = max(shape["tensors"], key=lambda t: t["numel"])  # a parameter, never a sliced buffer
        big["shape"] = [big["shape"][0] + 1] + big["shape"][1:]
        huge = copy.deepcopy(m)
        cfg = huge["config"].setdefault("config.json", {}) or {}
        for c in [cfg] + [v for v in cfg.values() if isinstance(v, dict)]:  # nested text_config too
            for k in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
                if k in c:
                    c[k] = 10 ** 7
        if "gguf" in huge and huge["gguf"]:
            kv = huge["gguf"]["kv"]
            kv[f"{kv.get('general.architecture')}.block_count"] = 10 ** 7
        for name, mm in (("truncated", half), ("misshaped", shape), ("huge", huge)):
            try:
                r = resolve_and_bind(mm, job.row, only=did)
                if r.binding and r.binding.ok:
                    bad.append(f"{name} bound")
            except (DescriptorError, Exception) as e:  # noqa: BLE001 - a hostile input must fail as a near-miss, not raise
                bad.append(f"{name} raised {type(e).__name__}")
        out["C5"] = "PASS" if not bad else f"FAIL {bad}"
    # C6
    cfg = (m["config"].get("config.json") if m else None) or (json.loads((SPELLED / f"{Path(golden).stem}.config.json").read_text()) if golden else None)
    if cfg is None and m is None:
        out["C6"] = "n/a"
    else:
        h = resolve_hyper(d, m, job.row) if m else hyper_from_config(d, cfg)
        dd = _with_gguf(d) if scheme == "gguf" else d
        a, b = (dump(build(dd, h, scheme, cfg=cfg or {})[0]) for _ in range(2))
        out["C6"] = "PASS" if a == b else "FAIL two builds differ"
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    todo = ids() if argv == ["--all"] else argv
    failed = 0
    for did in todo:
        r = check(did)
        print(did, " ".join(f"{k} {v}" for k, v in r.items()))
        failed += any(v.startswith("FAIL") for v in r.values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
