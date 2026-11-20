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
from gpt2_pytorch.registry.descriptor import DescriptorError, hyper_from_config, ids, layer_blocks, load
from gpt2_pytorch.registry.hyper import resolve_hyper
from gpt2_pytorch.registry.namemaps import c3, gguf_scheme, split_roles
from gpt2_pytorch.registry.resolve import resolve_and_bind

SPELLED = Path("tests/fixtures/spelled")


def block_counts(block: str, d: dict) -> tuple[int, int]:
    """(contractions, MACs) one layer block adds at the example shape (B, T); validated on the 8 goldens' totals."""
    B, T, D = d["B"], d["T"], d["D"]
    BT = B * T
    if block in ("gqa", "gqa_local"):
        Hkv, Dh = d["Hkv"], d["Dh"]
        Hq = d["H"] * Hkv
        return 8, BT * (2 * D * d["Dq"] + 2 * D * d["Dkv"]) + BT * Dh * Dh * (Hq + Hkv) + 2 * B * Hq * T * T * Dh
    if block in ("mla", "mla_direct"):
        H, Dn, Dr, Dv = d["H"], d["Dn"], d["Dr"], d["Dv"]
        q = BT * D * d["QH"] if block == "mla_direct" else BT * (D * d["QR"] + d["QR"] * d["QH"])
        rest = BT * D * d["KA"] + BT * d["KR"] * d["KH"] + B * H * T * Dr * Dr + B * T * Dr * Dr \
            + B * H * T * T * (Dn + Dr + Dv) + BT * d["HV"] * D
        return (9 if block == "mla_direct" else 10), q + rest
    if block == "gqa_norope":
        Hkv, Dh = d["Hkv"], d["Dh"]
        Hq = d["H"] * Hkv
        return 6, BT * (2 * D * d["Dq"] + 2 * D * d["Dkv"]) + 2 * B * Hq * T * T * Dh
    if block == "lfm2_conv":  # in_proj, conv taps over (k, Kc), out_proj
        return 3, BT * 3 * D * D + B * D * d["Kc"] * T * T + BT * D * D
    if block == "mamba":  # in, taps, x_proj, dt_proj, out + per step dt, x, B, C, dBx, y, place (+ dtA after step 0)
        Di, N = d["Di"], d["N"]
        steps = T * (2 * B * T * Di + 2 * B * T * N + 2 * B * Di * N + B * Di * T) + (T - 1) * B * Di * N
        return 5 + 7 * T + (T - 1), BT * D * 2 * Di + B * Di * d["Kc"] * T * T + BT * Di * d["XP"] + BT * d["R"] * Di + BT * Di * D + steps
    if block == "swiglu":
        return 3, 3 * BT * D * d["F"]
    if block in ("moe_mixtral", "moe_qwen"):
        return 4, BT * D * d["E"] + 3 * BT * d["E"] * d["Fe"] * D
    if block == "moe_deepseek":
        return 7, BT * D * d["E"] + 3 * BT * d["E"] * d["Fe"] * D + 3 * BT * D * d.get("Fs", d["Fe"])
    raise KeyError(block)


def decoder_counts(d: dict, layers: list[tuple[str, str]]) -> dict:
    """Totals of the decoder template: every layer's (attention, mlp) blocks plus the head."""
    n, macs = 1, d["B"] * d["T"] * d["D"] * d["V"]
    for blocks in layers:
        for blk in blocks:
            c, m = block_counts(blk, d)
            n, macs = n + c, macs + m
    return {"contractions": n, "contraction_flops": 2 * macs}


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
    text = d["provenance"].get("text", "").split("@")[0]  # an assembly is judged by its text model's own names
    base = load(text) if text else d
    if not base["match"].get("gguf_architecture"):
        out["C3"] = "n/a"  # no GGUF form of its own (mistral converts to llama; distilbert has none)
    else:
        bad3 = c3(base["schemes"].get("hf", {}), skip=split_roles(text or did))
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
        if d["recipe"]["template"] == "decoder" and out["C4"] == "PASS":  # the closed form must reproduce the golden as well
            slots = {s: layer_blocks(sel, cfg, int(hyper_from_config(d, cfg)["L"][0])) for s, sel in d["recipe"]["layers"].items()}
            closed = decoder_counts(g["dims"], list(zip(slots["attention"], slots["mlp"])))
            if any(want[k] != v for k, v in closed.items()):
                out["C4"] = f"FAIL closed form {closed} vs golden"
    elif m is not None and d["recipe"]["template"] == "encoder":
        h = resolve_hyper(d, m, job.row)
        g, _ = build(d, h, scheme, cfg=m["config"].get("config.json") or {})
        dd, L = g["dims"], int(h["L"][0])
        BT = dd["B"] * dd["T"]
        layer = BT * 4 * dd["D"] * dd["Dq"] + 2 * dd["B"] * dd["H"] * dd["T"] * dd["T"] * dd["Dh"] + 2 * BT * dd["D"] * dd["F"]
        want = {"contractions": 8 * L, "contraction_flops": 2 * L * layer}  # q, k, v, o, scores, context, intermediate, output
        out["C4"] = "PASS" if all(g["totals"][k] == v for k, v in want.items()) else f"FAIL {want} vs {g['totals']}"
    elif m is not None and d["recipe"]["template"] == "seq2seq":
        h = resolve_hyper(d, m, job.row)
        cfg = m["config"].get("config.json") or {}
        g, _ = build(d, h, scheme, cfg=cfg)
        dd, Le, Ld = g["dims"], int(h["L"][0]), int(h["Ld"][0])
        B, T, D, Dq, F = dd["B"], dd["T"], dd["D"], dd["Dq"], dd["F"]
        nf = 3 if d["recipe"]["flavour"] == "t5" and cfg.get("feed_forward_proj", "relu").startswith("gated") else 2
        att = 4 * B * T * D * Dq + 2 * B * dd["H"] * T * T * dd["Dh"]  # projections + scores + context
        want = {"contractions": Le * (6 + nf) + Ld * (12 + nf) + 1,
                "contraction_flops": 2 * (Le * (att + nf * B * T * D * F) + Ld * (2 * att + nf * B * T * D * F) + B * T * D * dd["V"])}
        out["C4"] = "PASS" if all(g["totals"][k] == v for k, v in want.items()) else f"FAIL {want} vs {g['totals']}"
    elif m is not None and d["recipe"]["template"] == "decoder":
        h = resolve_hyper(d, m, job.row)
        cfg = m["config"].get("config.json") or {}
        g, _ = build(_with_gguf(d) if scheme == "gguf" else d, h, scheme, cfg=cfg)
        names = [t["key"] for t in m["tensors"]]
        slots = {s: layer_blocks(sel, cfg, h["L"][0], names) for s, sel in d["recipe"]["layers"].items()}
        want = decoder_counts(g["dims"], list(zip(slots["attention"], slots["mlp"])))
        out["C4"] = "PASS" if all(g["totals"][k] == v for k, v in want.items()) else f"FAIL {want} vs {g['totals']}"
    else:
        out["C4"] = "n/a"
    # C5
    if m is None:
        out["C5"] = "n/a"
    else:
        bad = []
        half = copy.deepcopy(m)
        half["tensors"] = half["tensors"][::2]  # every other tensor: hits every part of the model
        shape = copy.deepcopy(m)
        big = max(shape["tensors"], key=lambda t: t["numel"])  # a parameter, never a sliced buffer
        big["shape"] = [big["shape"][0] + 1] + big["shape"][1:]
        huge = copy.deepcopy(m)
        cfg = huge["config"].setdefault("config.json", {}) or {}
        layer_keys = [k for k in d["hyperparameters"].get("L", []) if not k.startswith(("=", "gguf:", "shape:", "catalog:", "missing:"))]
        for c in [cfg] + [v for v in cfg.values() if isinstance(v, dict)]:  # the descriptor's own layer-count keys, nested too
            for k in layer_keys + ["num_hidden_layers"]:
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
