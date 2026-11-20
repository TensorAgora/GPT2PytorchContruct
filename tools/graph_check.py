"""python -m tools.graph_check GRAPH... [--manifest M] [--against GOLDEN] [--ignore FIELD]...

Checks tensormorph-viz-graph v1 files (master §8.2): the JSON Schema in src/gpt2_pytorch/schemas/, ops in order with
every input produced earlier (so the graph is acyclic) or a graph input/parameter/buffer/constant, edges and totals that
recompute exactly, and with --manifest every parameter's shape equal to its header tensor's shape.

--against compares each GRAPH with GOLDEN structurally (§8.1): dims, the parameter set with shapes, checkpoint_key and
view per parameter, op/edge/multiplication counts, and the golden's totals keys (additive keys in GRAPH are ignored).
--ignore FIELD (dims, shape, checkpoint_key, view, totals) skips one field, e.g. header-derived views.
Prints PASS or one line per failure; exits 1 on any.
"""
import argparse
import json
import sys
from functools import cache
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[1] / "src/gpt2_pytorch/schemas/viz-graph-v1.schema.json"
PARAM_ROLES = ("parameter", "buffer")


@cache
def _validator():
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def check(g: dict, manifest: dict | None = None, limit: int = 20) -> list[str]:
    errs = [f"schema: {'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for _, e in zip(range(limit), _validator().iter_errors(g))]
    if errs:
        return errs
    tensors = {t["name"]: t for t in g["tensors"]}
    for i, op in enumerate(g["ops"]):
        if op["index"] != i:
            errs.append(f"op {i}: index {op['index']} out of order")
        for name in op["inputs"]:
            t = tensors.get(name)
            if t is None:
                errs.append(f"op {i}: input {name!r} is not a tensor")
            elif t["produced_by"] is None:
                if t["role"] == "activation":
                    errs.append(f"op {i}: input {name!r} is an activation nobody produces")
            elif t["produced_by"] >= i:
                errs.append(f"op {i}: input {name!r} is produced later, by op {t['produced_by']} (cycle or misorder)")
        out = tensors.get(op["out"])
        if out is None or out["produced_by"] != i:
            errs.append(f"op {i}: output {op['out']!r} does not name op {i} as producer")
    mults = g["multiplications"]
    want = {"tensors": len(g["tensors"]), "ops": len(g["ops"]), "edges": sum(len(o["inputs"]) for o in g["ops"]),
            "contractions": sum(o["primitive"] == "Contraction" for o in g["ops"]),
            "contraction_flops": sum(m["flops"] for m in mults), "cost": sum(o["cost"] for o in g["ops"])}
    if len(g["edges"]) != want["edges"]:
        errs.append(f"edges: {len(g['edges'])} listed, ops read {want['edges']} inputs")
    if len(mults) != want["contractions"]:
        errs.append(f"multiplications: {len(mults)} listed, {want['contractions']} Contraction ops")
    errs += [f"totals.{k}: {g['totals'][k]} != recomputed {v}" for k, v in want.items() if g["totals"][k] != v]
    errs += [f"multiplication op {m['op']}: flops {m['flops']} != 2 x macs {m['macs']}" for m in mults if m["flops"] != 2 * m["macs"]]
    if manifest is not None:
        stored = {t["key"]: t for t in manifest["tensors"]}
        for t in g["tensors"]:
            key = t["checkpoint_key"]
            if t["role"] in PARAM_ROLES and key and not (t["view"] or "").startswith("["):  # a [..] view cuts the stored tensor
                if key not in stored:
                    errs.append(f"tensor {t['name']}: checkpoint_key {key!r} not in the manifest")
                elif list(stored[key]["shape"]) != t["shape"]:
                    errs.append(f"tensor {t['name']}: shape {t['shape']} != header {stored[key]['shape']} ({key})")
    return errs[:limit] + ([f"... {len(errs) - limit} more"] if len(errs) > limit else [])


def compare(g: dict, golden: dict, ignore=(), limit: int = 40) -> list[str]:
    diffs = []
    if "dims" not in ignore:
        for k in sorted(set(g["dims"]) | set(golden["dims"])):
            if g["dims"].get(k) != golden["dims"].get(k):
                diffs.append(f"dims.{k}: {g['dims'].get(k)} != golden {golden['dims'].get(k)}")
    params = lambda d: {t["name"]: t for t in d["tensors"] if t["role"] in PARAM_ROLES}
    mine, gold = params(g), params(golden)
    diffs += [f"parameter {n}: missing" for n in sorted(set(gold) - set(mine))]
    diffs += [f"parameter {n}: not in golden" for n in sorted(set(mine) - set(gold))]
    for n in sorted(set(mine) & set(gold)):
        for f in ("shape", "checkpoint_key", "view"):
            if f not in ignore and mine[n][f] != gold[n][f]:
                diffs.append(f"parameter {n}.{f}: {mine[n][f]!r} != golden {gold[n][f]!r}")
    for k in ("ops", "edges", "multiplications"):
        if len(g[k]) != len(golden[k]):
            diffs.append(f"{k}: {len(g[k])} != golden {len(golden[k])}")
    if "totals" not in ignore:
        diffs += [f"totals.{k}: {g['totals'].get(k)} != golden {v}" for k, v in golden["totals"].items() if g["totals"].get(k) != v]
    return diffs[:limit] + ([f"... {len(diffs) - limit} more differences"] if len(diffs) > limit else [])


def _load(path: Path):
    """JSON, or gzipped JSON (the pipeline writes manifests and shared graphs as .json.gz)."""
    import gzip
    if path.suffix != ".gz" and not path.exists():
        path = path.with_name(path.name + ".gz")
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--against", type=Path)
    ap.add_argument("--ignore", action="append", default=[], choices=["dims", "shape", "checkpoint_key", "view", "totals"])
    a = ap.parse_args(argv)
    manifest = _load(a.manifest) if a.manifest else None
    golden = _load(a.against) if a.against else None
    failed = 0
    for path in a.graphs:
        g = _load(path)
        problems = check(g, manifest)
        if golden is not None:
            if a.ignore:
                print(f"note {path}: ignoring {', '.join(a.ignore)}")
            problems += compare(g, golden, a.ignore)
        for p in problems:
            print(f"FAIL {path}: {p}")
        if not problems:
            print(f"PASS {path}" + (f" (0 differences against {a.against})" if golden is not None else ""))
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
