"""python -m tools.tel_coverage GRAPH... [--const-as-buffer] [--csv OUT]

Counts, per viz-graph, the ops T♭ can spell and each construct docs/GRAPH_TEL.md "Exclusions" lists (§9.4): routed MoE
(the ops of a module that reads a runtime top-k buffer, and of its submodules, or `execution.kind = routed`), unrolled
recurrence (`execution.kind = scan`, or a GatedDeltaNet / Mamba module in graphs older than the annotation),
tensor-valued constants, B/T axes without a symbol, and opaque (path-B) ops. An op is spellable when it falls in none of them. With --const-as-buffer, constants
count as bound buffers and stop excluding ops.
"""
import argparse
import csv
import json

from gpt2_pytorch.hub.manifest import load_json
import sys
from pathlib import Path

RECURRENT_MODULES = ("GatedDeltaNet", "Mamba", "MambaMixer")
FIELDS = ["graph", "ops", "spellable", "routed_moe", "recurrence", "list_constants", "ops_on_list_constants",
          "concrete_axes", "opaque"]


def coverage(g: dict, const_as_buffer: bool = False) -> dict:
    tensors = {t["name"]: t for t in g["tensors"]}
    runtime = {n for n, t in tensors.items() if t["role"] == "buffer" and not t["checkpoint_key"] and t["view"] and "runtime" in t["view"]}
    lists = {n for n, t in tensors.items() if t["role"] == "constant" and isinstance(t["value"], list)}
    # the MoE blocks: modules that read the router's runtime top-k choice; their ops and their submodules' are routed
    moe = {op["module"] for op in g["ops"] if any(i in runtime for i in op["inputs"])}
    counts = dict.fromkeys(FIELDS[2:], 0)
    for op in g["ops"]:
        ex = op.get("execution") or {}
        routed = ex.get("kind") == "routed" or any(op["module"] == m or op["module"].startswith(m + ".") for m in moe)
        recur = ex.get("kind") == "scan" or op["module_type"] in RECURRENT_MODULES
        on_list = not const_as_buffer and any(i in lists for i in op["inputs"])
        opaque = op["primitive"] == "Opaque"
        counts["routed_moe"] += routed
        counts["recurrence"] += recur
        counts["ops_on_list_constants"] += on_list
        counts["opaque"] += opaque
        counts["spellable"] += not (routed or recur or on_list or opaque)
    counts["list_constants"] = len(lists)
    counts["concrete_axes"] = sum(1 for a in g["axes"] if a["name"] in ("B", "T") and not a["symbol"])
    return {"graph": g["model"]["name"], "ops": len(g["ops"]), **counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--const-as-buffer", action="store_true")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args(argv)
    paths = [p for g in a.graphs for p in (sorted([*g.glob("*.json"), *g.glob("*.json.gz")]) if g.is_dir() else [g])]
    rows = [coverage(load_json(p), a.const_as_buffer) for p in paths]
    w = csv.DictWriter(open(a.csv, "w", newline="") if a.csv else sys.stdout, FIELDS)
    w.writeheader()
    w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
