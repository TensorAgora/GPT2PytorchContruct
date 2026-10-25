"""python -m tools.viz_graph [--program artifacts/tensor_program.json] [--out artifacts/viz_graph.json]

Turns the verified tensor program (python -m tools.tensor_program) into one JSON a graph viewer can draw directly:
every tensor and op, explicit edges and consumers, a collapsible group tree (model > module > TEL statement > op)
with each group's boundary tensors and cost, and every tensor multiplication with its batch/free/contracted axes.
It also carries what a TEL printer needs (python -m tools.viz_to_tel): symbolic axes (`axes`), the composite each op
spells out (`regions`: layernorm, softmax) and, per parameter, its PyTorch `state_dict_key` and checkpoint key.
"""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--program", default="artifacts/tensor_program.json")
parser.add_argument("--out", default="artifacts/viz_graph.json")
args = parser.parse_args()

prog = json.loads(Path(args.program).read_text())
dims = prog["dims"]
ops = prog["ops"]
tensors = {t["name"]: {**t, "consumers": []} for t in prog["tensors"]}

edges = []
for op in ops:
    for port, name in enumerate(op["inputs"]):
        src = tensors[name]["produced_by"]
        assert src is None or src < op["index"], f"op {op['index']} reads {name} before it is produced"
        edges.append({"tensor": name, "from": src, "to": op["index"], "port": port})
        if op["index"] not in tensors[name]["consumers"]:
            tensors[name]["consumers"].append(op["index"])
outputs = [n for n, t in tensors.items() if t["produced_by"] is not None and not t["consumers"]]
assert outputs == ["L"], f"unexpected graph outputs {outputs}"

# Group tree: one node per module path segment, then one per TEL statement (the symbol before the first '.').
root = {"id": "model", "label": prog["model"]["name"], "kind": "model", "module_type": None, "children": {}, "ops": []}
for op in ops:
    node, path = root, op["module"].split(".")
    for depth, label in enumerate(path):
        gid = ".".join(path[: depth + 1])
        node = node["children"].setdefault(gid, {"id": gid, "label": label, "kind": "block" if label.isdigit() else "module",
                                                 "module_type": None, "children": {}, "ops": []})
    node["module_type"] = op["module_type"]
    symbol = op["out"].split(".")[0]
    stmt = node["children"].setdefault(f"{node['id']}:{symbol}", {"id": f"{node['id']}:{symbol}", "label": symbol, "kind": "statement",
                                                                  "module_type": None, "children": {}, "ops": []})
    stmt["ops"].append(op["index"])


def finish(group: dict) -> dict:
    children = [finish(c) for c in group.pop("children").values()]
    members = sorted(group["ops"] + [i for c in children for i in c["members"]])
    inside = set(members)
    produced = {ops[i]["out"] for i in members}
    reads = dict.fromkeys(n for i in members for n in ops[i]["inputs"] if n not in produced)
    writes = [ops[i]["out"] for i in members if ops[i]["out"] in outputs or set(tensors[ops[i]["out"]]["consumers"]) - inside]
    return {**group, "members": members, "inputs": list(reads), "outputs": writes,
            "cost": sum(ops[i]["cost"] for i in members), "children": children}


groups = finish(root)
assert groups["members"] == list(range(len(ops)))


def size(axes) -> int:
    return math.prod(dims[a] for a in axes)


multiplications = []
for op in ops:
    if op["primitive"] != "Contraction":
        continue
    left, right = (tensors[n]["axes"] for n in op["inputs"])
    out, over = tensors[op["out"]]["axes"], op["attrs"]["over"]
    batch = [a for a in out if a in left and a in right]
    free_left = [a for a in left if a not in batch and a not in over]
    free_right = [a for a in right if a not in batch and a not in over]
    mnk = {"batch": size(batch), "m": size(free_left), "n": size(free_right), "k": size(over)}
    assert math.prod(mnk.values()) == op["cost"], f"op {op['index']}: {mnk} vs cost {op['cost']}"
    multiplications.append({"op": op["index"], "out": op["out"], "left": op["inputs"][0], "right": op["inputs"][1],
                            "module": op["module"], "batch_axes": batch, "free_left": free_left, "free_right": free_right,
                            "contracted": over, **mnk, "macs": op["cost"], "flops": 2 * op["cost"], "equation": op["equation"]})

doc = {
    "format": "tensormorph-viz-graph", "version": 1, "source": args.program,
    "model": prog["model"], "example": prog["example"], "dims": dims, "axes": prog["axes"], "aliases": prog["aliases"],
    "elided_modules": prog["elided_modules"], "regions": prog["regions"],
    "inputs": [n for n, t in tensors.items() if t["role"] == "input"], "outputs": outputs,
    "totals": {"tensors": len(tensors), "ops": len(ops), "edges": len(edges), "contractions": len(multiplications),
               "contraction_flops": sum(m["flops"] for m in multiplications), "cost": groups["cost"]},
    "tensors": list(tensors.values()), "ops": ops, "edges": edges, "groups": groups,
    "multiplications": multiplications, "verification": prog["verification"],
}
out = Path(args.out)
out.write_text(json.dumps(doc, indent=1))
t = doc["totals"]
print(f"{t['tensors']} tensors, {t['ops']} ops, {t['edges']} edges, {t['contractions']} contractions "
      f"({t['contraction_flops']:,} FLOPs at {prog['example']}) -> {out}")
