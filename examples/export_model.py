"""python -m examples.export_model: torch.export.export on the fixed input; writes artifacts/export/."""
import json
from pathlib import Path

import torch

from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--out-dir", default="artifacts/export")
args = parser.parse_args()

model, _, _ = load_model(args)
input_ids = FIXED_INPUT_IDS.to(next(model.parameters()).device)
with torch.no_grad():
    exported = torch.export.export(model, (input_ids,))

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)


def val_info(node):
    val = node.meta.get("val")
    return {"shape": list(val.shape), "dtype": str(val.dtype).removeprefix("torch.")} if isinstance(val, torch.Tensor) else None


nodes = [
    {"name": n.name, "op": n.op, "target": str(n.target), "args": [a.name if isinstance(a, torch.fx.Node) else repr(a) for a in n.args], **(val_info(n) or {})}
    for n in exported.graph.nodes
]
signature = [
    {"kind": spec.kind.name, "arg": spec.arg.name, "target": spec.target, "persistent": getattr(spec, "persistent", None)}
    for spec in exported.graph_signature.input_specs
]
(out / "graph.json").write_text(json.dumps({"format": "torch-export-graph", "inputs": signature, "nodes": nodes}, indent=1))
(out / "exported_program.txt").write_text(str(exported))

kinds = {}
for s in signature:
    kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
ops = {}
for n in nodes:
    if n["op"] == "call_function":
        ops[n["target"]] = ops.get(n["target"], 0) + 1
print(f"exported ({len(nodes)} graph nodes) -> {out}/graph.json, {out}/exported_program.txt")
print("graph inputs by kind:", kinds)
print("top aten ops:", sorted(ops.items(), key=lambda kv: -kv[1])[:10])
with torch.no_grad():
    same = torch.allclose(exported.module()(input_ids), model(input_ids), atol=1e-5)
print("exported module matches eager:", same)
