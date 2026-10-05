"""python -m examples.fx_trace: torch.fx.symbolic_trace of the unmodified model.

The model returns a plain tensor for `model(input_ids)`; `concrete_args` freezes the other optional
arguments (labels, position_ids, return_debug) so FX specialises on the logits-only path. The reference code
avoids what breaks FX (shape unpacking, dtype-dependent constants, dataclass returns on this path).
Built-in layers (Linear/LayerNorm/Embedding/Dropout) stay call_module leaves; GELU, attention maths are inlined.
"""
import json
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--out-dir", default="artifacts")
parser.add_argument("--rows", type=int, default=45, help="rows to print (the full table goes to the .txt)")
args = parser.parse_args()

model, _, _ = load_model(args)
gm = symbolic_trace(model, concrete_args={"labels": None, "position_ids": None, "return_debug": False})
ShapeProp(gm).propagate(FIXED_INPUT_IDS.to(next(model.parameters()).device))


def describe(node):
    meta = node.meta.get("tensor_meta")
    shape = list(meta.shape) if hasattr(meta, "shape") else None
    return {
        "name": node.name,
        "op": node.op,
        "target": node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target)),
        "args": [a.name if isinstance(a, torch.fx.Node) else repr(a) for a in node.args],
        "shape": shape,
        "dtype": str(meta.dtype).removeprefix("torch.") if hasattr(meta, "dtype") else None,
    }


nodes = [describe(n) for n in gm.graph.nodes]
name_w = max(len(n["name"]) for n in nodes) + 2
target_w = max(len(str(n["target"])) for n in nodes) + 2
header = f"{'opcode':<16}{'name':<{name_w}}{'target':<{target_w}}{'shape':<20}args"
lines = [header] + [
    f"{n['op']:<16}{n['name']:<{name_w}}{str(n['target']):<{target_w}}{str(n['shape']).replace(' ', ''):<20}{', '.join(n['args'])}"
    for n in nodes
]
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "fx_graph.txt").write_text("\n".join(lines) + "\n")
(out / "fx_graph.json").write_text(json.dumps({"format": "fx-graph", "nodes": nodes}, indent=1))

print("\n".join(lines[: args.rows + 1]))
print(f"... {len(nodes)} nodes total -> {out / 'fx_graph.txt'}, {out / 'fx_graph.json'}")
ops = {}
for n in nodes:
    ops[n["op"]] = ops.get(n["op"], 0) + 1
print("node kinds:", ops)
with torch.no_grad():
    same = torch.equal(gm(FIXED_INPUT_IDS.to(next(model.parameters()).device)), model(FIXED_INPUT_IDS.to(next(model.parameters()).device)))
print("traced module output == eager output:", same)
