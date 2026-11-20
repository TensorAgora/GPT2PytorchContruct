"""python -m tools.viz_graph [--program artifacts/tensor_program.json] [--out artifacts/viz_graph.json]

Turns the verified tensor program (python -m tools.tensor_program) into the tensormorph-viz-graph JSON the Tensorviewport
and TensorNodes open (gpt2_pytorch.tensorprog.viz). Other models: python -m tools.model_graph.
"""
import argparse
import json
from pathlib import Path

from gpt2_pytorch.tensorprog.viz import viz_graph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--program", default="artifacts/tensor_program.json")
parser.add_argument("--out", default="artifacts/viz_graph.json")
args = parser.parse_args()

prog = json.loads(Path(args.program).read_text())
doc = viz_graph(prog, args.program)
out = Path(args.out)
out.write_text(json.dumps(doc, indent=1))
t = doc["totals"]
print(f"{t['tensors']} tensors, {t['ops']} ops, {t['edges']} edges, {t['contractions']} contractions "
      f"({t['contraction_flops']:,} FLOPs at {prog['example']}) -> {out}")
