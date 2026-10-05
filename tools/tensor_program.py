"""python -m tools.tensor_program [--tel-model-json PATH]

Lowers the model into a flat list of tensor operations (every operand a tensor), verifies it numerically against
the pure PyTorch model on all fixed samples, and writes artifacts/tensor_program.{md,json,txt}.
"""
import csv
import json
import sys
from pathlib import Path

import torch

from gpt2_pytorch.checkpoint import load_pretrained, load_state_dict_file
from gpt2_pytorch.data import FIXED_INPUT_IDS, SAMPLES
from gpt2_pytorch.tensorprog import listing, lower, render_markdown, verify
from gpt2_pytorch.utils.cli import common_parser

parser = common_parser(__doc__)
parser.add_argument("--out-dir", default="artifacts")
parser.add_argument("--tel-model-json", default="artifacts/tel/model.json",
                    help="model.json written by TEL's tensorparser; adds a side-by-side op count when the file exists")
args = parser.parse_args()
if not Path(args.tel_model_json).exists():
    args.tel_model_json = None

model, _, _ = load_pretrained(args.checkpoint)
state = load_state_dict_file(args.checkpoint)
batch, seq = FIXED_INPUT_IDS.shape
program, instances = lower(model, batch, seq)

results = []
for name, ids in SAMPLES.items():
    results.append({"sample": name, **verify(program, model, state, ids)})
    r = results[-1]
    print(f"{name:<24}{str(r['shape']):<9}{r['compared']} tensors  worst rel err {r['worst_relative_error']:.2e} ({r['worst_tensor']})  {'PASS' if r['ok'] else 'FAIL'}")

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
example = {"B": batch, "T": seq}
(out / "tensor_program.json").write_text(json.dumps(program.to_dict("distilgpt2", example) | {"verification": results}, indent=1))
(out / "tensor_program.txt").write_text(listing(program) + "\n")
(out / "tensor_program.md").write_text(render_markdown(program, instances, results, Path(args.checkpoint).name, example, args.tel_model_json))
mults = [op for op in program.ops if program.is_multiplication(op)]
with open(out / "tensor_multiplications.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "op", "kind", "output", "left", "right", "summed_axes", "cost", "module"])
    for n, op in enumerate(mults, 1):
        left, right = op.inputs
        writer.writerow([n, op.index, "contraction" if op.primitive == "Contraction" else "mul", program.ref(op.out), program.ref(left),
                         program.ref(right), ",".join(op.attrs.get("over", [])), program.cost(op), op.module])
print(f"{len(program.tensors)} tensors, {len(program.ops)} ops, {len(mults)} multiplications -> {out}/tensor_program.md, .json, .txt, tensor_multiplications.csv")
sys.exit(0 if all(r["ok"] for r in results) else 1)
