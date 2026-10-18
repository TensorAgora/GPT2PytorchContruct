"""python -m tools.matmul_graph [--checkpoint PATH] [--model module:function] [--out-dir artifacts]

state_dict + model code -> graph of matrix multiplications in execution order -> artifacts/matmul_graph.{json,md}.

The weights file holds no graph, so the model code supplies it: `--model pkg.mod:build` names a function
build(state_dict) -> (loaded model, example_args). Typical body:

    model = MyNet(...)                                  # shapes may be inferred from state_dict
    load_state_dict_checked(model, state_dict)          # from gpt2_pytorch.tracing
    return model, (torch.randn(1, 3, 224, 224),)

Without --model, the checkpoint is read as Hugging Face DistilGPT2 (this repo's model and key mapping).
"""
import importlib
import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

from gpt2_pytorch.checkpoint import infer_config, load_state_dict_file
from gpt2_pytorch.checkpoint.loader import load_state_dict_into
from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.model import DistilGPT2LMHeadModel
from gpt2_pytorch.tracing import matmul_graph, to_markdown
from gpt2_pytorch.utils.cli import common_parser
from gpt2_pytorch.utils.seed import resolve_device


def distilgpt2(state):
    model = DistilGPT2LMHeadModel(infer_config(state))
    load_state_dict_into(model, state)  # HF names -> ours, Conv1D transposed
    return model.eval(), (FIXED_INPUT_IDS,)


parser = common_parser(__doc__)
parser.add_argument("--model", help="module:function returning (model, example_args); default: DistilGPT2")
parser.add_argument("--out-dir", default="artifacts")
args = parser.parse_args()

if args.model:
    module, fn = args.model.split(":")
    build = getattr(importlib.import_module(module), fn)
else:
    build = distilgpt2
model, example_args = build(load_state_dict_file(args.checkpoint))
device = resolve_device(args.device)
model, example_args = model.to(device), tree_map(lambda a: a.to(device) if isinstance(a, torch.Tensor) else a, example_args)
graph = matmul_graph(model, example_args)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "matmul_graph.json").write_text(json.dumps(graph, indent=1))
(out / "matmul_graph.md").write_text(to_markdown(graph))
for v in graph["nodes"]:
    o = v["operands"]
    weight = next((o[r]["source"] for r in ("right", "left") if o[r]["kind"] == "parameter"), "activation × activation")
    print(f"{v['index']:>3} {v['id']:<10} {v['op']:<6} {str(v['out_shape']):<16} {weight:<55} <- {', '.join(v['deps'])}")
print(f"{len(graph['nodes'])} matmuls, {len(graph['edges'])} edges, {graph['total_flops'] / 1e6:,.1f} MFLOPs, "
      f"max relative diff vs eager {graph['max_rel_diff_vs_eager']:.1e} -> {out}/matmul_graph.json, .md")
