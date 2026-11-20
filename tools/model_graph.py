"""python -m tools.model_graph MODEL_DIR [--name SLUG] [--seq 8] [--out artifacts/models/SLUG/viz_graph.json]

A Hugging Face model folder (config.json + one or more *.safetensors) -> the tensormorph-viz-graph JSON the
Tensorviewport and TensorNodes open. Only the safetensors headers are read, never the weights. The graph is spelled
from config.json (supported: model_type qwen3_5, see gpt2_pytorch/tensorprog/qwen3_5.py), and every parameter must
bind a checkpoint tensor of the same shape, or the tool fails. An FP8 weight's `*_scale_inv` is named in its `view`.
Stored tensors the text forward pass never reads (vision tower, MTP head) are counted in the output's `checkpoint.unbound`.

Then copy it to where the viewer fetches it: node <TensormorphDesignUI>/tensormorph/ui/activity_bar/viewport/graphs/build.mjs OUT
(-> viewport/graphs/<slug>.json), and list the slug in FILES in viewport/graph.js.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from gpt2_pytorch.checkpoint.safetensors_loader import read_safetensors_header
from gpt2_pytorch.tensorprog.qwen3_5 import ROLES, lower_qwen3_5
from gpt2_pytorch.tensorprog.viz import viz_graph

LOWERINGS = {"qwen3_5": (lower_qwen3_5, ROLES, "model.language_model.")}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("model_dir", type=Path)
parser.add_argument("--name", help="model slug (default: the folder name, lower-cased)")
parser.add_argument("--seq", type=int, default=8, help="example sequence length T (the delta-rule recurrence is unrolled over it)")
parser.add_argument("--out", type=Path)
args = parser.parse_args()

name = args.name or args.model_dir.name.lower()
assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name), f"{name!r} is not a model slug the viewer accepts"
cfg = json.loads((args.model_dir / "config.json").read_text())
if cfg.get("model_type") not in LOWERINGS:
    raise SystemExit(f"model_type {cfg.get('model_type')!r} has no lowering (have: {', '.join(LOWERINGS)})")
lower, roles, prefix = LOWERINGS[cfg["model_type"]]

stored = {}  # checkpoint key -> (file, dtype, shape)
for f in sorted(args.model_dir.glob("*.safetensors")):
    header = read_safetensors_header(f)
    for key, entry in header.items():
        if not key.startswith("__"):
            stored[key] = (f.name, entry["dtype"], entry["shape"])
if not stored:
    raise SystemExit(f"no *.safetensors in {args.model_dir}")

program = lower(cfg, batch=1, seq=args.seq, prefix=prefix)
bound = set()
for t in program.tensors.values():
    if not t.checkpoint_key:
        continue
    if t.checkpoint_key not in stored:
        raise SystemExit(f"{t.name} binds {t.checkpoint_key}, which no safetensors file holds")
    file, dtype, shape = stored[t.checkpoint_key]
    if list(t.shape) != shape:
        raise SystemExit(f"{t.name} {list(t.shape)} binds {t.checkpoint_key} {shape} ({file})")
    scale = t.checkpoint_key + "_scale_inv"
    if dtype.startswith("F8") and scale in stored:
        t.view = f"dequantized: {dtype} x {scale} {stored[scale][2]} (block-wise)"
        bound.add(scale)
    bound.add(t.checkpoint_key)
unbound = sorted(set(stored) - bound)
lm = [k for k in unbound if k.startswith(prefix) or k.startswith("lm_head")]
if lm:
    raise SystemExit(f"{len(lm)} language-model tensors are not in the graph, first {lm[:3]}")

prog = program.to_dict(name, {"B": 1, "T": args.seq}) | {"verification": [], "roles": roles}
doc = viz_graph(prog, f"tools/model_graph.py {args.model_dir.name} (spelled from config.json)")
doc["checkpoint"] = {
    "files": sorted({f for f, _, _ in stored.values()}), "tensors": len(stored), "bound": len(bound),
    "unbound": dict(Counter(re.match(r"(model\.)?[^.]+", k).group() for k in unbound)),  # model.visual, mtp
}
out = args.out or Path("artifacts/models") / name / "viz_graph.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, separators=(",", ":")))
t = doc["totals"]
print(f"{name}: {t['tensors']} tensors, {t['ops']} ops, {t['contractions']} contractions; "
      f"{len(bound)} of {len(stored)} stored tensors bound, unbound {doc['checkpoint']['unbound']} -> {out} "
      f"({out.stat().st_size / 2**20:.1f} MiB)")
