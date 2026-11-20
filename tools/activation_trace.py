"""python -m tools.activation_trace [--sample sample_batch] [--out-dir artifacts/trace] [--dtype float16]

Runs the verified tensor program (the same program viz_graph.json draws) on one fixed sample and keeps every tensor it
computes: the model inputs and every op output, keyed by the graph's tensor names (X_0, N1_0.mean, A_2, L, ...).
Parameters stay in the checkpoint and are not copied. Writes

  artifacts/trace/<sample>.safetensors   the values (floats in --dtype, integer inputs as int64)
  artifacts/trace/<sample>.json          the manifest: format tensormorph-activation-trace v1, dims, input ids, tokens,
                                          the safetensors file name and this sample's check against the PyTorch model

The viewer copies both to viewport/graphs/<model>.trace.{json,safetensors} with tensormorph/ui/activity_bar/viewport/graphs/build.mjs.
"""
import json
import struct
import sys
from pathlib import Path

import torch

from gpt2_pytorch.checkpoint import load_state_dict_file, pretrained_from_state
from gpt2_pytorch.data import SAMPLES
from gpt2_pytorch.tensorprog import lower, verify
from gpt2_pytorch.tensorprog.interpret import run_program
from gpt2_pytorch.utils.cli import common_parser

ST_DTYPE = {torch.float16: "F16", torch.float32: "F32", torch.int64: "I64"}


def save(tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> bytes:
    """The safetensors layout (8-byte header length, JSON header, raw little-endian data), without numpy."""
    header, chunks, at = {"__metadata__": metadata}, [], 0
    for name, t in tensors.items():
        raw = bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
        header[name] = {"dtype": ST_DTYPE[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        chunks.append(raw)
        at += len(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # the data starts 8-byte aligned
    return struct.pack("<Q", len(head)) + head + b"".join(chunks)


parser = common_parser(__doc__)
parser.add_argument("--sample", default="sample_batch", choices=sorted(SAMPLES))
parser.add_argument("--out-dir", default="artifacts/trace")
parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
parser.add_argument("--tokenizer", default="../tensormorph-model-export/distilgpt2.tensormorph/tokenizer/tokenizer.json",
                    help="a byte-level BPE tokenizer.json, only to label the tokens; ids are kept when it is missing")
args = parser.parse_args()

state = load_state_dict_file(args.checkpoint)
model, _, _ = pretrained_from_state(state)
ids = SAMPLES[args.sample]
batch, seq = ids.shape
program, _ = lower(model, batch, seq)

check = verify(program, model, state, ids)  # the program against the PyTorch model on this very input
if not check["ok"]:
    sys.exit(f"{args.sample}: the program disagrees with PyTorch (worst {check['worst_relative_error']:.2e} on {check['worst_tensor']})")
env = run_program(program, state, ids)

keep, skipped = {}, []
want = getattr(torch, args.dtype)
for name, decl in program.tensors.items():
    if decl.role == "parameter":
        continue
    t = env.get(name)
    if t is None:
        skipped.append(name)
        continue
    t = t.detach().cpu().contiguous()
    keep[name] = t.to(torch.int64) if not t.is_floating_point() else t.to(want)

tokens = [str(i) for i in ids.reshape(-1).tolist()]
tok = Path(args.tokenizer)
if tok.exists():
    vocab = {v: k for k, v in json.loads(tok.read_text())["model"]["vocab"].items()}
    tokens = [vocab.get(i, str(i)).replace("Ġ", " ").replace("Ċ", "\\n") for i in ids.reshape(-1).tolist()]

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
blob = save(keep, metadata={"format": "tensormorph-activation-trace", "model": "distilgpt2", "sample": args.sample})
(out / f"{args.sample}.safetensors").write_bytes(blob)
manifest = {
    "format": "tensormorph-activation-trace", "version": 1, "model": "distilgpt2", "sample": args.sample,
    "dims": {"B": batch, "T": seq}, "input_ids": ids.tolist(), "tokens": [tokens[b * seq:(b + 1) * seq] for b in range(batch)],
    "dtype": args.dtype, "file": f"{args.sample}.safetensors", "tensors": len(keep), "bytes": len(blob),
    "source": "tools/tensor_program run_program (torch, the checkpoint's own weights)",
    "check": {"compared": check["compared"], "worst_relative_error": check["worst_relative_error"], "worst_tensor": check["worst_tensor"],
              "logits_max_abs_error": check["logits_max_abs_error"], "argmax_identical": check["argmax_identical"]},
}
(out / f"{args.sample}.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"{args.sample} {list(ids.shape)}: {len(keep)} tensors, {len(blob) / 1e6:.1f} MB {args.dtype} -> {out}/{args.sample}.safetensors"
      f" (checked {check['compared']} tensors, worst rel err {check['worst_relative_error']:.2e}){'; no value: ' + ', '.join(skipped) if skipped else ''}")
