"""python -m tools.convert_to_safetensors [--checkpoint pytorch_model.bin] [--out artifacts/model.safetensors]

Tensor values are copied unchanged. Tied storages are written once: `lm_head.weight` is OMITTED (it shares
storage with `transformer.wte.weight`) and reconstructed on load by the model's own weight tying.
Also written: the 6 `attn.bias` causal-mask buffers, so the key set equals Hugging Face's model.safetensors.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT
from gpt2_pytorch.checkpoint.pytorch_loader import load_torch_state_dict
from gpt2_pytorch.checkpoint.safetensors_loader import convert_to_safetensors

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
parser.add_argument("--out", default="artifacts/model.safetensors")
args = parser.parse_args()

state = load_torch_state_dict(args.checkpoint)
dropped = convert_to_safetensors(state, args.out, source=Path(args.checkpoint).name)

reloaded = load_file(args.out)
assert reloaded.keys() == state.keys() - dropped.keys()
assert all(torch.equal(reloaded[k], state[k]) for k in reloaded), "value mismatch after round trip"
print(f"wrote {args.out} ({Path(args.out).stat().st_size:,} bytes): {len(reloaded)} tensors, values bit-identical to the .bin")
print(f"omitted tied tensors (reconstructed on load): {dropped}")
