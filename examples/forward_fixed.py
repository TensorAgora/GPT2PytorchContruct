"""python -m examples.forward_fixed [--all-blocks]: one deterministic eval forward on fixed token ids."""
import torch

from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.tracing import fmt_shape
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--all-blocks", action="store_true", help="print the tensors of every block, not only block 0")
args = parser.parse_args()

model, config, report = load_model(args)
device = next(model.parameters()).device
input_ids = FIXED_INPUT_IDS.to(device)
print(f"loaded: {len(report.loaded)}  transposed: {len(report.transposed)}  tied: {len(report.tied)}  ignored: {len(report.ignored)}\n")

assert not model.training
with torch.no_grad():
    result = model(input_ids, return_debug=True)

for name, tensor in result.debug.items():
    block = name.split(".")[1] if name.startswith("block.") else None
    if block is not None and not args.all_blocks and block != "0" and not (block == str(config.num_layers - 1) and name.endswith(".output") and name.count(".") == 2):
        continue
    print(f"{name:<34}{fmt_shape(tensor.shape)}")

logits = result.logits
print(f"\ninput_ids:\n{input_ids}")
print(f"\nlogits[0, -1, :5] = {logits[0, -1, :5].tolist()}")
print(f"logits[1, 0, :5]  = {logits[1, 0, :5].tolist()}")
print(f"argmax next token per position:\n{logits.argmax(-1)}")
print(f"logits stats: min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f} std={logits.std():.4f}")
