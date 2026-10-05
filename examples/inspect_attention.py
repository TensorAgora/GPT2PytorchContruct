"""python -m examples.inspect_attention: Q/K/V, scores, mask and probabilities of block 0 on fixed ids."""
import torch

from gpt2_pytorch.data import SAMPLES
from gpt2_pytorch.tracing import fmt_shape
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--block", type=int, default=0)
parser.add_argument("--head", type=int, default=0)
args = parser.parse_args()

model, _, _ = load_model(args)
input_ids = SAMPLES["sample_attention_debug"].to(next(model.parameters()).device)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

with torch.no_grad():
    debug = model(input_ids, return_debug=True).debug
    prefix = f"block.{args.block}.attention"
    # the same tensors, straight from the attention module's own debug mode
    attn = model.transformer.blocks[args.block].attention
    output, local = attn(debug[f"block.{args.block}.ln_1.output"], return_debug=True)

for key, tensor in local.items():
    print(f"{key:<18}{fmt_shape(tensor.shape):<18}{str(tensor.dtype)}")
probs = local["attention_probs"]
assert torch.equal(probs, debug[f"{prefix}.probs"])
print(f"\ncausal_mask [T, T] (True = may attend):\n{local['causal_mask'].int()}")
print(f"\nattention_probs[0, head {args.head}] (row i = query token i):\n{probs[0, args.head]}")
future = probs.masked_fill(local["causal_mask"], 0.0)
print(f"\nrow sums: min={probs.sum(-1).min():.6f} max={probs.sum(-1).max():.6f}")
print(f"max probability on future positions (j > i): {future.max().item():.1f}")
print(f"most attended key per query (head {args.head}): {probs[0, args.head].argmax(-1).tolist()}")
