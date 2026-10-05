"""python -m examples.causal_lm_loss: next-token loss with every intermediate exposed."""
import torch
import torch.nn.functional as F

from gpt2_pytorch.data import fixed_dataloader
from gpt2_pytorch.tracing import fmt_shape
from gpt2_pytorch.utils.cli import common_parser, load_model

args = common_parser(__doc__).parse_args()
model, _, _ = load_model(args)
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in next(iter(fixed_dataloader())).items()}
input_ids, labels = batch["input_ids"], batch["labels"]

with torch.no_grad():
    result = model(input_ids, labels=labels, return_debug=True)
shift_logits, shift_labels = result.debug["loss.shift_logits"], result.debug["loss.shift_labels"]
print(f"logits       {fmt_shape(result.logits.shape)}\nshift_logits {fmt_shape(shift_logits.shape)}  (logits[:, :-1])")
print(f"shift_labels {fmt_shape(shift_labels.shape)}  (labels[:, 1:])\n{shift_labels}")
per_token = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="none")
print(f"per-token loss: {[round(x, 3) for x in per_token.tolist()]}")
print(f"loss = {result.loss.item():.6f}  (mean of per-token = {per_token.mean().item():.6f})")
assert torch.allclose(result.loss, per_token.mean())
