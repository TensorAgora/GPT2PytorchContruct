"""python -m examples.single_train_step: forward -> loss -> backward -> gradients -> one SGD step (on one fixed batch).

Runs in eval() mode so dropout cannot make the step nondeterministic; this is a lifecycle demo, not training.
"""
import torch

from gpt2_pytorch.data import fixed_dataloader
from gpt2_pytorch.tracing import tensor_stats
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

model, _, _ = load_model(args)
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in next(iter(fixed_dataloader())).items()}

model.zero_grad(set_to_none=True)
result = model(batch["input_ids"], labels=batch["labels"])
print(f"loss before: {result.loss.item():.6f}  grad_fn={type(result.loss.grad_fn).__name__}")
result.loss.backward()

watch = {
    "embedding (tied with lm_head)": model.transformer.token_embedding.weight,
    "attention qkv weight [out, in]": model.transformer.blocks[0].attention.qkv_projection.weight,
    "mlp fc weight [out, in]": model.transformer.blocks[0].mlp.fc.weight,
    "ln_1 weight (block 0)": model.transformer.blocks[0].ln_1.weight,
    "final layernorm weight": model.transformer.final_layer_norm.weight,
}
for label, param in watch.items():
    g = tensor_stats(param.grad)
    print(f"{label:<32}grad shape={g['shape']} mean={g['mean']:+.3e} std={g['std']:.3e} max|g|={max(abs(g['min']), abs(g['max'])):.3e}")
rows_hit = (model.transformer.token_embedding.weight.grad.abs().sum(1) > 0).sum().item()
print(f"embedding rows with nonzero grad: {rows_hit} of {model.config.vocab_size} (lm_head softmax touches all rows)")

with torch.no_grad():
    for param in model.parameters():
        param -= args.lr * param.grad
    after = model(batch["input_ids"], labels=batch["labels"]).loss
print(f"loss after one SGD step (lr={args.lr}): {after.item():.6f}")
