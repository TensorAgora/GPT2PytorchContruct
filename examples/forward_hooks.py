"""python -m examples.forward_hooks: execution order via register_forward_pre_hook / register_forward_hook."""
import torch

from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.tracing import fmt_shape
from gpt2_pytorch.utils.cli import common_parser, load_model

args = common_parser(__doc__).parse_args()
model, _, _ = load_model(args)
input_ids = FIXED_INPUT_IDS.to(next(model.parameters()).device)

entered, exited, shapes, depth = [], [], {}, []
current = []  # stack of module names being executed


def pre_hook(name):
    def hook(module, inputs):
        entered.append((name, type(module).__name__, len(current)))
        current.append(name)

    return hook


def post_hook(name):
    def hook(module, inputs, output):
        current.pop()
        exited.append((name, type(module).__name__))
        shapes[name] = (fmt_shape(inputs[0].shape), fmt_shape(output.shape))

    return hook


handles = []
for name, module in model.named_modules():
    handles.append(module.register_forward_pre_hook(pre_hook(name)))
    handles.append(module.register_forward_hook(post_hook(name)))
with torch.no_grad():
    model(input_ids)
for h in handles:
    h.remove()

print("entry order (pre-hook):")
for i, (name, kind, d) in enumerate(entered):
    print(f"{i:03d} {'  ' * d}{kind:<22}{name or '<root>':<{58 - 2 * d}}{shapes[name][0]} -> {shapes[name][1]}")
print("\ncompletion order (forward hook), first 14:")
for i, (name, kind) in enumerate(exited[:14]):
    print(f"{i:03d} {kind}")
print(f"\n{len(entered)} module calls; parameters/buffers untouched, hooks removed.")
