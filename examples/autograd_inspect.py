"""python -m examples.autograd_inspect: grad_fn / is_leaf / requires_grad / .grad and the backward graph."""
from gpt2_pytorch.data import fixed_dataloader
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--max-nodes", type=int, default=30)
args = parser.parse_args()

model, _, _ = load_model(args)
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in next(iter(fixed_dataloader())).items()}
out = model(batch["input_ids"], labels=batch["labels"], return_debug=True)
weight = model.transformer.blocks[0].mlp.fc.weight
hidden = out.debug["block.0.mlp.fc.output"]

print(f"parameter: requires_grad={weight.requires_grad} is_leaf={weight.is_leaf} grad_fn={weight.grad_fn} grad={weight.grad}")
print(f"activation block.0.mlp.fc.output: requires_grad={hidden.requires_grad} is_leaf={hidden.is_leaf} grad_fn={type(hidden.grad_fn).__name__}")
print(f"input_ids: requires_grad={batch['input_ids'].requires_grad} (integer tensors never do)")
print(f"loss: grad_fn={type(out.loss.grad_fn).__name__}\n\nbackward graph from loss (BFS, first {args.max_nodes} nodes):")

seen, queue = set(), [out.loss.grad_fn]
while queue and len(seen) < args.max_nodes:
    node = queue.pop(0)
    if node is None or node in seen:
        continue
    seen.add(node)
    nexts = [type(n).__name__ for n, _ in node.next_functions if n is not None]
    print(f"  {type(node).__name__:<28} -> {nexts}")
    queue += [n for n, _ in node.next_functions]

hidden.retain_grad()  # non-leaf tensors drop their .grad unless asked to keep it
out.loss.backward()
print(f"\nafter backward: weight.grad shape={list(weight.grad.shape)}, hidden.grad shape={list(hidden.grad.shape)} (kept via retain_grad)")
