import re

import torch

from ..tracing.tensor_stats import storage_groups

Node = tuple[str, list]  # (label, children)


def _sig(node: Node) -> tuple:
    return (re.sub(r"^\d+", "", node[0]), tuple(_sig(c) for c in node[1]))


def _lines(children: list[Node], prefix: str, collapse: bool) -> list[str]:
    # identical numbered siblings (the 6 blocks) print once, then one summary line
    if collapse and len(children) > 1 and len({_sig(c) for c in children}) == 1:
        first = children[0]
        children = [first, (f"1..{len(children) - 1}: same structure as 0", [])]
    out = []
    for i, (label, grandchildren) in enumerate(children):
        last = i == len(children) - 1
        out.append(f"{prefix}{'└── ' if last else '├── '}{label}")
        out += _lines(grandchildren, prefix + ("    " if last else "│   "), collapse)
    return out


def render_tree(node: Node, collapse: bool = True) -> str:
    return "\n".join([node[0], *_lines(node[1], "", collapse)])


def module_tree(model: torch.nn.Module, collapse: bool = True) -> str:
    """Module hierarchy with parameter counts; tied parameters are flagged."""
    tied = {n for n, (_, others) in storage_groups(model.named_parameters(remove_duplicate=False)).items() if others}

    def node(name: str, module: torch.nn.Module, path: str) -> Node:
        count = sum(p.numel() for p in module.parameters())
        label = f"{name + ': ' if name else ''}{type(module).__name__}"
        label += f"  [{count:,} params]" if count else ""
        own_tied = [n for n, _ in module.named_parameters(recurse=False) if (f"{path}.{n}" if path else n) in tied]
        label += "  (tied)" if own_tied else ""
        return label, [node(n, m, f"{path}.{n}" if path else n) for n, m in module.named_children()]

    return render_tree(node("", model, ""), collapse)
