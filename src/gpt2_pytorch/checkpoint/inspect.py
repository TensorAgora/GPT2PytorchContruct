from dataclasses import dataclass
from pathlib import Path

import torch

from ..tracing.tensor_stats import fmt_shape
from ..utils.model_tree import Node, render_tree
from .infer_config import infer_config
from .loader import load_state_dict_file


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int
    storage_ptr: int


def inspect_state_dict(state: dict[str, torch.Tensor]) -> list[TensorInfo]:
    return [
        TensorInfo(k, tuple(v.shape), str(v.dtype), v.numel(), v.numel() * v.element_size(), v.untyped_storage().data_ptr())
        for k, v in state.items()
    ]


def summarize(infos: list[TensorInfo]) -> dict:
    """Logical totals count every key; unique totals count each underlying storage once (= file size)."""
    unique = {i.storage_ptr: i for i in infos}
    return {
        "tensor_count": len(infos),
        "logical_numel": sum(i.numel for i in infos),
        "logical_bytes": sum(i.nbytes for i in infos),
        "unique_numel": sum(i.numel for i in unique.values()),
        "unique_bytes": sum(i.nbytes for i in unique.values()),
        "tied": {i.name: [j.name for j in infos if j.storage_ptr == i.storage_ptr and j is not i] for i in infos
                 if any(j.storage_ptr == i.storage_ptr and j is not i for j in infos)},
    }


def checkpoint_tree(infos: list[TensorInfo], collapse: bool = True) -> str:
    root: dict = {}
    for info in infos:
        node = root
        for part in info.name.split("."):
            node = node.setdefault(part, {})
        node["__shape__"] = info.shape

    def to_node(label: str, tree: dict) -> Node:
        if "__shape__" in tree:
            return f"{label} {fmt_shape(tree['__shape__']).replace(' ', '')}", []
        return label, [to_node(k, v) for k, v in tree.items()]

    return render_tree(("(checkpoint)", [to_node(k, v) for k, v in root.items()]), collapse)


def describe_checkpoint(path, full: bool = False) -> str:
    path = Path(path)
    state = load_state_dict_file(path)
    infos = inspect_state_dict(state)
    s = summarize(infos)
    fmt = "safetensors" if path.suffix == ".safetensors" else "PyTorch state_dict (pickle, weights_only)"
    lines = [
        f"Checkpoint:\n  {path.name}",
        f"Format:\n  {fmt}",
        f"Tensor count:\n  {s['tensor_count']}",
        f"Scalar count:\n  {s['logical_numel']:,} logical, {s['unique_numel']:,} unique storage",
        f"Size:\n  {s['unique_bytes'] / 2**20:.2f} MiB unique ({s['unique_bytes']:,} bytes); on-disk {path.stat().st_size:,} bytes",
        f"Dtypes:\n  {sorted({i.dtype for i in infos})}",
        f"Tied (shared storage):\n  {s['tied'] or 'none'}",
    ]
    try:
        c = infer_config(state)
        lines.append(
            "Architecture inference:\n"
            f"  vocab_size: {c.vocab_size}\n  hidden_size: {c.hidden_size}\n  positions: {c.max_position_embeddings}\n"
            f"  layers: {c.num_layers}\n  heads: {c.num_heads} (head_dim {c.head_dim}; heads are assumed, not in any shape)\n"
            f"  mlp_size: {c.intermediate_size}"
        )
    except KeyError as e:
        lines.append(f"Architecture inference: failed, missing {e}")
    lines.append("Tensors:")
    lines += [f"{i.name}\n  shape = {fmt_shape(i.shape)}\n  dtype = {i.dtype}" for i in infos if full or ".h." not in i.name or ".h.0." in i.name]
    lines.append("Hierarchy:\n" + checkpoint_tree(infos, collapse=not full))
    return "\n".join(lines)
