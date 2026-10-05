import torch


def fmt_shape(shape) -> str:
    return "[" + ", ".join(str(int(d)) for d in shape) + "]"


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def storage_info(t: torch.Tensor) -> dict:
    """Layout facts only (no values)."""
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": t.storage_offset(),
        "is_contiguous": t.is_contiguous(),
        "is_view": t._base is not None,
        "dtype": _dtype(t),
        "device": str(t.device),
        "requires_grad": t.requires_grad,
        "numel": t.numel(),
        "element_size": t.element_size(),
        "bytes": t.numel() * t.element_size(),  # logical bytes of this tensor
        "storage_ptr": t.untyped_storage().data_ptr(),
        "storage_bytes": t.untyped_storage().nbytes(),
    }


def tensor_stats(t: torch.Tensor) -> dict:
    """Layout facts plus min/max/mean/std (computed on a detached float copy)."""
    stats = storage_info(t)
    if t.numel() == 0:
        return stats | {"min": None, "max": None, "mean": None, "std": None}
    f = t.detach().float()
    return stats | {
        "min": f.min().item(),
        "max": f.max().item(),
        "mean": f.mean().item(),
        "std": f.std(correction=0).item(),
    }


def storage_groups(named_tensors) -> dict[str, tuple[str, list[str]]]:
    """name -> (stable storage id, other names sharing that storage). Detects tying / aliasing."""
    named = list(named_tensors)
    ids, members = {}, {}
    for name, t in named:
        sid = ids.setdefault(t.untyped_storage().data_ptr(), f"storage:{len(ids)}")
        members.setdefault(sid, []).append(name)
    return {
        name: (sid, [m for m in members[sid] if m != name])
        for name, t in named
        for sid in [ids[t.untyped_storage().data_ptr()]]
    }
