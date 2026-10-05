"""Versioned JSON trace (format "tensormorph-pytorch-trace" v1): parameters and activations kept apart."""
import json
from pathlib import Path

import torch

from .activation_recorder import ActivationRecorder
from .tensor_stats import storage_groups, storage_info

FORMAT = "tensormorph-pytorch-trace"
VERSION = 1
RECORD_KEYS = ("name", "shape", "dtype", "device", "numel")


def _records(stats_list, kept, max_values):
    """Trace tensor records; values only when the tensor was kept and numel <= max_values."""
    out = []
    for s in stats_list:
        record = {k: s[k] for k in RECORD_KEYS}
        if max_values and s["numel"] <= max_values:
            record["values"] = kept[s["name"]].cpu().flatten().tolist()
        out.append(record)
    return out


def _persistent_layout(name, kind, tensor):
    info = storage_info(tensor)
    info["storage_ptr"] = None  # raw addresses are not stable across runs; storage_id (parameters) is
    return {"name": name, "kind": kind, **info}


def build_trace(model, input_ids, model_name="distilgpt2", max_values=0) -> dict:
    """Run one eval forward under hooks. max_values > 0 also serializes tensors with numel <= max_values."""
    was_training = model.training
    model.eval()
    with torch.no_grad(), ActivationRecorder(model, keep_tensors=bool(max_values)) as recorder:
        model(input_ids)
    model.train(was_training)
    kept = {k: v for r in recorder.records for k, v in r.tensors.items()}

    params = list(model.named_parameters(remove_duplicate=False))
    groups = storage_groups(params)
    parameters = [
        _persistent_layout(n, "parameter", p) | {"storage_id": groups[n][0], "tied_with": groups[n][1]} for n, p in params
    ]
    buffers = [_persistent_layout(n, "buffer", b) for n, b in model.named_buffers()]
    modules = dict(model.named_modules())

    events = []
    for r in recorder.records:
        owned = [f"{r.module}.{n}" if r.module else n for n, _ in modules[r.module].named_parameters(recurse=False)]
        events.append(
            {
                "index": r.index,
                "exit_index": r.exit_index,
                "depth": r.depth,
                "module": r.module or "model",
                "module_type": r.module_type,
                "input_shapes": [s["shape"] for s in r.inputs],
                "output_shapes": [s["shape"] for s in r.outputs],
                "dtype": r.outputs[0]["dtype"] if r.outputs else None,
                "device": r.outputs[0]["device"] if r.outputs else None,
                "inputs": _records(r.inputs, kept, max_values),
                "outputs": _records(r.outputs, kept, max_values),
                "parameters": owned,
                "internals": _records([s | {"name": n} for n, s in r.internals.items()], kept, max_values),
            }
        )

    root = recorder.records[0]
    backbone = next(r for r in recorder.records if r.module == "transformer")
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": {"name": model_name, "class": type(model).__name__, "config": model.config.__dict__},
        "inputs": _records(root.inputs, kept, max_values),
        "parameters": parameters,
        "buffers": buffers,
        "events": events,
        "outputs": _records(backbone.outputs + root.outputs, kept, max_values),
    }


def write_trace(trace: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=1))
    return path
