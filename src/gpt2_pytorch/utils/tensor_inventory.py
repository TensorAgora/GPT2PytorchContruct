import csv
import json
from pathlib import Path

import torch

from ..tracing.tensor_stats import storage_groups, storage_info

COLUMNS = [
    "name", "module", "role", "shape", "rank", "dtype", "numel", "bytes", "requires_grad", "shared",
    "stride", "storage_offset", "is_contiguous", "is_view", "storage_id",
]


def build_inventory(model: torch.nn.Module) -> list[dict]:
    """One row per parameter name (tied names listed separately) plus buffers."""
    modules = dict(model.named_modules())
    params = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers())
    groups = storage_groups(params + buffers)
    rows = []
    for role, items in (("parameter", params), ("buffer", buffers)):
        for name, tensor in items:
            owner = modules[name.rpartition(".")[0]]
            info = storage_info(tensor)
            storage_id, others = groups[name]
            rows.append(
                {
                    "name": name,
                    "module": type(owner).__name__,
                    "role": role,
                    "shape": info["shape"],
                    "rank": tensor.dim(),
                    "dtype": info["dtype"],
                    "numel": info["numel"],
                    "bytes": info["bytes"],
                    "requires_grad": info["requires_grad"],
                    "shared": "shared-with-" + ",".join(others) if others else "no",
                    "stride": info["stride"],
                    "storage_offset": info["storage_offset"],
                    "is_contiguous": info["is_contiguous"],
                    "is_view": info["is_view"],
                    "storage_id": storage_id,
                }
            )
    return rows


def write_inventory(rows: list[dict], json_path, csv_path) -> None:
    for p in (json_path, csv_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(rows, indent=1))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
