import json
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_safetensors_state_dict(path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


def read_safetensors_header(path) -> dict:
    """Parse only the header: 8-byte little-endian length, then JSON.

    Returns {"__metadata__": {...}, name: {"dtype", "shape", "data_offsets": [begin, end]}, ...};
    offsets are relative to the end of the header. No tensor data is read.
    """
    with open(Path(path), "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header["__header_bytes__"] = 8 + header_len
    return header


def convert_to_safetensors(state: dict[str, torch.Tensor], out, source: str = "") -> dict[str, str]:
    """Write `state` to `out` with values unchanged. Tensors sharing storage are written once (first key wins);
    the omitted aliases are returned and also stored in the file metadata under "tied"."""
    kept, dropped, owner = {}, {}, {}
    for name, tensor in state.items():
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in owner:
            assert torch.equal(tensor, state[owner[ptr]]), f"{name} aliases {owner[ptr]} but differs"
            dropped[name] = owner[ptr]
        else:
            owner[ptr] = name
            kept[name] = tensor.contiguous()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(out), metadata={"format": "pt", "source": source, "tied": json.dumps(dropped)})
    return dropped
