import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file


def read_safetensors_header(path) -> dict:
    """Parse only the header: 8-byte little-endian length, then JSON.

    Returns {"__metadata__": {...}, name: {"dtype", "shape", "data_offsets": [begin, end]}, ...};
    offsets are relative to the end of the header. No tensor data is read.
    """
    with open(Path(path), "rb") as f:
        def get_range(start, end):
            f.seek(start)
            return f.read(end - start + 1)

        header, header_bytes = parse_safetensors_header(get_range)
    header["__header_bytes__"] = header_bytes
    return header


def parse_safetensors_header(get_range, max_header=100 << 20) -> tuple[dict, int]:
    """The header through a byte-range reader `get_range(start, end_inclusive) -> bytes`: bytes 0-7 give N (u64 LE),
    bytes 8..7+N the JSON. Returns (header, 8 + N); nothing past the header is requested."""
    raw = get_range(0, 7)
    if len(raw) != 8:
        raise EOFError("truncated safetensors length prefix")
    (n,) = struct.unpack("<Q", raw)
    if n > max_header:
        raise ValueError(f"safetensors header length {n} is implausible")
    body = get_range(8, 7 + n)
    if len(body) != n:
        raise EOFError(f"truncated safetensors header: {len(body)} of {n} bytes")
    return json.loads(body), 8 + n


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
