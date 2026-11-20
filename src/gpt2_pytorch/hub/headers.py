"""S3 headers by HTTP range (master §5.1), 0 payload bytes.

Safetensors: bytes 0-7 give N, bytes 8..7+N the JSON (checkpoint/safetensors_loader.parse_safetensors_header).
GGUF: magic, version, counts, the KV section and the tensor-info table, read in windows (1 MiB, doubling) that are each
capped by a provable lower bound on the header bytes still ahead, so no request reaches tensor data. Never GGUFReader.
"""
import json
import struct

from ..checkpoint.safetensors_loader import parse_safetensors_header
from .http import HttpError

GGUF_MAGIC = b"GGUF"
ARRAY_HEAD, ARRAY_SUMMARY_OVER = 8, 64
MIN_KV = 8 + 1 + 4 + 1  # key length, >= 1 key byte, value type, >= 1 value byte
MIN_TENSOR_INFO = 8 + 1 + 4 + 8 + 4 + 8  # name length, >= 1 name byte, n_dims, >= 1 dim, ggml type, offset
DEFAULT_BUDGET = 64 << 20  # D3 per GGUF file: 4x the 100-row smoke maximum (15.8 MB); per model see manifest.MODEL_HEADER_BUDGET


class HeaderError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


def _gguf_types():
    from gguf.constants import GGUFValueType as V, GGMLQuantizationType as Q
    fixed = {V.UINT8: ("<B", 1), V.INT8: ("<b", 1), V.UINT16: ("<H", 2), V.INT16: ("<h", 2), V.UINT32: ("<I", 4),
             V.INT32: ("<i", 4), V.FLOAT32: ("<f", 4), V.BOOL: ("<?", 1), V.UINT64: ("<Q", 8), V.INT64: ("<q", 8),
             V.FLOAT64: ("<d", 8)}
    return V, Q, {int(k): v for k, v in fixed.items()}


class _Reader:
    """A growing prefix of the file. `take(n)` returns the next n bytes; when it must fetch, it asks for
    min(window, lower bound of the header left) bytes, never less than n."""

    def __init__(self, get_range, budget):
        self.get_range, self.budget, self.window = get_range, budget, 1 << 20
        self.data, self.pos = bytearray(), 0
        self.kv_left = self.ti_left = 0
        self.pending = 0  # minimum bytes of the current element still due after the read in progress

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            floor = self.pos + n + self.pending + MIN_KV * self.kv_left + MIN_TENSOR_INFO * self.ti_left
            have = len(self.data)
            size = max(end - have, min(self.window, floor - have))
            if have + size > self.budget:
                raise HeaderError("over_budget", f"header needs more than {self.budget} bytes")
            try:
                chunk = self.get_range(have, have + size - 1)
            except HttpError as e:
                if e.status == 416:
                    raise HeaderError("truncated", "range past end of file") from None
                raise
            self.data += chunk
            self.window *= 2
            if len(self.data) < end:
                raise HeaderError("truncated", f"file ends at {len(self.data)} bytes, header needs {end}")
        out = bytes(self.data[self.pos:end])
        self.pos = end
        return out

    def unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def string(self, after: int, min_chars: int = 0) -> str:
        """u64 length + UTF-8; `after` is the minimum size of what follows inside the current element. Keys and tensor
        names are never empty (min_chars=1); string values may be."""
        self.pending = after + min_chars
        n = self.unpack("<Q", 8)
        if n > 1 << 24:
            raise HeaderError("corrupt", f"string of {n} bytes")
        self.pending = after
        return self.take(n).decode("utf-8", errors="replace")


def read_gguf_header(get_range, budget: int = DEFAULT_BUDGET) -> dict:
    V, Q, FIXED = _gguf_types()
    r = _Reader(get_range, budget)
    r.pending = 20
    if r.take(4) != GGUF_MAGIC:
        raise HeaderError("corrupt", "bad GGUF magic")
    r.pending = 16
    version = r.unpack("<I", 4)
    if version not in (2, 3):
        raise HeaderError("unsupported_version", f"GGUF v{version}")
    r.pending = 8
    tensor_count = r.unpack("<Q", 8)
    r.pending = 0
    kv_count = r.unpack("<Q", 8)
    if tensor_count > 10_000_000 or kv_count > 1_000_000:
        raise HeaderError("corrupt", f"{tensor_count} tensors, {kv_count} KV pairs")
    r.kv_left, r.ti_left = kv_count, tensor_count

    def min_size(t: int) -> int:
        return FIXED[t][1] if t in FIXED else 8 if t == V.STRING else 12

    def value(t: int, after: int = 0):
        if t in FIXED:
            r.pending = after
            return r.unpack(*FIXED[t])
        if t == V.STRING:
            return r.string(after)
        if t != V.ARRAY:
            raise HeaderError("corrupt", f"value type {t}")
        r.pending = after + 8
        et = r.unpack("<I", 4)
        r.pending = after
        count = r.unpack("<Q", 8)
        if et in FIXED and count > ARRAY_SUMMARY_OVER:  # fixed width: read the head, take the rest in one go
            fmt, size = FIXED[et]
            r.pending = after + size * (count - ARRAY_HEAD)
            head = list(struct.unpack("<" + fmt[1] * ARRAY_HEAD, r.take(size * ARRAY_HEAD)))
            r.pending = after
            r.take(size * (count - ARRAY_HEAD))
            return {"type": V(et).name, "count": count, "head": head}
        items = []
        for i in range(count):
            items.append(value(et, after + min_size(et) * (count - i - 1)))
        if count > ARRAY_SUMMARY_OVER:
            return {"type": V(et).name, "count": count, "head": items[:ARRAY_HEAD]}
        return items

    kv = {}
    for _ in range(kv_count):
        r.kv_left -= 1
        key = r.string(after=4 + 1, min_chars=1)
        r.pending = 1
        t = r.unpack("<I", 4)
        kv[key] = value(t)
        r.pending = 0
    tensors = []
    for _ in range(tensor_count):
        r.ti_left -= 1
        name = r.string(after=4 + 8 + 4 + 8, min_chars=1)
        r.pending = 8 + 4 + 8
        n_dims = r.unpack("<I", 4)
        if not 0 < n_dims <= 8:
            raise HeaderError("corrupt", f"tensor {name!r} has {n_dims} dims")
        r.pending = 4 + 8
        ne = list(struct.unpack(f"<{n_dims}Q", r.take(8 * n_dims)))
        r.pending = 8
        ggml = r.unpack("<I", 4)
        r.pending = 0
        offset = r.unpack("<Q", 8)
        try:
            type_name = Q(ggml).name
        except ValueError:
            raise HeaderError("unsupported_type", f"tensor {name!r} has ggml type {ggml}, unknown to gguf.constants") from None
        tensors.append({"name": name, "ne": ne, "type": type_name, "offset": offset})
    alignment = kv.get("general.alignment", 32)
    return {"kind": "gguf", "version": version, "kv": kv, "alignment": alignment, "tensor_count": tensor_count,
            "tensors": tensors, "header_bytes": r.pos, "bytes_read": len(r.data)}


def read_safetensors_header(get_range) -> dict:
    try:
        header, n = parse_safetensors_header(get_range)
    except EOFError as e:
        raise HeaderError("truncated", str(e)) from None
    except (ValueError, json.JSONDecodeError) as e:
        raise HeaderError("corrupt", str(e)) from None
    return {"kind": "safetensors", "header": header, "header_bytes": n, "bytes_read": n}


def read_header(client, repo: str, sha: str, part: str, budget: int = DEFAULT_BUDGET) -> dict:
    """The header record of one weight file. A FixtureClient answers from its stored record instead."""
    if hasattr(client, "header"):
        return client.header(part)
    url = client.file_url(repo, sha, part)
    get_range = lambda start, end: client.get_range(url, start, end)  # noqa: E731
    return read_gguf_header(get_range, budget) if part.endswith(".gguf") else read_safetensors_header(get_range)
