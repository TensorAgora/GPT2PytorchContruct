import json
import struct

import pytest

gguf = pytest.importorskip("gguf")
import numpy as np  # noqa: E402

from gpt2_pytorch.hub.headers import HeaderError, read_gguf_header, read_safetensors_header  # noqa: E402


class Ranges:
    """get_range over in-memory bytes, recording the furthest byte asked for."""

    def __init__(self, data: bytes):
        self.data, self.max_end, self.calls = data, -1, 0

    def __call__(self, start, end):
        self.calls += 1
        self.max_end = max(self.max_end, end)
        return self.data[start:end + 1]


def safetensors_bytes(payload=1000):
    header = {"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [10, 25], "data_offsets": [0, payload]}}
    body = json.dumps(header).encode()
    return struct.pack("<Q", len(body)) + body + b"\x01" * payload, len(body)


def test_safetensors_reads_header_only():
    data, n = safetensors_bytes()
    r = Ranges(data)
    rec = read_safetensors_header(r)
    assert rec["header"]["w"]["shape"] == [10, 25] and rec["bytes_read"] == 8 + n
    assert r.max_end == 7 + n and r.calls == 2  # never a payload byte


def test_safetensors_truncated():
    data, n = safetensors_bytes()
    with pytest.raises(HeaderError) as e:
        read_safetensors_header(Ranges(data[: 8 + n // 2]))
    assert e.value.code == "truncated"


def write_gguf(path, tokens=5000):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_block_count(2)
    w.add_embedding_length(16)
    w.add_token_list([f"tok{i}" if i % 97 else "" for i in range(tokens)])  # empty strings are legal values
    w.add_token_scores([float(i) for i in range(tokens)])
    w.add_tensor("token_embd.weight", np.zeros((tokens, 16), np.float32))
    w.add_tensor("blk.0.attn_k.weight", np.zeros((4, 16), np.float16))
    w.add_tensor("blk.0.ffn_gate_exps.weight", np.zeros((8, 6, 16), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path.read_bytes()


def test_gguf_header_never_reaches_tensor_data(tmp_path):
    data = write_gguf(tmp_path / "m.gguf")
    r = Ranges(data)
    rec = read_gguf_header(r)
    assert rec["tensor_count"] == len(rec["tensors"]) == 3
    assert rec["bytes_read"] == rec["header_bytes"] and r.max_end == rec["header_bytes"] - 1
    assert rec["header_bytes"] <= len(data) - 4 * 5000 * 16  # the token_embd payload was never touched
    tokens = rec["kv"]["tokenizer.ggml.tokens"]
    assert tokens == {"type": "STRING", "count": 5000, "head": ["", "tok1", "tok2", "tok3", "tok4", "tok5", "tok6", "tok7"]}
    assert rec["kv"]["tokenizer.ggml.scores"]["count"] == 5000 and rec["kv"]["llama.block_count"] == 2
    t = {x["name"]: x for x in rec["tensors"]}
    assert t["blk.0.ffn_gate_exps.weight"]["ne"] == [16, 6, 8]  # innermost first; shape = ne[::-1]
    assert t["blk.0.attn_k.weight"]["type"] == "F16"


def test_gguf_truncated_and_corrupt(tmp_path):
    data = write_gguf(tmp_path / "m.gguf", tokens=200)
    full = read_gguf_header(Ranges(data))["header_bytes"]
    with pytest.raises(HeaderError) as e:
        read_gguf_header(Ranges(data[: full // 2]))
    assert e.value.code == "truncated"
    with pytest.raises(HeaderError) as e:
        read_gguf_header(Ranges(b"GGML" + data[4:]))
    assert e.value.code == "corrupt"
    with pytest.raises(HeaderError) as e:
        read_gguf_header(Ranges(data), budget=64)
    assert e.value.code == "over_budget"
