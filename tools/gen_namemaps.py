"""python tools/gen_namemaps.py [--check]

Generates src/gpt2_pytorch/registry/namemaps/gguf.json, the HF <-> GGUF tensor-name tables, from the pinned gguf-py
(`gguf.constants`, `gguf.tensor_mapping`; MIT licence, (c) the llama.cpp authors). --check regenerates and fails when the
committed file differs (a different gguf version, or a hand edit).
"""
import json
import sys
from pathlib import Path

import gguf
from gguf import constants as C
from gguf.tensor_mapping import TensorNameMap

OUT = Path(__file__).resolve().parents[1] / "src/gpt2_pytorch/registry/namemaps/gguf.json"
META = {"gguf_version": "0.19.0", "llama_cpp_commit": "a83d68b562b19fba1f8836edc93eb76d9ca1ce25",
        "license": "MIT, (c) the llama.cpp authors (gguf-py)", "generated_by": "tools/gen_namemaps.py"}


def table() -> dict:
    import importlib.metadata
    assert importlib.metadata.version("gguf") == META["gguf_version"], f"gguf {importlib.metadata.version('gguf')} is not the pinned {META['gguf_version']}"
    t = lambda s: s.replace("{bid}", "{i}")  # noqa: E731
    hf, arch_hf = {}, {}
    for cfg in (TensorNameMap.mappings_cfg, TensorNameMap.block_mappings_cfg):
        for tensor, names in cfg.items():
            for n in names:
                hf.setdefault(t(n), set()).add(tensor.name)
    for arch, cfg in TensorNameMap.arch_block_mappings_cfg.items():
        arch_hf[C.MODEL_ARCH_NAMES[arch]] = {t(n): [tensor.name] for tensor, names in cfg.items() for n in names}
    return {"_meta": META,
            "tensor_names": {k.name: t(v) for k, v in sorted(C.TENSOR_NAMES.items(), key=lambda kv: kv[0].name)},
            "arch_tensors": {C.MODEL_ARCH_NAMES[a]: sorted(x.name for x in ts) for a, ts in C.MODEL_TENSORS.items()},
            "hf_to_tensor": {k: sorted(v) for k, v in sorted(hf.items())},
            "arch_hf_to_tensor": arch_hf}


def main() -> int:
    raw = json.dumps(table(), sort_keys=True, separators=(",", ":")) + "\n"
    if "--check" in sys.argv:
        same = OUT.exists() and OUT.read_text() == raw
        print("namemaps up to date" if same else f"{OUT} differs from gguf {gguf.__name__} {META['gguf_version']}")
        return 0 if same else 1
    OUT.write_text(raw)
    print(f"{OUT}: {len(raw)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
