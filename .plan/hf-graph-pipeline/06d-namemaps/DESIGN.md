# 06d DESIGN

Drafted at plan time; completed in Stage A.

## Current state

mixtral's GGUF names are hand-listed (06b). gguf 0.19.0 ships `gguf.constants` and `gguf.tensor_mapping`; llama.cpp's converter permutes Q/K rows for llama models.

## Interfaces

- `registry/namemaps/gguf.json`: `{arch: {gguf_name_pattern: hf_name_pattern}}` with `_meta {gguf_version, commit, license}`.
- `registry/namemaps/overlay.json`: `{arch: {pattern: {transform, args}}}`.

## Files to change

| path | change | requirement |
|---|---|---|
| `tools/gen_namemaps.py` | new | R-37 |
| `src/gpt2_pytorch/registry/namemaps/` | gguf.json, overlay.json, NOTICE | R-37 |
| `src/gpt2_pytorch/registry/descriptors/` | schemes.gguf → generated | R-37 |
| `tests/test_namemaps.py` | new | R-37 |

## Approach

Generate from tensor_mapping's block mappings per MODEL_ARCH; keep the MIT notice. Overlay transforms: rope_permute (attn_q, attn_k), split (fused qkv, gate_up), offset (Gemma norm +1).

## Likely blockers

none
