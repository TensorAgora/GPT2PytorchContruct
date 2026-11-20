# 05b DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`checkpoint/safetensors_loader.py:9` `read_safetensors_header(path)` parses a local file's header. No GGUF reader exists; the `gguf` package (0.19.0) is not in .venv until 00a. huggingface_hub has `get_safetensors_metadata` and `parse_safetensors_file_metadata` (no byte count, root files only).

## Interfaces

- `read_safetensors_header(get_range) -> (header: dict, bytes_read: int)`.
- `read_gguf_header(get_range, budget) -> (gguf: dict, tensors: list, bytes_read)`, raises `HeaderError(code)` with code truncated|corrupt|over_budget.
- `build_manifest(job, discovery, small_files, headers) -> dict`; `validate_manifest(m)`; `dump(obj) -> bytes` (canonical).
- `capture(repo, sha)`; `fixture_reader(dir) -> get_range`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/hub/headers.py` | new | R-04, R-05 |
| `src/gpt2_pytorch/hub/manifest.py` | new | R-06 |
| `src/gpt2_pytorch/hub/fixtures.py` | new | R-25 |
| `src/gpt2_pytorch/schemas/tensor-manifest-v1.schema.json` | new | R-06 |
| `tests/test_headers.py` | new | R-04, R-05 |
| `tests/test_manifest.py` | new | R-06 |
| `tests/fixtures/headers/` | captured fixtures | R-25 |

## Approach

Safetensors (§5.1): range 0-7 → u64 LE N; range 8..7+N → JSON. GGUF (§5.1): read magic/version/counts, then KV in 1 MiB windows doubling up to the budget, then tensor infos; arrays longer than 64 entries stored as {type, count, head (first 8)}. Manifest (§5.2): Observed only; `ne` as read, `shape = ne[::-1]`; tensors sorted (file, key).
Fixtures (D-15): JSON only; the GGUF byte reader is tested on a synthetic header from gguf.GGUFWriter in tmp_path.
Rejected: GGUFReader (memory-maps a whole local file); raw header bytes in git (storage rule).

## Likely blockers

- D3 not yet measured: provisional 64 MiB GGUF budget (DECISIONS D-12).
