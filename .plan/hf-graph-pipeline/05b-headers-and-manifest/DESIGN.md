# 05b DESIGN

## Current state

`checkpoint/safetensors_loader.py:9` parses a local safetensors header. No GGUF reader exists; `gguf` 0.19.0 is in the
`hub` extra (constants only). huggingface_hub has `get_safetensors_metadata` / `parse_safetensors_file_metadata`.

## Readers

**Safetensors: own 2-range reader** (D-08). `get_range(0, 7)` → N (u64 LE, sanity cap 100 MB), `get_range(8, 7 + N)` →
JSON. `bytes_read = 8 + N`, payload 0 by construction. The hub helpers expose no byte count, read no component
subfolders, and buy nothing over two range calls; 05b.C cross-checks one repo against `get_safetensors_metadata`.

**GGUF: lower-bound windows.** Layout: magic `GGUF`, u32 version (2 or 3; else `failed:unsupported_version`), u64
tensor_count, u64 kv_count, KV pairs (string key, u32 type, value; arrays = u32 type, u64 count, items), tensor infos
(string name, u32 n_dims, u64 × n_dims `ne`, u32 ggml type, u64 offset), padding to `general.alignment` (default 32),
data. The parser keeps a provable lower bound on the bytes left before the end of the tensor-info table (14 B per
remaining KV, 33 B per remaining tensor info, 8 B per remaining string-array item, the exact size of fixed-width items).
Each fetch is `min(window, lower bound)`, window 1 MiB doubling, so no request ever crosses into tensor data:
`bytes_read == header_bytes`, payload 0. Over the budget → `failed:over_budget`; a short read → `failed:truncated`;
bad magic or type → `failed:corrupt`. Arrays with more than 64 items are stored as `{type, count, head}` (head = first 8).
Type names come from `gguf.constants` (GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES). Multi-part GGUF: every
part's header is read. Never GGUFReader.

Provisional budget: 64 MiB per file (D-12) until 05d sets D3.

## Header record

`read_header(client, repo, sha, weight_part) -> dict`:
`{kind: "safetensors", header: {...verbatim}, bytes_read}` or
`{kind: "gguf", version, kv, alignment, tensors: [{name, ne, type, offset}], header_bytes, bytes_read}`.

## Manifest

| field | source |
|---|---|
| format, version | constants `tensormorph-tensor-manifest`, 1 |
| model.id, url, category, format, catalog_row | Job |
| model.revision_sha, gated, library_name | model_info |
| model.license, base_model | catalog row (`license`, `base_model`) |
| files[] path, size | model_info siblings (weights read + small files) |
| files[] kind | safetensors, gguf, config (config.json, params.json), index, model_index |
| files[] header_bytes, shard_of | header record; index path or logical GGUF name |
| config | small files verbatim, keyed by path |
| gguf | first part's version, kv, alignment |
| tensors[] key, file, dtype, shape, ne, numel, nbytes, offset_begin, offset_end | header record; GGUF `shape = ne[::-1]`, nbytes from GGML_QUANT_SIZES, offsets absolute |
| safetensors_metadata | each file's `__metadata__` |
| stats | counts and sums over tensors; header_bytes_read = Σ bytes_read; payload_bytes_read = 0 (asserted) |
| provenance | fetched_at (UTC ISO, the only volatile field), tool_version, urls (resolve/<sha>), catalog_sources (row `sources`) |
| diagnostics | e.g. duplicate tensor key across shards, unread index shard |

`dump(obj) -> bytes`: `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"`.

## Fixtures

`tests/fixtures/headers/<org>__<repo>@<sha>/` (D-15, JSON only, never weight bytes): `model_info.json` (sha, gated,
siblings, library_name, pipeline_tag, tags), the small files at their repo paths, `headers/<part>.json` (the header record;
`.json.gz` when over 32 KiB) and `provenance.json`. `FixtureClient(dir)` serves `model_info`, `get_small` and `header`,
so the whole pipeline runs offline; `fixture_reader(dir)` rebuilds safetensors header bytes for range tests.
