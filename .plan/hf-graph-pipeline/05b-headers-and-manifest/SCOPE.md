# 05b SCOPE: Safetensors and GGUF header readers, fixtures, tensor manifest v1

Kind feature, wave 0, complexity L. Master sections §3, §5.1, §5.2, §8.3.
Depends on 05a, 00a; blocks 06c, 05c, 08a. Progress: [TASKS.md](TASKS.md).

## Goal

Read safetensors and GGUF headers by range with 0 payload bytes, capture offline header fixtures, and write a validated, deterministic manifest.json.

## Invariant

`payload_bytes_read == 0` for every manifest; a reader that would cross the header end fails instead.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-04 | Safetensors headers of every shard are read by HTTP range (bytes 0-7, then 8..7+N) with 0 payload bytes; tensor count equals the index `weight_map` | §1, §3, §5.1 |
| R-05 | GGUF headers (KV and tensor infos) are read by incremental HTTP range (1 MiB windows, doubling) within the D3 byte budget, large arrays stored as {type, count, head}, types from `gguf.constants`, never `GGUFReader`; tensor count equals `tensor_count` | §1, §3, §5.1 |
| R-06 | `manifest.json` follows tensor-manifest v1 (§5.2) with a JSON Schema and a validator, holds Observed facts only, stores GGUF `ne` as read plus `shape` reversed and llama.cpp names as stored, and a re-run is byte-identical apart from `fetched_at` | §1, §5.2 |
| R-25 | Header fixtures live in tests/fixtures/headers/<org>__<repo>@<sha>/ (config.json, index, raw header bytes, provenance) and tests run offline from them | §8.3 |

## Scope and non-goals

- R-04: safetensors 2-range reader, every shard.
- R-05: GGUF incremental reader with array summaries and a byte budget.
- R-06: manifest v1 schema, builder, validator; canonical JSON.
- R-25: fixture capture and offline fixture reader.

Non-goals:

- Sidecar binding (06e)
- Anything architecture-dependent in the manifest

## Write set

- `src/gpt2_pytorch/hub/headers.py`
- `src/gpt2_pytorch/hub/manifest.py`
- `src/gpt2_pytorch/hub/fixtures.py`
- `src/gpt2_pytorch/schemas/tensor-manifest-v1.schema.json`
- `tests/test_headers.py`
- `tests/test_manifest.py`
- `tests/fixtures/headers/`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m gpt2_pytorch.hub.fixtures <repo> [--revision SHA]      # [network]
```

## Deliverables and acceptance

- src/gpt2_pytorch/hub/headers.py
- src/gpt2_pytorch/hub/manifest.py
- src/gpt2_pytorch/hub/fixtures.py
- src/gpt2_pytorch/schemas/tensor-manifest-v1.schema.json
- tests/fixtures/headers/

| criterion | verified by |
|---|---|
| Sharded safetensors and GGUF real headers read with 0 payload bytes and matching tensor counts | 05b.C.P1.T1 |
| Manifests validate and rebuild byte-identically apart from fetched_at | 05b.B.P3.T2 |
| Fixture folders exist and tests read them offline | 05b.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A GGUF header exceeds the provisional budget on a real repo: record its size in HARNESS.md, return failed:over_budget, continue.
