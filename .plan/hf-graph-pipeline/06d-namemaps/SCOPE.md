# 06d SCOPE: Generated HF↔GGUF name maps with a storage-transform overlay

Kind feature, wave 1, complexity M. Master sections §4 (§7.6), §6.
Depends on 06a, 08a; blocks 07a. Progress: [TASKS.md](TASKS.md).

## Goal

Generate HF↔GGUF name tables from pinned gguf-py and record GGUF storage transforms in an overlay, replacing hand-listed GGUF names.

## Invariant

Generated name maps regenerate byte-identically from the pinned gguf-py; goldens stay equal.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-37 | HF↔GGUF name maps are generated from a pinned gguf-py (commit and MIT notice kept) plus an overlay of storage transforms (llama Q/K rope permutation checked at the pinned commit, fused QKV split, gate_up split, Gemma norm +1) | §4 (§7.6), §6 |

## Scope and non-goals

- R-37: generated HF↔GGUF tables, overlay of storage transforms, descriptors switched to them.

Non-goals:

- Undoing transforms on weight values (no weights are read)

## Write set

- `tools/gen_namemaps.py`
- `src/gpt2_pytorch/registry/namemaps/`
- `src/gpt2_pytorch/registry/descriptors/`
- `tests/test_namemaps.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python tools/gen_namemaps.py [--check]
```

## Deliverables and acceptance

- tools/gen_namemaps.py
- src/gpt2_pytorch/registry/namemaps/

| criterion | verified by |
|---|---|
| Generated maps regenerate identically and the overlay records the four transforms | 06d.B.P1.T2 |
| HF and GGUF schemes agree on canonical names for llama and mixtral | 06d.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- The pinned gguf version lacks a tensor a descriptor needs: add it to overlay.json with a comment, never edit the generated file.
