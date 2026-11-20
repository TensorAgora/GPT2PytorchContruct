# 07c SCOPE: Delta rule, Mamba, layer selectors, scan annotation, hybrid descriptors

Kind feature, wave 3, complexity L. Master sections §5.3 (D8), §4 (§7.2, §15), §7 (Wave 3).
Depends on 07a, 07b; blocks 07d, 06f. Progress: [TASKS.md](TASKS.md).

## Goal

Make delta-rule and Mamba layers selectable blocks with scan annotations and cover the top decoder_hybrid keys.

## Invariant

qwen3.8-27b-fp8 stays structurally equal; recurrence stays unrolled with scan annotations.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-42 | Delta-rule and Mamba blocks are selected per layer (layer_types, hybrid_override_pattern), unrolled steps carry execution {kind: scan, axis: T, step, body}, the qwen3.8-27b-fp8 golden stays structurally equal, and descriptors cover the top decoder_hybrid keys | §5.3 (D8), §7 (Wave 3) |

## Scope and non-goals

- R-42: delta-rule and Mamba blocks, selectors, scan annotation, hybrid descriptors.

Non-goals:

- A scan primitive (D8 alternative)

## Write set

- `src/gpt2_pytorch/registry/blocks.py`
- `src/gpt2_pytorch/registry/descriptors/`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>...
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- delta-rule and Mamba blocks
- hybrid descriptors

| criterion | verified by |
|---|---|
| qwen3.8 golden stays equal with scan annotations, and hybrid coverage rises | 07c.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
