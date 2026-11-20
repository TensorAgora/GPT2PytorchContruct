# 07d SCOPE: VLM assemblies, MTP and draft heads

Kind feature, wave 4, complexity M. Master sections §4 (§7.3), §5.3, §7 (Wave 4).
Depends on 07c, 07b; blocks 06f. Progress: [TASKS.md](TASKS.md).

## Goal

Bind wrapper checkpoints: text tower bound, other components bound or declared without a graph, never dropped.

## Invariant

No stored tensor of an assembly falls outside bound, tied, ignored, aux or component.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-43 | Assemblies bind the VLM text tower and mark vision towers, projectors, MTP and draft heads bound or declared_without_graph in `components`, never silently dropped; no stored tensor falls outside a class | §4 (§7.3), §5.3, §7 (Wave 4) |

## Scope and non-goals

- R-43: assembly descriptors, components field, partial status, VLM and draft-head coverage.

Non-goals:

- Graphs for vision towers (declared_without_graph unless a descriptor exists)

## Write set

- `src/gpt2_pytorch/registry/build.py`
- `src/gpt2_pytorch/registry/bind.py`
- `src/gpt2_pytorch/registry/descriptors/`
- `tests/test_assemblies.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>...
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- assembly support in build/bind
- VLM and draft-head descriptors

| criterion | verified by |
|---|---|
| No stored tensor falls outside a class on the VLM and draft rows | 07d.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
