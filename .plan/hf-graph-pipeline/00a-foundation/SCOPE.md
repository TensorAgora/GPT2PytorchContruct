# 00a SCOPE: Confirm inputs, extras, gitignore, viz-graph schema, checker and comparator

Kind foundation, wave 0, complexity M. Master sections §2, §3, §8.1, §8.2, D6.
Depends on none; blocks 05a, 05b, 06a. Progress: [TASKS.md](TASKS.md).

## Goal

Confirm every input path, add the hub/trace extras, ignore output/, and ship tools/graph_check.py: viz-graph v1 schema, §8.2 invariants and the §8.1 structural comparator every later module verifies with.

## Invariant

The 8 files in `json/` pass `tools/graph_check.py` unchanged; the checker never mutates its input.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-17 | The first task confirms every §2 input path; a missing or moved path is recorded in HARNESS.md, never assumed | §2 |
| R-18 | Python work uses uv and the existing .venv (never conda); new dependencies go in optional extras only: `hub` (gguf, jsonschema) and `trace` (transformers, timm, diffusers) | §3 |
| R-19 | `output/` is git-ignored; git holds test fixtures only | D6 |
| R-20 | A viz-graph v1 JSON Schema written from the 8 goldens plus a checker enforce the §8.2 graph invariants (schema, acyclic, every op input produced earlier or an input/parameter, parameter shape equals header shape, totals recompute exactly); all 8 goldens pass | §8.2 |
| R-21 | A structural comparator reports every difference between two graphs over the §8.1 fields (dims, parameter set and shapes, checkpoint_key or view, op/edge/multiplication counts, totals) | §8.1 |

## Scope and non-goals

- R-17: confirm every §2 input path; corrections go to HARNESS.md.
- R-18: `hub` and `trace` optional extras in pyproject.toml; install with `uv sync --inexact`.
- R-19: `output/` in .gitignore.
- R-20: viz-graph v1 JSON Schema from the goldens plus the §8.2 invariant checker.
- R-21: `--against` structural comparator over the §8.1 fields.

Non-goals:

- Manifest schema (05b)
- Golden pinning (08a)

## Write set

- `.gitignore`
- `pyproject.toml`
- `uv.lock`
- `src/gpt2_pytorch/schemas/viz-graph-v1.schema.json`
- `tools/graph_check.py`
- `tests/test_graph_check.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.graph_check G.json [--against json/<golden>.json] [--manifest M.json]
uv sync --inexact --extra hub --extra dev
```

## Deliverables and acceptance

- tools/graph_check.py
- src/gpt2_pytorch/schemas/viz-graph-v1.schema.json
- pyproject.toml extras

| criterion | verified by |
|---|---|
| Every §2 input is confirmed or recorded in HARNESS.md | 00a.A.P1.T1 |
| The hub extra imports and output/ is ignored | 00a.B.P1.T2 |
| All 8 goldens pass schema and §8.2 invariants; a mutated golden fails naming the op | 00a.C.P1.T1 |
| A golden compared with itself gives 0 differences and with another golden lists them | 00a.B.P2.T3 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A golden fails the schema written from the goldens: the schema is wrong, fix it, never the golden.
