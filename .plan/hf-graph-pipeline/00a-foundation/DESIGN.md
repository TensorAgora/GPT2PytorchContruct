# 00a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

No graph schema or checker exists. `tools/viz_graph.py` and `tools/model_graph.py` write viz-graphs; `json/*.json` (8 files, 0.3 to 9.2 MB) are the oracles. pyproject.toml has extras `dev`, `reference`, `export`; .gitignore does not list `output/`.

## Interfaces

- `python -m tools.graph_check FILE... [--against GOLDEN] [--manifest M]`: prints `PASS <file>` or one line per failure/difference; exit 0 or 1.
- `check(graph, manifest=None) -> list[str]` and `compare(a, b) -> list[str]` importable from `tools.graph_check`.
- `src/gpt2_pytorch/schemas/viz-graph-v1.schema.json` (draft 2020-12).

## Files to change

| path | change | requirement |
|---|---|---|
| `.gitignore` | add `output/` | R-19 |
| `pyproject.toml` | extras hub, trace | R-18 |
| `uv.lock` | relock | R-18 |
| `src/gpt2_pytorch/schemas/viz-graph-v1.schema.json` | new | R-20 |
| `tools/graph_check.py` | new | R-20, R-21 |
| `tests/test_graph_check.py` | new | R-20, R-21 |

## Approach

1. Confirm inputs (§2). 2. Extras and gitignore. 3. Schema: union of the goldens' keys; §5.3 additive keys optional. 4. Invariants: topological pass over `ops[]`, inputs must be produced earlier or be graph inputs/parameters/constants; recompute `totals` from `multiplications[]`. 5. Comparator: compare only keys present in the golden, so additive fields never count as differences.
Rejected: comparing whole JSON (fails on additive fields and key order).

## Likely blockers

- `uv sync` removes transformers: use `--inexact`.
