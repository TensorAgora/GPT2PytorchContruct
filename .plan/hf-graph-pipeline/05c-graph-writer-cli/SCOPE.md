# 05c SCOPE: run_one, graph and report writers, content store, single-URL CLI

Kind slice, wave 0, complexity M. Master sections §1, §5, §5.3, §5.4, D9.
Depends on 05a, 05b, 06c, 06a; blocks 05d, 08a. Progress: [TASKS.md](TASKS.md).

## Goal

Thin end-to-end slice: one URL (or a fixture dir) → manifest.json, report.json and a content-addressed graph through `python -m tools.hf_graph`.

## Invariant

Two runs on the same (repo, sha, registry version) write byte-identical files apart from `fetched_at`.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-00 | `python -m tools.hf_graph <huggingface_url> [--revision R]` writes manifest.json, report.json and graph.json for one repo, and the batch runner calls this same function | §1, §5 |
| R-13 | graph.json is viz-graph v1 plus only the §5.3 additive fields (checkpoint, provenance, components, execution), every parameter names a checkpoint_key or view, report.json is written, and graphs are stored once per content key with per-repo graph.ref (D9) | §1, §5.3, §5.4, D9 |
| R-32 | The same (repo, sha, registry version) produces byte-identical manifest.json, report.json and graph.json, `fetched_at` excluded | §3, §8.3 (C6) |
| R-33 | Each repo writes to output/categories/<category>/<format>/<org>__<repo>/ with a §5.4 status (ok, ok_assumed, partial, no_graph, skipped:*, failed:<code>); no_graph writes the tensor inventory and prefix tree | §5, §5.4 |

## Scope and non-goals

- R-00: `python -m tools.hf_graph <url>` and `--fixture DIR` over one `run_one`.
- R-13: graph.json with §5.3 additive fields, content store and graph.ref.
- R-32: canonical JSON everywhere.
- R-33: §5.4 paths and statuses, no_graph inventory.

Non-goals:

- Batch loop and coverage (05d)

## Write set

- `src/gpt2_pytorch/hub/pipeline.py`
- `tools/hf_graph.py`
- `tests/test_pipeline.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
.venv/bin/python -m tools.graph_check G.json [--against json/<golden>.json] [--manifest M.json]
```

## Deliverables and acceptance

- src/gpt2_pytorch/hub/pipeline.py
- tools/hf_graph.py

| criterion | verified by |
|---|---|
| Qwen2.5-7B-Instruct URL writes all three outputs and the graph passes §8.2 | 05c.C.P1.T1 |
| Identical configs share one stored graph | 05c.B.P1.T2 |
| Outputs are byte-identical across runs apart from fetched_at | 05c.B.P1.T3 |
| Unknown architectures end as no_graph with an inventory | 05c.B.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A graph field outside viz-graph v1 + §5.3 seems needed: stop; D8 is the owner's call.
