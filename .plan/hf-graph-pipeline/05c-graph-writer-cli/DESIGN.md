# 05c DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`tools/model_graph.py` is the one-model local path (qwen3_5 only) that this generalizes; it writes `checkpoint {files, tensors, bound, unbound}`.

## Interfaces

- `run_one(job, out_root, client=None, reader=None) -> Result(status, paths)`.
- Files under `output/categories/<category>/<format>/<org>__<repo>/`: manifest.json, report.json, graph.json or graph.ref; shared `output/categories/graphs/<content-key>.json`.
- CLI: `python -m tools.hf_graph <url> [--revision R] [--out DIR]`, `--fixture DIR`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/hub/pipeline.py` | new | R-13, R-32, R-33 |
| `tools/hf_graph.py` | new | R-00 |
| `tests/test_pipeline.py` | new | R-13, R-32, R-33 |

## Approach

S1–S12 (§5) as plain function calls in run_one; each side exit returns its status. Content key = sha256 of canonical (descriptor@version, resolved hyper, scheme). graph.ref holds `checkpoint_key` overrides when only names differ (D9).

## Likely blockers

none
