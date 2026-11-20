# 00a DESIGN

Drafted at plan time; completed in Stage A.

## Inputs

Confirmed 2026-11-21 with `test -e` (all exist).

| input | path |
|---|---|
| Model catalog | `/Users/thanh/Tensormorph/GPT2PytorchContruct/csv/categories/all_models.csv` |
| Model catalog index | `/Users/thanh/Tensormorph/GPT2PytorchContruct/csv/categories/index.csv` |
| Catalog README | `/Users/thanh/Tensormorph/GPT2PytorchContruct/csv/categories/README.md` |
| Catalog per category/format (example) | `/Users/thanh/Tensormorph/GPT2PytorchContruct/csv/categories/decoder_dense/safetensors.csv` |
| Metadata spec (21 categories; example) | `/Users/thanh/Tensormorph/GPT2PytorchContruct/csv/categories/decoder_dense/metadata_spec.json` |
| Catalog generator | `/Users/thanh/Tensormorph/GPT2PytorchContruct/tools/split_model_catalog.py` |
| Target output examples (8) | `/Users/thanh/Tensormorph/GPT2PytorchContruct/json` |
| Design reference | `/Users/thanh/Tensormorph/GPT2PytorchContruct/open-weight-model-graphs.md` |
| Program builder | `/Users/thanh/Tensormorph/GPT2PytorchContruct/src/gpt2_pytorch/tensorprog/program.py` |
| viz_graph() | `/Users/thanh/Tensormorph/GPT2PytorchContruct/src/gpt2_pytorch/tensorprog/viz.py` |
| qwen3_5 lowering | `/Users/thanh/Tensormorph/GPT2PytorchContruct/src/gpt2_pytorch/tensorprog/qwen3_5.py` |
| GPT-2 lowering | `/Users/thanh/Tensormorph/GPT2PytorchContruct/src/gpt2_pytorch/tensorprog/lower.py` |
| read_safetensors_header | `/Users/thanh/Tensormorph/GPT2PytorchContruct/src/gpt2_pytorch/checkpoint/safetensors_loader.py` |
| CLI model_graph | `/Users/thanh/Tensormorph/GPT2PytorchContruct/tools/model_graph.py` |
| CLI viz_graph | `/Users/thanh/Tensormorph/GPT2PytorchContruct/tools/viz_graph.py` |
| CLI viz_to_tel | `/Users/thanh/Tensormorph/GPT2PytorchContruct/tools/viz_to_tel.py` |
| CLI tel_compile | `/Users/thanh/Tensormorph/GPT2PytorchContruct/tools/tel_compile.mjs` |
| JS architecture library | `/Users/thanh/Tensormorph/TensormorphDesignUI/tensormorph/ui/activity_bar/viewport/graphs/arch.mjs` |
| Spelled configs (MOVED: master says viewport/shared/mock.js) | `/Users/thanh/Tensormorph/TensormorphDesignUI/tensormorph/ui/activity_bar/shared/mock.js` |
| TensorNodes T♭ bridge | `/Users/thanh/Tensormorph/TensormorphDesignUI/tensormorph/ui/activity_bar/nodes/tel.js` |
| TEL operators (OPS) | `/Users/thanh/Tensormorph/tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/docs/operators` |
| TEL ADRs (ADR) | `/Users/thanh/Tensormorph/tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/docs/adr` |
| TEL architecture | `/Users/thanh/Tensormorph/tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/docs/architecture` |
| TEL packages (PKG) | `/Users/thanh/Tensormorph/tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/packages` |
| TEL dist (compiler) | `/Users/thanh/Tensormorph/tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/packages/tel/dist/index.js` |
| TEL bridge notes | `/Users/thanh/Tensormorph/GPT2PytorchContruct/docs/TEL_NEXT_STEPS.md` |

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
