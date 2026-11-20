# 06f DESIGN

Drafted at plan time; completed in Stage A.

## Current state

torch 2.14.1 and transformers 5.23.0 in .venv; timm and diffusers missing. `examples/fx_trace.py` and `tools/matmul_graph.py` already trace GPT-2 with fx.

## Interfaces

- `trace(config, library) -> (graph, expected)`; op `{primitive: 'opaque', fn: <aten>, ports}` for unmapped ATen ops.

## Files to change

| path | change | requirement |
|---|---|---|
| `pyproject.toml` | trace extra used | R-12 |
| `uv.lock` | relock | R-12 |
| `src/gpt2_pytorch/registry/trace.py` | new | R-12 |
| `src/gpt2_pytorch/hub/pipeline.py` | path B fallback | R-12 |
| `tools/conformance.py` | --c2 | R-45 |
| `tests/test_trace.py` | new | R-12 |
| `tests/test_pipeline.py` | path_b case | R-12 |

## Approach

Reuse the fx/export approach of `tools/matmul_graph.py`; ATen names via OPS/generated/pytorch-tel-mapping.md.

## Likely blockers

- Data-dependent MoE traces: MoE stays on path A.
