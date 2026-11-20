# 06e DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`tools/model_graph.py` names FP8 `*_scale_inv` in the principal's `view` for qwen3_5 only.

## Interfaces

- `match_sidecars(stored, principals) -> list[Sidecar(tensor, principal, scheme)]`.
- report.json `sidecars: [{tensor, principal, scheme}]`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/sidecars.py` | new | R-07 |
| `src/gpt2_pytorch/registry/bind.py` | aux class via sidecars | R-07 |
| `src/gpt2_pytorch/registry/descriptors/qwen3_5.json` | drop ad-hoc views | R-07 |
| `tests/test_sidecars.py` | new | R-07 |

## Approach

One rule table (pattern, principal derivation, shape relation) per scheme from design doc §7.4; e.g. FP8: `X.weight_scale_inv` shape == ceil(X.weight.shape / 128).

## Likely blockers

none
