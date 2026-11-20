# 07b DESIGN

Drafted at plan time; completed in Stage A.

## Current state

Goldens draw every expert densely (mixtral 128/385 multiplications over E; deepseek-v3 232/1,026).

## Interfaces

- op/multiplication `execution: {kind: 'routed', of: 'E', active: K, router: <op index>}`.
- `totals.contraction_flops`, `totals.active_flops`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/blocks.py` | experts annotations | R-40 |
| `src/gpt2_pytorch/registry/descriptors/` | MoE descriptors | R-41 |
| `src/gpt2_pytorch/registry/namemaps/overlay.json` | stack/split for expert forms | R-41 |
| `tests/test_moe_storage.py` | new | R-41 |

## Approach

active_flops = contraction_flops − (E − K)/E × expert flops (exact formula and hand counts in Stage A).

## Likely blockers

none
