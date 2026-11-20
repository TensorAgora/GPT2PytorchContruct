# 07c DESIGN

Drafted at plan time; completed in Stage A.

## Current state

qwen3_5.py unrolls the delta rule at --seq T (70 contractions per linear-attention layer at T=8).

## Interfaces

- op `execution: {kind: 'scan', axis: 'T', step: i, body: <group id>}`.
- blocks `delta_rule(...)` (wraps qwen3_5.py), `mamba(...)`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/blocks.py` | delta_rule, mamba | R-42 |
| `src/gpt2_pytorch/registry/descriptors/` | hybrid descriptors | R-42 |

## Approach

Selectors from 06a (`list(layer_types)`, `pattern(hybrid_override_pattern)`) choose the block per layer; each unrolled step's ops get the scan tag.

## Likely blockers

none
