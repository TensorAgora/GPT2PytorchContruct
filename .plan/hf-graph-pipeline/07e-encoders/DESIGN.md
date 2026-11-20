# 07e DESIGN

Drafted at plan time; completed in Stage A.

## Current state

No encoder lowering exists; encoder has 4,122 rows in scope, encoder_decoder 757.

## Interfaces

- blocks `layernorm_post`, `bidir_attention`, `gelu_ffn`, `rel_pos_bias`, `cross_attention`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/blocks.py` | encoder blocks | R-44 |
| `src/gpt2_pytorch/registry/descriptors/` | 10 descriptors | R-44 |

## Approach

C4 uses hand counts in descriptor provenance (no goldens, DECISIONS D-10).

## Likely blockers

none
