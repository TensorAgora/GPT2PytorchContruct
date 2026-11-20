# 09b DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`nodes/tel.js` printFlat/build/decodeExpr (JS, TensormorphDesignUI); `decodeExpr` reads `contract(…, over=…)` as one axis.

## Interfaces

- `print_flat(graph) -> (source: str, sigma: dict)`; `build(source, sigma) -> graph` in `src/gpt2_pytorch/tensorprog/tel_flat.py`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/tensorprog/tel_flat.py` | new | R-51 |
| `tools/tel_roundtrip.cjs` | --emit-flat | R-51 |
| `tests/test_tel_flat.py` | new | R-51 |

## Approach

Line-by-line port with tel.js as oracle (text equality), then decode back and diff with the same field comparison tel_roundtrip uses.

## Likely blockers

none
