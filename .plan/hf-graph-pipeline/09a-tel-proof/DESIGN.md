# 09a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`tools/tel_roundtrip.cjs` (untracked) takes one graph and the flags --const-as-buffer and --fix-contract; on mistral it gives diff_total 0 and D3 pass (checked at planning). `tools/viz_to_tel.py` + `tools/tel_compile.mjs` give PASS on distilgpt2.

## Interfaces

- `node tools/tel_roundtrip.cjs <graph|dir>... [--const-as-buffer] [--fix-contract]`: one JSON line per graph plus `{"summary":…}`.
- `python -m tools.tel_coverage <graph>...`: one row per graph `name, ops, spellable, routed_moe, recurrence, list_constants, concrete_axes, opaque`.

## Files to change

| path | change | requirement |
|---|---|---|
| `docs/GRAPH_TEL.md` | new | R-16 |
| `tools/tel_roundtrip.cjs` | multi-graph + summary | R-16 |
| `tools/tel_coverage.py` | new | R-49 |

## Approach

Design doc condenses §9.1–§9.6 with verified ADR-0027 store naming. Round trip and coverage are read-only over graphs.

## Likely blockers

- tel_roundtrip.cjs untracked: commit it first.
