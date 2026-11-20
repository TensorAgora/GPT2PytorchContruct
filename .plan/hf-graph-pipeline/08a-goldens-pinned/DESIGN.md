# 08a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

Goldens: distilgpt2 (tools/viz_graph.py), qwen2.5, mistral, phi-3.5, gemma-2, deepseek-v3, mixtral (arch.mjs, from mock.js configs), qwen3.8 (tools/model_graph.py). gemma-2 is gated; mixtral is not in the catalog.

## Interfaces

- `csv/goldens.csv` columns `repo,golden,format,revision`.
- `tests/test_goldens.py` parametrized over csv/goldens.csv.

## Files to change

| path | change | requirement |
|---|---|---|
| `csv/goldens.csv` | fill revisions | R-36 |
| `tests/fixtures/headers/` | one folder per golden | R-36 |
| `tests/test_goldens.py` | new | R-11 |

## Approach

Capture fixtures at pinned shas; diff real config.json against the spelled config; run `tools.hf_graph --fixture` then `graph_check --against` (structural for all 8; distilgpt2 byte identity is 06a's, D-17). Expected deltas (e.g. rope scaling, max positions) are listed, not hidden.

## Likely blockers

- gemma-2 gated: skip with a message.
