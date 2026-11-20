# 07f DESIGN

Drafted at plan time; completed in Stage A.

## Current state

Batch, coverage and backlog exist from 05d; descriptors from waves 1–6.

## Interfaces

- `python -m tools.hf_graph --readme` writes `output/categories/<category>/README.md`.

## Files to change

| path | change | requirement |
|---|---|---|
| `tools/hf_graph.py` | --readme | R-47 |

## Approach

Run, coverage, readme, round trip; then repeat `--all` to prove the no-op.

## Likely blockers

- Run length: resumable; may span sessions.
