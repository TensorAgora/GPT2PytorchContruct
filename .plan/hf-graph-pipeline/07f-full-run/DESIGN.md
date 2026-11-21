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

## Observed

- Run 9 (`--all`, concurrency 8, RSS watchdog) covered 17,751 rows in 3.8 h (77 rows/min; HF 429 limits it). The final
  pass re-resolved the 12,793 registry-bound rows on registry 8571cf96ee37 in 81 min with 0 requests per row (same shas,
  manifests reused offline).
- Final ledger: 29,780/29,780 jobs terminal, 0 exception rows. ok 9,003, partial 4,965, ok_assumed 189, no_graph 5,898,
  skipped 9,076 (6,407 deferred formats, 1,708 remote code, 738 no weights, 223 gated), failed 649 (431 index_401).
- The 21 exception rows of run 9 were workers still on code from before the `keys_of` fix, plus one EXL3 repo whose int8
  embed (qweight [V, D] + scales [V]) hit the GPTQ rule; the rule now needs 2-D qweight and scales.
- `--readme`: 21 category READMEs, each total equal to coverage.csv. `--all` again: 0 rows, 0 header bytes.
- TEL coverage over 7,356 stored graphs: 14.16M ops, 9.56M spellable; 4.28M are Opaque ops in 5,616 path-B graphs (D5).
- Storage: output/ about 3.2 GB after pruning (graphs 1.3 GB, gzipped header manifests 1.3 GB); no weight files.
