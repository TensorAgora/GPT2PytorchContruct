# 05d DESIGN

Drafted at plan time; completed in Stage A.

## Current state

§2 gives catalog-level numbers (23,372 rows in scope; 671 gated; 1,844 remote code; 1,870 truly unkeyed). No ledger or coverage code exists.

## Interfaces

- ledger.jsonl line: `{repo, sha, stage, status, code, duration_s}`.
- coverage.csv `category,format,status,rows`; coverage_model_type.csv `model_type,status,rows`; near_misses.csv `key,descriptor,missing,rows_unlockable`; descriptor_backlog.csv `key,category,rows,rows_unlocked`.

## Files to change

| path | change | requirement |
|---|---|---|
| `tools/hf_graph.py` | --all, --coverage | R-14, R-15 |
| `src/gpt2_pytorch/hub/batch.py` | new | R-14, R-15, R-35 |
| `tests/test_batch.py` | new | R-14, R-15, R-35 |
| `.plan/hf-graph-pipeline/DECISIONS.md` | D3 row | R-34 |

## Approach

Thread pool of `concurrency` workers over jobs; each finished row appends one ledger line (flush + fsync). Resume reads the ledger first. Coverage groups ledger rows by the latest status per repo.

## Likely blockers

- Hub throughput unknown until the smoke run.
