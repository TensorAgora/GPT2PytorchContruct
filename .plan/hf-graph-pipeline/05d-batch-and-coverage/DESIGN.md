# 05d DESIGN

## Ledger

`output/categories/ledger.jsonl`, append-only (flush + fsync per line, one lock): `{repo, sha, category, format, stage,
status, code, duration_s, header_bytes, requests}`. `stage` is where the row ended: discover, headers, resolve or write.
Lines pass through `http.redact`, so HF_TOKEN never lands there; a run function that raises becomes
`failed:exception_<Type>` and never stops the batch.

## Resume

A restart skips every repo whose latest line has a terminal status (ok, ok_assumed, partial, no_graph, skipped:*,
failed:*) without any request. `--refresh` re-reads model_info and re-runs only repos whose sha changed.

## Concurrency

One `Client(concurrency=K)` (its semaphore bounds requests) shared by a `ThreadPoolExecutor(K)` over jobs; backoff and
Retry-After live in the client. `--limit N` takes the first N jobs of the category × format round-robin
(`catalog.stratified`), so a smoke run samples every group.

## Coverage

Written by `--coverage` from the ledger's latest line per repo plus each report.json:

- `coverage.csv`: category, format, status, rows.
- `coverage_model_type.csv`: model_type, status, rows. The key comes from the fetched files when a report exists
  (`report.keys`: model_type, gguf architecture, library class), else from the catalog.
- `near_misses.csv`: key, descriptor, reason, missing_example, rows_unlockable. no_graph rows grouped by
  (key, descriptor tried, reason), most rows first.

## Backlog

- `descriptor_backlog.csv`: key, category, rows, rows_unlocked, catalog_rows. rows and rows_unlocked are the no_graph
  ledger rows with that key. catalog_rows predicts the whole catalog: unresolved jobs with that key, the fetched key
  replacing the catalog's where known.
- `category_keys.csv` recomputes the §2 table: category, rows in scope, distinct keys, top keys.
