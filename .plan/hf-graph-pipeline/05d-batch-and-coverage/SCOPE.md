# 05d SCOPE: Batch ledger, resume, coverage reports, backlog, 100-row smoke

Kind feature, wave 0, complexity M. Master sections §2, §3, §5.4, §7, D3.
Depends on 05c; blocks 07a, 07f. Progress: [TASKS.md](TASKS.md).

## Goal

Loop run_one over jobs with a resumable ledger, write coverage and the descriptor backlog, and measure throughput on 100 rows to set D3.

## Invariant

The ledger is append-only and a restart never re-runs a row whose latest terminal status has the same sha.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-14 | The batch runs over all rows with an append-only ledger.jsonl that drives resume, bounded concurrency and backoff, HF_TOKEN optional and never logged; killing and restarting skips completed rows | §1, §5.4 |
| R-15 | Coverage reports coverage.csv (category × format × status), coverage_model_type.csv (model_type × status) and near_misses.csv ranked by unlockable rows, with totals matching the ledger | §1, §5.4 |
| R-34 | A 100-row smoke run measures throughput and header bytes per model before the full run is sized, and sets the D3 header byte budget | §3, §7 (Wave 0), D3 |
| R-35 | The §2 category/key table is recomputed from fetched data and the descriptor backlog is ordered by rows unlocked | §2 |

## Scope and non-goals

- R-14: `--all` batch with ledger, resume, concurrency.
- R-15: coverage CSVs and near-miss ranking.
- R-34: 100-row smoke measurement and D3.
- R-35: descriptor backlog and recomputed §2 table.

Non-goals:

- The full run (07f)

## Write set

- `tools/hf_graph.py`
- `src/gpt2_pytorch/hub/batch.py`
- `tests/test_batch.py`
- `.plan/hf-graph-pipeline/DECISIONS.md`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- tools/hf_graph.py --all/--coverage
- output/categories/descriptor_backlog.csv (untracked)

| criterion | verified by |
|---|---|
| A killed batch resumes without redoing completed rows | 05d.B.P1.T1 |
| Coverage totals match the ledger | 05d.C.P1.T1 |
| D3 is set from the 100-row measurement | 05d.C.P1.T1 |
| The backlog ranks keys by rows unlocked | 05d.B.P1.T3 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- Projected full-run time exceeds a day at safe concurrency: record it in DECISIONS.md and ask the owner before 07f.
