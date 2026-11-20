# 07f SCOPE: Full catalog run, category READMEs, TEL round trip over all graphs

Kind validation, wave 7, complexity M. Master sections §7 (Wave 7), §5.4, §8.2, §9.6.
Depends on 06f, 05d, 09a; blocks none. Progress: [TASKS.md](TASKS.md).

## Goal

Run every job to a terminal status, prove the reports add up and the re-run is a no-op, write category READMEs and round-trip every stored graph through TEL.

## Invariant

One terminal status per job; a second run with unchanged shas adds nothing.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-46 | After the full run every job has one terminal status, the reports add up to the R-01 job count, and a re-run with unchanged shas is a no-op | §7 (Wave 7), §8.2 |
| R-47 | Each output/categories/<category>/README.md summarizes that category's statuses, keys and near-misses | §5.4 |
| R-48 | tel_roundtrip runs over every graph the pipeline writes | §9.6 |

## Scope and non-goals

- R-46: full run, totals, no-op re-run.
- R-47: category READMEs.
- R-48: TEL round trip over all stored graphs.

Non-goals:

- Deferred formats (D7, later wave)
- The TEL converter

## Write set

- `tools/hf_graph.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage | --readme
node tools/tel_roundtrip.cjs output/categories/graphs --const-as-buffer --fix-contract
```

## Deliverables and acceptance

- output/categories/ (untracked)
- tools/hf_graph.py --readme

| criterion | verified by |
|---|---|
| Every job is terminal, reports add up, and the re-run is a no-op | 07f.C.P1.T1 |
| Each category has a README whose totals match coverage | 07f.B.P1.T2 |
| Every stored graph goes through the TEL round trip | 07f.B.P1.T3 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- The run would exceed the D3-sized budget or hit sustained 429s: pause, record, resume later (the ledger resumes).
