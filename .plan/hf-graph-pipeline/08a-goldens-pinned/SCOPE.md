# 08a SCOPE: Pin the 8 goldens, diff their configs, rebuild them from headers

Kind validation, wave 0, complexity M. Master sections §8.1, §7.
Depends on 05c, 05b; blocks 06d, 06e. Progress: [TASKS.md](TASKS.md).

## Goal

Wave 0 exit: each golden repo pinned at a sha, its real config diffed against the spelled one, and rebuilt by the pipeline from its header fixture.

## Invariant

`json/*.json` are never edited; every remaining difference has a written cause.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-11 | The pipeline rebuilds each golden repo from its pinned header structurally equal to its `json/*.json` (same dims, parameter set and shapes, checkpoint_key or view per parameter, op/edge/multiplication counts and totals), each remaining difference listed with its cause | §1, §8.1 |
| R-36 | Each golden repo is pinned at a sha in csv/goldens.csv and its real config.json is diffed against the config the golden was spelled from, every delta listed with its cause | §8.1 |

## Scope and non-goals

- R-11: pipeline rebuild from pinned header fixtures equals each golden.
- R-36: shas pinned in csv/goldens.csv; real vs spelled config deltas listed.

Non-goals:

- Regenerating goldens (they are the oracle)

## Write set

- `csv/goldens.csv`
- `tests/fixtures/headers/`
- `tests/test_goldens.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
.venv/bin/python -m tools.graph_check G.json [--against json/<golden>.json] [--manifest M.json]
```

## Deliverables and acceptance

- csv/goldens.csv revisions
- tests/fixtures/headers/<golden>@<sha>/
- tests/test_goldens.py

| criterion | verified by |
|---|---|
| All goldens rebuild from pinned headers structurally equal, remaining differences listed with causes | 08a.C.P1.T1 |
| Each golden is pinned and its config deltas listed | 08a.B.P1.T2 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- The mixtral GGUF repo lacks the Q4_K_M file named in mock.js: pick a repo that has it and record it in DECISIONS.md.
