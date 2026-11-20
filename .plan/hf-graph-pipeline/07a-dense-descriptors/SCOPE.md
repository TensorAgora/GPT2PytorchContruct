# 07a SCOPE: Conformance runner and top dense-decoder descriptors

Kind feature, wave 1, complexity L. Master sections §7 (Wave 1), §8.3.
Depends on 05d, 06d, 06e; blocks 07b, 07c, 07e. Progress: [TASKS.md](TASKS.md).

## Goal

Add tools/conformance.py (C1, C3–C6) and descriptors for the top decoder_dense keys by rows, then show the coverage gain.

## Invariant

Every registered descriptor passes C1 and C6; adding one never breaks a golden.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-38 | A conformance runner checks C1 (binding on a real header fixture), C3 (HF and GGUF schemes give the same canonical names), C4 (matmul, edge and FLOP counts), C5 (hostile input) and C6 (determinism) per descriptor | §8.3 |
| R-39 | Descriptors for the top decoder_dense keys by rows pass C1, C4 and C6, and decoder_dense coverage rises by the predicted rows | §7 (Wave 1) |

## Scope and non-goals

- R-38: tools/conformance.py C1, C3, C4, C5, C6.
- R-39: top decoder_dense descriptors and the coverage gain.

Non-goals:

- C2 (06f)

## Write set

- `tools/conformance.py`
- `src/gpt2_pytorch/registry/descriptors/`
- `tests/fixtures/headers/`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>...
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- tools/conformance.py
- dense descriptors

| criterion | verified by |
|---|---|
| Every dense descriptor passes C1, C4 and C6 | 07a.B.P2.T2 |
| decoder_dense coverage rises by the predicted rows | 07a.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A dense key needs a new block: add it to 06a's blocks.py only if no existing block fits; record why in DESIGN.md.
