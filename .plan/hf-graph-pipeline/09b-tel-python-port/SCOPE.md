# 09b SCOPE: Python T♭ printer and decoder (T-2)

Kind feature, wave 1, complexity M. Master sections §9.5, §9.6, §10 (T-2).
Depends on 09a; blocks none. Progress: [TASKS.md](TASKS.md).

## Goal

Port tel.js printFlat/build to Python next to viz_graph() with both fixes, checked against tel.js on the goldens.

## Invariant

Python T♭ text equals tel.js output for the 8 goldens.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-51 | The T♭ printer and decoder are ported to Python next to viz_graph() with both §9.6 fixes, matching tel.js on the 8 goldens (T-2 default) | §9.6, §10 (T-2) |

## Scope and non-goals

- R-51: print_flat and build in Python with both fixes, checked against tel.js.

Non-goals:

- The batch TEL converter

## Write set

- `src/gpt2_pytorch/tensorprog/tel_flat.py`
- `tools/tel_roundtrip.cjs`
- `tests/test_tel_flat.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
node tools/tel_roundtrip.cjs <graph> --const-as-buffer --fix-contract --emit-flat OUT
```

## Deliverables and acceptance

- src/gpt2_pytorch/tensorprog/tel_flat.py

| criterion | verified by |
|---|---|
| Python T♭ equals tel.js on the 8 goldens and round-trips with 0 differences | 09b.C.P1.T1 |

## Stop conditions

- tel.js behaviour differs from §9.5: follow tel.js, record the difference in HARNESS.md.
