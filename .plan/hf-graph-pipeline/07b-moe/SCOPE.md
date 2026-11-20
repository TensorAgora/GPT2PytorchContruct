# 07b SCOPE: Routed MoE annotations, expert storage forms, MoE descriptors

Kind feature, wave 2, complexity L. Master sections §5.3 (D8), §4 (§15), §7 (Wave 2).
Depends on 07a; blocks 07c, 07d, 06f. Progress: [TASKS.md](TASKS.md).

## Goal

Keep MoE dense in v1 primitives but annotate routing and report active FLOPs; bind all expert storage forms; cover the top decoder_moe keys.

## Invariant

MoE stays expressed in the eight v1 primitives; routing lives only in `execution` and totals.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-40 | Expert ops and their multiplications carry execution {kind: routed, of: E, active: K, router}, totals add contraction_flops and active_flops, active_flops equals the K-of-E hand count, and the mixtral and deepseek-v3 goldens stay structurally equal | §5.3 (D8), §7 (Wave 2) |
| R-41 | All four expert storage forms bind and descriptors cover the top decoder_moe keys | §4 (§15), §7 (Wave 2) |

## Scope and non-goals

- R-40: routed annotations, contraction_flops and active_flops, goldens equal.
- R-41: four storage forms and top decoder_moe descriptors.

Non-goals:

- Conditional execution or new route primitives (D8 alternative)

## Write set

- `src/gpt2_pytorch/registry/blocks.py`
- `src/gpt2_pytorch/registry/descriptors/`
- `src/gpt2_pytorch/registry/namemaps/overlay.json`
- `tests/test_moe_storage.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>...
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- routed `execution` annotations
- MoE descriptors

| criterion | verified by |
|---|---|
| mixtral and deepseek-v3 stay equal and active_flops matches the hand count | 07b.C.P1.T1 |
| All four storage forms bind and decoder_moe coverage rises | 07b.C.P1.T2 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- The owner overrides D8 (v2 route primitives): stop the module and re-plan.
