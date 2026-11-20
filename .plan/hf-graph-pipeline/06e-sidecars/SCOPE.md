# 06e SCOPE: Quantization sidecar binding

Kind feature, wave 1, complexity M. Master sections §4 (§7.4).
Depends on 06c, 08a; blocks 07a. Progress: [TASKS.md](TASKS.md).

## Goal

Bind quantization sidecars to their principals only when the shape relation holds, and record each binding in report.json.

## Invariant

A sidecar binds only when its shape relation to the principal holds.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-07 | Quantization sidecars (FP8 weight_scale_inv, MXFP4, GPTQ/AWQ, MLX, bitsandbytes, GGUF ggml types) bind only when their shape relation holds and each binding is recorded in report.json; Qwen3.8-27B-FP8 has every weight_scale_inv attached and 0 unbound sidecars | §1, §4 (§7.4) |

## Scope and non-goals

- R-07: sidecar rules for six schemes, report.json bindings, Qwen3.8 FP8 with 0 unbound.

Non-goals:

- Dequantization
- TEL quantization records (§9.3 counted exclusions)

## Write set

- `src/gpt2_pytorch/registry/sidecars.py`
- `src/gpt2_pytorch/registry/bind.py`
- `src/gpt2_pytorch/registry/descriptors/qwen3_5.json`
- `tests/test_sidecars.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- src/gpt2_pytorch/registry/sidecars.py

| criterion | verified by |
|---|---|
| Qwen3.8-27B-FP8: every weight_scale_inv attached, 0 unbound sidecars | 06e.C.P1.T1 |
| A sidecar with the wrong shape relation never binds | 06e.B.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
