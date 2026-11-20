# 06f SCOPE: Path B library trace and the C2 signature check

Kind feature, wave 6, complexity L. Master sections §6, §3, D4, D5, §7 (Wave 6).
Depends on 07b, 07c, 07d, 07e; blocks 07f. Progress: [TASKS.md](TASKS.md).

## Goal

Trace transformers/timm/diffusers classes on the meta device into Program primitives as the coverage fallback, and check path A against path B (C2).

## Invariant

Path B never sets trust_remote_code and never loads weights (meta device only).

## Requirements

| id | requirement | master source |
|---|---|---|
| R-12 | Path B traces timm and diffusers classes and descriptor-less LLM model_types on the `meta` device without remote code, maps ATen ops to Program primitives and the rest to typed `opaque` ops (D5); a tiny timm model and a tiny diffusers UNet give valid graphs whose parameters all bind | §1, §6, D4, D5 |
| R-45 | C2: for every path-A descriptor with an upstream class, the tiny-config path-A matmul signature equals path B's, modulo dense-vs-routed experts | §6, §7 (Wave 6) |

## Scope and non-goals

- R-12: trace.py for transformers/timm/diffusers, opaque ops, run_one fallback.
- R-45: C2 signature check.

Non-goals:

- Remote-code classes (D4)
- Numeric execution

## Write set

- `pyproject.toml`
- `uv.lock`
- `src/gpt2_pytorch/registry/trace.py`
- `src/gpt2_pytorch/hub/pipeline.py`
- `tools/conformance.py`
- `tests/test_trace.py`
- `tests/test_pipeline.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>... --c2 all
uv sync --inexact --extra trace --extra hub --extra dev
```

## Deliverables and acceptance

- src/gpt2_pytorch/registry/trace.py
- tools/conformance.py --c2

| criterion | verified by |
|---|---|
| Tiny timm and UNet graphs are valid and bind fully | 06f.B.P1.T2 |
| C2 passes for every path-A descriptor with an upstream class | 06f.B.P1.T4 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- torch.export cannot trace a class: return failed:trace_<code>; do not patch library code.
