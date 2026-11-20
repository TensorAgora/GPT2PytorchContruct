# 06b SCOPE: mixtral, deepseek_v3 and qwen3_5 descriptors

Kind feature, wave 0, complexity M. Master sections §2, §7, §8.1.
Depends on 06a; blocks 06c. Progress: [TASKS.md](TASKS.md).

## Goal

Finish the Wave 0 golden set: the MoE (mixtral, DeepSeek-V3 with MLA) and hybrid (qwen3_5) descriptors rebuild their goldens.

## Invariant

The 5 goldens of 06a still pass; qwen3_5.py is called, not copied.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-31 | The mixtral and deepseek_v3 descriptors (arch.mjs ports) and the qwen3_5 descriptor (over qwen3_5.py) rebuild their goldens structurally equal from their spelled configs, deepseek's 174 keyless views matched rather than given invented keys | §7 (Wave 0), §8.1 |

## Scope and non-goals

- R-31: mixtral, deepseek_v3 and qwen3_5 descriptors rebuild their goldens from spelled configs.

Non-goals:

- Routed/scan annotations (07b, 07c)
- Generated name maps (06d)

## Write set

- `src/gpt2_pytorch/registry/blocks.py`
- `src/gpt2_pytorch/registry/descriptors/`
- `tests/fixtures/spelled/`
- `tests/test_path_a_goldens.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m gpt2_pytorch.registry.build --descriptor ID --config FILE --out FILE
.venv/bin/python -m tools.graph_check G.json [--against json/<golden>.json] [--manifest M.json]
```

## Deliverables and acceptance

- registry/descriptors/mixtral.json
- registry/descriptors/deepseek_v3.json
- registry/descriptors/qwen3_5.json

| criterion | verified by |
|---|---|
| mixtral, deepseek-v3 and qwen3.8 rebuild from spelled configs with 0 differences, deepseek's views matched | 06b.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- deepseek-v3 needs invented checkpoint keys to match: stop; the 174 views stay keyless.
