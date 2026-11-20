# 06a SCOPE: Descriptor format, selectors, blocks from arch.mjs, GPT-2 and llama-family descriptors

Kind foundation, wave 0, complexity L. Master sections §2, §4, §6, §8.1, §9.4.
Depends on 00a; blocks 06b, 05c, 06d. Progress: [TASKS.md](TASKS.md).

## Goal

Path A core: descriptor JSON with selectors, the arch.mjs blocks ported onto Program, build(), and the gpt2 and llama-family descriptors reproducing five goldens.

## Invariant

`json/distilgpt2.json` rebuilds byte-identically; blocks only call existing Program primitives.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-27 | Descriptors are JSON documents with match, hyperparameters (ordered sources plus a shape check), schemes (hf, gguf, mlx), recipe, ties, ignored and provenance, and support the layer selectors uniform, list(key), first_k(key), period, pattern, index_sets, per_layer_config and by_presence | §4 (§7.2), §6 |
| R-28 | The arch.mjs blocks rms, rope, softmax, gqa, swiglu and experts are ported to Python functions on Program, reusing program.py and viz.py rather than re-implementing them | §2, §6 |
| R-29 | The GPT-2 descriptor rebuilds json/distilgpt2.json byte-identically | §8.1 |
| R-30 | Llama-family descriptors rebuild the qwen2.5-7b-instruct, mistral-7b-instruct-v0.3, phi-3.5-mini-instruct and gemma-2-2b-it goldens structurally equal from the configs they were spelled from | §7 (Wave 0) |
| R-52 | Pipeline graphs keep `axes[].symbol` for B and T | §9.4 |

## Scope and non-goals

- R-27: descriptor JSON loader/validator and all §7.2 selectors.
- R-28: rms, rope, softmax, gqa, swiglu, experts as Python blocks on Program.
- R-29: gpt2 descriptor over lower.py.
- R-30: llama-family descriptors for 4 goldens.
- R-52: blocks set B/T axis symbols.

Non-goals:

- Resolution and binding (06c)
- Generated GGUF name maps (06d)

## Write set

- `src/gpt2_pytorch/registry/__init__.py`
- `src/gpt2_pytorch/registry/descriptor.py`
- `src/gpt2_pytorch/registry/blocks.py`
- `src/gpt2_pytorch/registry/build.py`
- `src/gpt2_pytorch/registry/descriptors/`
- `tests/fixtures/spelled/`
- `tests/test_descriptor.py`
- `tests/test_blocks.py`
- `tests/test_path_a_goldens.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m gpt2_pytorch.registry.build --descriptor ID --config FILE --out FILE
.venv/bin/python -m tools.graph_check G.json [--against json/<golden>.json] [--manifest M.json]
```

## Deliverables and acceptance

- src/gpt2_pytorch/registry/descriptor.py
- src/gpt2_pytorch/registry/blocks.py
- src/gpt2_pytorch/registry/build.py
- src/gpt2_pytorch/registry/descriptors/

| criterion | verified by |
|---|---|
| distilgpt2 rebuilds byte-identically through the gpt2 descriptor | 06a.C.P1.T1 |
| qwen2.5, mistral, phi-3.5 and gemma-2 rebuild with 0 structural differences | 06a.C.P1.T1 |
| Every selector maps a sample config to the expected layer kinds | 06a.B.P1.T1 |
| Blocks reproduce a golden layer's primitives and shapes, with B/T symbols set | 06a.B.P2.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A golden needs a primitive Program lacks: stop and record it; do not add a primitive.
