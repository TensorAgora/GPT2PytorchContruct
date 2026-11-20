# 06a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`tensorprog/program.py` (Program: contract, elementwise, reduce, gather, slice, reshape, permute, rename, composite), `tensorprog/viz.py:11` `viz_graph(prog, source)`, `tensorprog/lower.py` (GPT-2), `tensorprog/qwen3_5.py`. arch.mjs (368 lines) in TensormorphDesignUI holds llamaGraph, mixtralGraph, deepseekGraph and the six blocks; its configs are in `activity_bar/shared/mock.js`.

## Interfaces

- Descriptor JSON: `{id, version, match, hyperparameters, schemes:{hf,gguf,mlx}, recipe, ties, ignored, provenance}`.
- `load(id) -> Descriptor`; `layer_kinds(descriptor, cfg) -> list[str]`.
- Blocks: `rms(p, x, w, eps)`, `rope(p, x, theta, ...)`, `softmax(p, s, axis)`, `gqa(p, x, layer, hyper)`, `swiglu(p, x, layer, hyper)`, `experts(p, x, layer, hyper)` (final signatures in Stage A).
- `build(descriptor, hyper, scheme) -> (graph, expected: {key_or_view: shape})`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/__init__.py` | new | R-27 |
| `src/gpt2_pytorch/registry/descriptor.py` | new | R-27 |
| `src/gpt2_pytorch/registry/blocks.py` | new | R-28, R-52 |
| `src/gpt2_pytorch/registry/build.py` | new | R-27 |
| `src/gpt2_pytorch/registry/descriptors/` | gpt2, llama, qwen2, mistral, phi3, gemma2 | R-29, R-30 |
| `tests/fixtures/spelled/` | 7 spelled configs | R-30 |
| `tests/test_descriptor.py` | new | R-27 |
| `tests/test_blocks.py` | new | R-28 |
| `tests/test_path_a_goldens.py` | new | R-29, R-30 |

## Approach

Port arch.mjs block by block, diffing each against the matching golden layer. Descriptor recipe = ordered block calls per layer kind; selectors (§7.2) give the kind per layer. One `blocks.py` until it passes ~800 lines (D-04).
Rejected: a class per architecture (descriptors are data, §7.2).

## Likely blockers

- B/T symbols differ from arch.mjs goldens (null): comparator ignores symbols.
- distilgpt2 byte identity: keep the golden's `source` string and match its JS `JSON.stringify` serialization (HARNESS).
