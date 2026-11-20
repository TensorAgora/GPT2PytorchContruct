# 07d DESIGN

Drafted at plan time; completed in Stage A.

## Current state

tools/model_graph.py counts the Qwen3.8 vision tower and MTP head in `checkpoint.unbound`.

## Interfaces

- graph `components: [{name, state: 'bound'|'declared_without_graph', tensors, params}]`.
- Assembly descriptor `{components: [{name, prefix, descriptor?}]}`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/build.py` | assemblies | R-43 |
| `src/gpt2_pytorch/registry/bind.py` | component class | R-43 |
| `src/gpt2_pytorch/registry/descriptors/` | VLM wrappers, draft heads | R-43 |
| `tests/test_assemblies.py` | new | R-43 |

## Approach

Prefix-partition the stored tensors per component; bind the text tower with its descriptor; others become `component` class with state declared_without_graph.

## Likely blockers

none
