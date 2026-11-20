# 06c DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`csv/categories/<category>/metadata_spec.json` gives per dim a config string and a GGUF string in prose (`"hidden_size | n_embd (GPT-2)"`, `"<arch>.embedding_length"`). `tools/model_graph.py` has an ad-hoc binding check for qwen3_5 only.

## Interfaces

- `candidates(manifest, row) -> list[Candidate(descriptor, rung, evidence, score)]`.
- `resolve_hyper(descriptor, manifest, row) -> {dim: (value, source)}`.
- `verify(expected, manifest) -> Binding(partition: {class: [keys]}, ok, near_miss)`.
- `python -m gpt2_pytorch.registry.resolve DIR` prints descriptor, rung, unbound count.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/resolve.py` | new | R-08, R-26 |
| `src/gpt2_pytorch/registry/hyper.py` | new | R-09 |
| `src/gpt2_pytorch/registry/bind.py` | new | R-10 |
| `tests/test_resolve.py` | new | R-08, R-26 |
| `tests/test_hyper.py` | new | R-09 |
| `tests/test_bind.py` | new | R-10 |

## Approach

Ladder (§4/§6): R1 GGUF `general.architecture`; R2 `architectures` before `model_type`, checking nested text_config/llm_config/language_config/thinker_config, diffusers `_class_name`, timm `architecture`/`pretrained_cfg` and the catalog's backbone_class/timm_module; R3 fingerprint = set of normalized tensor-name stems; R4 repo-name hint. Rungs only propose; `verify` decides; R3/R4 winners get `ok_assumed`. Hyper: descriptor sources first, then the spec's `a | b` alternatives and `default`.

## Likely blockers

- Prose metadata_spec strings: parse alternation only; descriptors cover the rest.
