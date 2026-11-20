# 06b DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`json/mixtral-8x7b-instruct-q4km.json` (385 multiplications, 128 over E), `json/deepseek-v3.json` (1,026; 174 keyless views), `json/qwen3.8-27b-fp8.json` (built by tools/model_graph.py from qwen3_5.py with --seq 8). No spelled config for qwen3.8 is local.

## Interfaces

- `registry/descriptors/mixtral.json`, `deepseek_v3.json`, `qwen3_5.json`.
- `mla(p, x, layer, hyper)` block in blocks.py.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/registry/blocks.py` | add mla, shared experts | R-31 |
| `src/gpt2_pytorch/registry/descriptors/` | mixtral, deepseek_v3, qwen3_5 | R-31 |
| `tests/fixtures/spelled/` | qwen3.8 config | R-31 |
| `tests/test_path_a_goldens.py` | all 8 | R-31 |

## Approach

mixtral = llama recipe with experts in place of swiglu; deepseek_v3 = mla + first_k(first_k_dense_replace) dense then experts with shared experts; qwen3_5 = recipe step that calls `lower_qwen3_5` with the descriptor's scheme prefix `model.language_model.`.

## Likely blockers

- Qwen3.8 config only on the Hub: fetch config.json (allowlisted).
