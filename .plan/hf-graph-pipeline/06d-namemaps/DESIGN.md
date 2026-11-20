# 06d DESIGN

## Pinned sources

- gguf-py **0.19.0** (PyPI, the `hub` extra) = llama.cpp `gguf-py/pyproject.toml` at commit
  **a83d68b562b19fba1f8836edc93eb76d9ca1ce25** (2026-11-20). MIT licence (llama.cpp). The tables come from
  `gguf.constants` (`MODEL_ARCH`, `MODEL_TENSOR`, `TENSOR_NAMES`, `MODEL_TENSORS`) and `gguf.tensor_mapping`
  (`TensorNameMap.mappings_cfg`, `block_mappings_cfg`, `arch_block_mappings_cfg`).
- The converter moved to `conversion/*.py` at that commit. Q/K RoPE interleave, `conversion/llama.py`:
  - 176-181 `def permute(weights, n_head, n_head_kv): … reshape(n_head, 2, rows // n_head // 2, …).swapaxes(1, 2)`
  - 262-265 `if name.endswith(("q_proj.weight", "q_proj.bias")): data_torch = LlamaModel.permute(data_torch, n_head, n_head)`
    / `k_proj` with `n_kv_head`
- Gemma 2 norm offset, `conversion/gemma.py:119-120`: `if name.endswith("norm.weight"): data_torch = data_torch + 1`.

## Generated table

`tools/gen_namemaps.py` writes `src/gpt2_pytorch/registry/namemaps/gguf.json`; `--check` regenerates and compares:

```json
{"_meta": {"gguf_version": "0.19.0", "llama_cpp_commit": "a83d68b…", "license": "MIT, © the llama.cpp authors", "generated_by": "tools/gen_namemaps.py"},
 "tensor_names": {"ATTN_Q": "blk.{i}.attn_q", "TOKEN_EMBD": "token_embd", "...": "..."},
 "arch_tensors": {"llama": ["TOKEN_EMBD", "ATTN_Q", "..."], "...": []},
 "hf_to_tensor": {"model.layers.{i}.self_attn.q_proj": ["ATTN_Q"], "...": []},
 "arch_hf_to_tensor": {"phi3": {"...": ["..."]}}}
```

Runtime (`registry/namemaps.py`): `ROLE_TENSOR` maps descriptor scheme roles to `MODEL_TENSOR` names (q → ATTN_Q,
experts_g → FFN_GATE_EXP, router → FFN_GATE_INP, q_a → ATTN_Q_A, …). `gguf_scheme(arch)` = `{role:
tensor_names[ROLE_TENSOR[role]] + ".{what}"}` for the tensors `arch_tensors[arch]` lists; `resolve_and_bind` uses it for
every GGUF manifest, so no descriptor hand-lists GGUF names. C3: each HF scheme key, with the layer index and suffix
stripped, maps through `hf_to_tensor` to the role's `MODEL_TENSOR`.

## Overlay

`registry/namemaps/overlay.json`, keyed by scheme then GGUF arch or descriptor id:

- `rope_permute` (gguf llama, mistral-in-llama, mixtral; rules `blk.{i}.attn_q.weight` heads `Hq`, `blk.{i}.attn_k.weight`
  heads `Hkv`): values only. Shapes bind unchanged; the report lists the rule and its tensor count, so a numeric or TEL run
  undoes it.
- `offset` (gguf gemma, gemma2, gemma3; `*norm.weight` +1): values only, reported likewise.
- `split` (fused storage): hf phi3 `model.layers.{i}.self_attn.qkv_proj.weight` → q (Dq), k (Dkv), v (Dkv) rows;
  `model.layers.{i}.mlp.gate_up_proj.weight` → gate (F), up (F); gguf phi3 `blk.{i}.attn_qkv.weight`, `blk.{i}.ffn_up.weight`
  (gate and up fused). When the fused tensor is stored, the parameters of those roles bind to it with a row-slice view
  `[a:b,:]`. Binding checks the fused shape: the parts sum along the axis, the other dims match.
- `stack`, `fuse`, `transpose`: reserved forms used by 07b (expert storage).

Graph effect: split parameters get `checkpoint_key` = the fused key and a slice `view`, carried as graph.ref overrides.
permute and offset leave the graph unchanged: the goldens must stay equal, and a GGUF mixtral graph is the same graph.
