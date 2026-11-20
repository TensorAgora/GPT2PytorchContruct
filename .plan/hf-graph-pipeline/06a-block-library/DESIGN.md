# 06a DESIGN

## Current state

`tensorprog/program.py` (Program: contract, elementwise, reduce, gather, slice, reshape, permute, rename, composite;
`to_dict`), `tensorprog/viz.py:11` `viz_graph(prog, source)`, `tensorprog/lower.py` (`lower(model, batch, seq)` walks a
DistilGPT2LMHeadModel; it needs module structure only, so a meta-device model works), `tensorprog/qwen3_5.py`.
arch.mjs builds the 6 arch goldens with a JS builder whose ops map 1:1 onto Program, then `doc.rebuild()` derives
edges/groups/multiplications/totals exactly as `viz_graph()` does (equation and cost strings are identical).

Prototype (2026-11-21): meta-device `DistilGPT2LMHeadModel(GPT2Config())` → `lower(model, 2, 8)` → `to_dict` + recorded
verification → JSON round trip → `viz_graph(..., "artifacts/tensor_program.json")` → JS-style serialization is
byte-identical to `json/distilgpt2.json` (319,920 bytes).

## Blocks

Python functions on a `Builder` (a thin wrapper over Program: `within(module, type)` context, `param`, `buffer`,
`constant`, and the arch-style shorthands `E/R/C`, `permute(out, x, perm)` converted to axis names). Ported from
`TensormorphDesignUI/.../viewport/graphs/arch.mjs`:

| block | Python | arch.mjs lines |
|---|---|---|
| rms | `rms(b, out, x, g, axis)` | 91-98 |
| rope | `rope(b, out, x, pos, dim, theta)` (R, cos, sin constants, 7-digit tables) | 101-111 |
| softmax | `softmax(b, out, s)` (region `softmax`, axis k) | 113-121 |
| silu | `silu(b, out, x)` | 123 |
| gqa | `gqa(b, h, i, at, key)` | 126-157 |
| swiglu | `swiglu(b, at, i, key, F, view)` | 159-166 |
| experts | `experts(b, at, i, x, key, view)` | 169-178 |

Templates: llamaGraph 199-219 (the `decoder` template), mixtralGraph 222-249, deepseekGraph 254-329 (06b).

Layer bodies (`attention`, `mlp` slots) dispatch by block name through `ATTENTION` and `MLP` tables; 06b adds `mla`,
`moe_mixtral`, `moe_deepseek`.

## Descriptor

`src/gpt2_pytorch/registry/descriptors/<id>.json`; `extends` merges another descriptor one level deep.

```json
{"id": "qwen2", "version": "1", "extends": "llama",
 "match": {"model_type": ["qwen2"], "architectures": ["Qwen2ForCausalLM"], "gguf_architecture": ["qwen2"]},
 "hyperparameters": {"L": ["num_hidden_layers"], "D": ["hidden_size"], "Hq": ["num_attention_heads"],
                     "Hkv": ["num_key_value_heads", "=Hq"], "Dh": ["head_dim", "=D // Hq"], "F": ["intermediate_size"],
                     "V": ["vocab_size"], "eps": ["rms_norm_eps", "=1e-6"], "theta": ["rope_theta", "=10000"],
                     "tie": ["tie_word_embeddings", "=0"]},
 "schemes": {"hf": {"embed": "model.embed_tokens.weight", "q": "model.layers.{i}.self_attn.q_proj.{what}", "...": "..."}},
 "recipe": {"template": "decoder", "root": "qwen2", "flags": {"qkv_bias": true},
            "layers": {"attention": {"uniform": "gqa"}, "mlp": {"uniform": "swiglu"}}},
 "ties": {"head": "embed", "when": "tie"},
 "ignored": ["model.layers.*.self_attn.rotary_emb.inv_freq"],
 "provenance": {"ported_from": "arch.mjs llamaGraph", "golden": "json/qwen2.5-7b-instruct.json"}}
```

Hyperparameter sources are config keys in order; `=expr` is a default over already-resolved names (+ - * / // and
numbers, parsed with `ast`, never `eval`). A scheme value is a key template (`{i}`, `{what}`, any dim) or
`{"view": "<text>"}` for a keyless parameter.

## Selectors

Each slot selector returns a block name per layer, `layer_blocks(selector, cfg, L, names=())`:

| selector | form | example |
|---|---|---|
| uniform | `{"uniform": B}` | every layer gqa |
| list | `{"list": key, "map": {value: B}}` | `layer_types` |
| first_k | `{"first_k": key, "first": B1, "then": B2}` | `first_k_dense_replace` |
| period | `{"period": n or key, "phase": p or key, "on": B1, "off": B2}` | Jamba attention every 8th |
| pattern | `{"pattern": key, "map": {char: B}}` | `hybrid_override_pattern` |
| index_sets | `{"index_sets": {B: key or list}, "default": B}` | listed attention layers |
| per_layer_config | `{"per_layer_config": key, "field": f, "map": {value: B}}` | per-layer dicts |
| by_presence | `{"by_presence": {B: "name with {i}"}, "default": B}` | needs header names |

## Build

`build(descriptor, hyper, scheme="hf") -> (graph, expected)`: template → Program (symbols B, T, q→T, k→T; R-52) →
`to_dict` → JSON round trip → `viz_graph()`; non-GPT-2 builds null `pytorch_name` (a GPT-2 debug mapping) and add
`lanes` when the template has experts. `expected` = `[{name, role, shape, checkpoint_key, view}]` for parameters and
buffers. The gpt2 template builds a meta-device DistilGPT2LMHeadModel from the HF config and calls `lower()`; its
descriptor provenance holds the recorded verification of that lowering, emitted only when the dims equal distilgpt2's.
`dump_js(graph)` writes JSON the way `JSON.stringify` does (compact, JS number formatting), which the goldens use.
CLI: `python -m gpt2_pytorch.registry.build --descriptor ID --config FILE --out FILE`.

## Spelled configs

`tests/fixtures/spelled/<golden>.config.json`, HF config.json keys, values from `activity_bar/shared/mock.js` `cfg` plus
arch.mjs constants (eps 1e-5, deepseek 1e-6; rope_theta per arch THETA: llama 500000, qwen2 1e6, gemma2 1e4, mixtral 1e6,
deepseek 1e4), so constants match the goldens. distilgpt2's from `GPT2Config()`.

## Likely blockers

- Gemma 2 and Phi-3 are llama-layout in arch.mjs (no Gemma post-norms/softcap/GeGLU; Phi's fused qkv/gate_up stored as
  separate keys). The descriptors reproduce the goldens; 06d (fused storage) and 07a (real Gemma 2) fix them for real headers.
