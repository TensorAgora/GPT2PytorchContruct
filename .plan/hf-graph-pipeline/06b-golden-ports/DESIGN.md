# 06b DESIGN

## Current state

06a's `decoder` template, `Builder` and blocks rebuild 5 goldens. Remaining: mixtral (GGUF names, MoE), deepseek-v3 (MLA,
first-k dense, shared expert, keyless per-expert views) and qwen3.8 (hybrid, from `tensorprog/qwen3_5.py`).

## mixtral

Descriptor `mixtral` extends `llama`; recipe: `decoder` template, attention `gqa`, mlp `moe_mixtral`, lanes [E, H, B].
`moe_mixtral` ports arch.mjs mixtralGraph 236-242: `router` (Wr_i [E, D]), `Sel_i` runtime buffer, softmax over the
chosen experts (exp, mul Sel, sum E, div), `experts` (arch 169-178), weighted sum over E. Schemes: `gguf` hand-listed
from mixtralGraph's `k(i, …)` names (generated in 06d); `hf` reads per-expert w1/w3/w2 as stacked views.

## deepseek_v3

Descriptor `deepseek_v3` (match also `kimi_k2`); attention `mla` for every layer; mlp `first_k(first_k_dense_replace)`
→ `swiglu`, then `moe_deepseek`. `mla` ports deepseekGraph 266-300 (q_a → q_a_layernorm → q_b; kv_a_proj_with_mqa sliced
into the KR latent and the shared rotary key Dr; kv_a_layernorm → kv_b; nope and rope contractions summed). `moe_deepseek`
ports 301-322: sigmoid router, Sel, normalize, × `c_route` (routed_scaling_factor), routed experts as keyless views
`stack of model.layers.{i}.mlp.experts.{0…255}.{gate,up,down}_proj.weight (FP8, 128 × 128 block scales)` (the 174 keyless
views = 58 MoE layers × 3), shared expert = `swiglu` with prefix `s` (Wsg/Wsu/Wsd) over Fe. `score_dim` = Dn + Dr.

## qwen3_5

Descriptor `qwen3_5`, template `qwen3_5`: calls `tensorprog/qwen3_5.py lower_qwen3_5(cfg, 1, 8, prefix)` unchanged, with
its own ROLES; scheme `hf` = prefix `model.language_model.`. The 400 FP8 `*_scale_inv` views come from the header, so the
spelled-config check uses `--ignore view` (D-16); sidecars (06e) restore them from the header.

## Result (2026-11-21)

From tests/fixtures/spelled: mixtral 0 differences, deepseek-v3 0 differences, qwen3.8 0 differences with `--ignore view`.
