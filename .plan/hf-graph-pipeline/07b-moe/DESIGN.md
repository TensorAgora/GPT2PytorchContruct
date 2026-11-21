# 07b DESIGN

## Routed annotation (D8)

MoE stays dense in the v1 primitives: every expert is drawn and weighted by the router's top-k choice (`Sel_i`). The
`experts` block's ops (gate, activation, up, product, down over the `E` axis) and their `multiplications[]` entries carry
`execution: {kind: "routed", of: "E", active: K, router: <index of the R_i op>}`. The router contraction, the gating
ops, the weighted sum over E and shared experts are dense and carry nothing.

`totals.active_flops = contraction_flops − Σ_routed flops · (E − K) / E` (each routed contraction's FLOPs are a multiple
of E, so the result is exact).

Hand counts (B·T = 8; per MoE layer three routed contractions of 2·B·T·E·Fe·D FLOPs each):

| golden | E | K | Fe | D | MoE layers | contraction_flops | routed | active_flops |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| mixtral-8x7b-instruct-q4km | 8 | 2 | 14336 | 4096 | 32 | 745,512,370,176 | 721,554,505,728 | **204,346,490,880** |
| deepseek-v3 | 256 | 8 | 2048 | 7168 | 58 | 10,722,415,017,984 | 10,462,540,333,056 | **586,829,070,336** |

## Expert storage forms (design doc §15)

| form | example | binding |
| --- | --- | --- |
| one tensor per expert | `mlp.experts.{e}.gate_proj.weight` (qwen3_moe, deepseek), `block_sparse_moe.experts.{e}.w1.weight` (mixtral) | `stack of …{0…E-1}…` view; each member has the per-expert shape (bind.py) |
| stacked per role | `blk.{i}.ffn_gate_exps.weight` (GGUF) | as stored, `[E, Fe, D]` |
| stacked and fused `[E, 2·Fe, D]` | `mlp.experts.gate_up_proj` (qwen3_5_moe) | overlay `split` on axis 1: gate `[:, 0:Fe, :]`, up `[:, Fe:2Fe, :]` |
| stacked and fused `[E, D, 2·Fe]` | `feed_forward.experts.gate_up_proj` (llama4) | overlay `split` on axis 2 with `in_out` layout (the parameter is `[E, Fe, D]`) |
| stacked, fused and packed | `mlp.experts.gate_up_proj_blocks` / `_scales` (gpt_oss) | MXFP4 sidecar rule gives the logical `gate_up_proj` (06e), then the split |

The split rule gains `axis` and `layout` (`out_in` default, `in_out` transposes the last two dims for the shape check);
the view names the slice, e.g. `[:,0:768,:]`.

## Descriptors

From the decoder_moe backlog (catalog rows): qwen3_moe 132, mixtral 74 (done), gpt_oss 57, deepseek_v3 54 (done),
glm_moe_dsa 52, deepseek_v4 37, deepseek_v2 34.

- **qwen3_moe** (132): qwen3 attention (q/k norms) + `moe_qwen`: softmax router, top-k renormalized
  (`norm_topk_prob`), per-expert tensors `mlp.experts.{e}.{gate,up,down}_proj.weight` as stacked views; GGUF `*_exps`.
  Fixture `Qwen/Qwen3-30B-A3B`.
- **deepseek_v2** (34): deepseek_v3's MLA and MoE with a softmax router and no `e_score_correction_bias`; V2-Lite has no
  query LoRA (`q_lora_rank: null` → `q_proj` directly). Fixture `deepseek-ai/DeepSeek-V2-Lite`.
- Not this wave: gpt_oss (attention sinks, alternating sliding windows, biased MXFP4 experts need new blocks; its
  transformers class is a path-B candidate, 06f), glm_moe_dsa and deepseek_v4 (sparse-attention indexer).

## Observed (07b.C, 2026-11-21 catalog run)

decoder_moe catalog rows, keyed by catalog `model_type`: qwen3_moe 126 ok of 132 (101 safetensors incl. MLX stacked
`switch_mlp` experts and NVFP4 + 25 GGUF `qwen3moe`); deepseek_v2 16 ok of 34. Beyond the prediction:
deepseek_v3 19 ok + 24 partial (MTP layer as component; `deepseek_v3_direct` binds q_lora_rank-null / multi-shared-
expert V3s: Moonlight, Kanana-2, GigaChat3), mixtral 69 ok. 142 ≥ 166 − the 24 misses below.

Misses: qwen3_moe 6 (colibri `merged_weight`/`qs` custom packing, 3 tiny-random configs whose layers are dense,
Loopie's looped config, 1 index 401); deepseek_v2 18 (12 ESFT expert-only deltas: no embedding or attention stored;
6 GGUF rows the catalog flags as remote code). Known class not bound: llama.cpp's split MLA GGUF form
(`attn_k_b` / `attn_v_b` in place of `attn_kv_b`, head dims only in `key_length_mla`) — GLM-4.7-Flash, Kimi-K2,
DeepSeek-R1-0528 GGUFs (~14 rows).
