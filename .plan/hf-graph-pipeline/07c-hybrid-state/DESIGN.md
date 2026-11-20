# 07c DESIGN

## Recurrence stays unrolled; the scan is an annotation (D8)

Ops of an unrolled step are named `<sym>_<layer>.t<step>.*`: the gated delta rule's `Dn_i` (`tensorprog/qwen3_5.py`,
called unchanged by the `qwen3_5` template) and Mamba's `Ss_i`. `build.annotate_scan` gives each of them, and their
multiplications, `execution: {kind: "scan", axis: "T", step: t, body: "<module>:<sym>"}`, where body is the statement
group of that recurrence in `groups`. qwen3.8-27b-fp8: 5,088 step ops annotated, golden still equal.

## Blocks

| block | used by | spelling |
| --- | --- | --- |
| delta rule | qwen3_5 / qwen3_5_text (layer_types inside the lowering) | qwen3_5.py, unchanged |
| `mamba` (Mamba-1) | jamba | in_proj → x, z; causal conv (conv_shift taps, qwen3_5.py's spelling) + bias + SiLU; x_proj → dt, B, C (Jamba RMSNorms each); dt = softplus(dt_proj); per step h_t = exp(dt_t ⊗ A) h_{t-1} + (dt_t x_t) ⊗ B_t, y_t = C_t · h_t; outer products are contractions over a size-1 axis U; (y + D x) · silu(z); out_proj |
| `lfm2_conv` | lfm2 | B, C, x = in_proj; y = C · causal_conv(B · x); out_proj |
| `gqa_norope` | jamba attention layers | gqa without rotary embeddings |

## Selectors and keys (catalog rows, decoder_hybrid + decoder_hybrid_moe)

| key | rows | descriptor | selector |
| --- | --: | --- | --- |
| qwen35 (GGUF) | 262 | — | not this wave: GGUF names for the qwen3_5 lowering need a per-arch map of its ssm tensors |
| lfm2 | 107 + 17 GGUF | **lfm2** | attention: `index_sets {gqa: full_attn_idxs}`, default `lfm2_conv` |
| qwen3_5_text | 83 + 33 GGUF | qwen3_5 (06b) | inside the lowering (`layer_types`) |
| jamba | 71 | **jamba** | attention: `period attn_layer_period, phase attn_layer_offset` → gqa_norope / mamba; mlp: `period expert_layer_period, phase expert_layer_offset` → moe_qwen (softmax, not renormalized) / swiglu |
| falcon_h1 38, nemotron_h 35, granitemoehybrid 25 | — | — | Mamba-2 mixers (not this wave; path B candidates) |

Predicted rows unlocked: lfm2 107 + jamba 71 = 178 safetensors rows (LFM2's final norm is `TOKEN_EMBD_NORM` in gguf-py,
role `embd_norm`).
