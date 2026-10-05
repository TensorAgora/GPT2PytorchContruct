# DistilGPT2 as tensor operations

Generated 2026-10-05 19:41:49 by `python -m tools.tensor_program` from `pytorch_model.bin`. Example shape: B=2, T=8 (the program itself is symbolic in B and T).

Every operand is a tensor: weights, biases, inputs, masks, constants (rank-0 tensors) and every intermediate. `y = x·W + b` is three tensors `x`, `W`, `b` and two operations, the product `x·W` and the sum with `b`.

## Summary

|  | count |
|---|---|
| tensors | 475 |
| parameters (checkpoint tensors, tied head counted once) | 76 |
| buffers (causal masks, from the checkpoint's `attn.bias`) | 6 |
| inputs / constants | 2 / 6 |
| intermediates | 385 |
| tensor operations | 385 |
| **multiplications** (Contraction + elementwise ×) | **112** = 37 contractions + 75 elementwise |
| contraction multiply-accumulates (B=2, T=8) | 1,298,214,912 |
| modules with no ops (dropout, eval mode) | 19 |

## 1. Conventions

- Axes are named: `B` batch, `T` tokens, `q`/`k` query/key copies of `T`, `D` hidden (768), `Dc` the head-merged channel (768), `H` heads (12), `Dh` head dim (64), `F` MLP width (3072), `QKV` (2304), `V` vocabulary (50257), `P` positions (1024).
- `a · b` is a contraction: the axes `a` and `b` share and that are listed as `Σ` are summed; the others stay. `a × b`, `+`, `-`, `/` are elementwise and align operands by axis name (a bias `b[QKV]` meets `y[B,T,QKV]` by the axis `QKV`). Named axes make transposes unnecessary: `K.T` and the tied head's `E.T` are just contractions over the right axis.
- Parameters are bound to the checkpoint **in its original Hugging Face layout**: Conv1D weights are `[in,out]` and need no transposition here.
- `Mb_l` is the checkpoint's `attn.bias[0,0,:T,:T]` (1 = visible): `log(1) = 0`, `log(0) = -inf`, so adding `log(Mb_l)` is the causal mask.
- Dropout is the identity in `eval()` and emits no operation; the primitive names follow TEL's TensorIR.

## 2. What each module became

| module type | instances | ops emitted | lowering |
|---|---|---|---|
| `DistilGPT2LMHeadModel` | 1 | 0 | no own ops; the tied head is a Contraction of Xf with E over D |
| `GPT2Model` | 1 | 0 | no own ops; chains embeddings, 6 blocks, final LayerNorm |
| `GPT2Embeddings` | 1 | 1 | X_0 = token rows + position rows (1 add); dropout elided |
| `TokenEmbedding` | 1 | 1 | Gather: rows of E selected by ids (= one-hot(ids) · E) |
| `PositionEmbedding` | 1 | 1 | Gather: rows of Wp selected by pos |
| `Dropout` | 19 | 0 | identity in eval(): 0 ops |
| `TransformerBlock` | 6 | 12 (2 each) | the two residual additions |
| `LayerNorm` | 13 | 117 (9 each) | 9 ops: mean, sub, square, mean, +eps, rsqrt, ·, ·gamma, +beta |
| `CausalSelfAttention` | 6 | 150 (25 each) | head split (Slice, Reshape, Permute, Rename ×3), scores, mask, softmax, context, head merge |
| `Linear` | 25 | 49 (1.96 each) | Contraction x·W, then + bias (the lm_head has no bias: 1 op) |
| `GPT2MLP` | 6 | 0 | no own ops; fc, GELU, projection |
| `GPT2GELU` | 6 | 54 (9 each) | 9 ops: x·x, ·x, ·0.044715, +x, ·√(2/π), tanh, +1, x·0.5, product |

## 3. All tensor multiplications (112)

`·` contraction (matrix/tensor product), `×` elementwise product. `cost` is multiply-accumulates for a contraction and output elements for `×` (B=2, T=8).

| # | op | kind | equation | cost | module |
|---|---|---|---|---|---|
| 1 | 005 | mul | `N1_0.squared[B,T,D] = N1_0.centered[B,T,D] × N1_0.centered[B,T,D]` | 12,288 | `transformer.blocks.0.ln_1` |
| 2 | 009 | mul | `N1_0.normalized[B,T,D] = N1_0.centered[B,T,D] × N1_0.rstd[B,T]` | 12,288 | `transformer.blocks.0.ln_1` |
| 3 | 010 | mul | `N1_0.scaled[B,T,D] = N1_0.normalized[B,T,D] × g1_0[D]` | 12,288 | `transformer.blocks.0.ln_1` |
| 4 | 012 | contract | `Hq_0.mm[B,T,QKV] = N1_0[B,T,D] · W_qkv_0[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.0.attention.qkv_projection` |
| 5 | 026 | contract | `S_0.qk[B,H,q,k] = Q_0[B,H,q,Dh] · K_0[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.0.attention` |
| 6 | 035 | contract | `C_0[B,H,q,Dh] = A_0[B,H,q,k] · V_0[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.0.attention` |
| 7 | 039 | contract | `O_0.mm[B,T,D] = C_0.merged[B,T,Dc] · W_o_0[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.0.attention.output_projection` |
| 8 | 044 | mul | `N2_0.squared[B,T,D] = N2_0.centered[B,T,D] × N2_0.centered[B,T,D]` | 12,288 | `transformer.blocks.0.ln_2` |
| 9 | 048 | mul | `N2_0.normalized[B,T,D] = N2_0.centered[B,T,D] × N2_0.rstd[B,T]` | 12,288 | `transformer.blocks.0.ln_2` |
| 10 | 049 | mul | `N2_0.scaled[B,T,D] = N2_0.normalized[B,T,D] × g2_0[D]` | 12,288 | `transformer.blocks.0.ln_2` |
| 11 | 051 | contract | `G_0.mm[B,T,F] = N2_0[B,T,D] · W_fc_0[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.0.mlp.fc` |
| 12 | 053 | mul | `Ga_0.squared[B,T,F] = G_0[B,T,F] × G_0[B,T,F]` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 13 | 054 | mul | `Ga_0.cubic[B,T,F] = Ga_0.squared[B,T,F] × G_0[B,T,F]` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 14 | 055 | mul | `Ga_0.cubic_a[B,T,F] = Ga_0.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 15 | 057 | mul | `Ga_0.inner[B,T,F] = Ga_0.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 16 | 060 | mul | `Ga_0.half_x[B,T,F] = G_0[B,T,F] × c_half` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 17 | 061 | mul | `Ga_0[B,T,F] = Ga_0.half_x[B,T,F] × Ga_0.one_plus[B,T,F]` | 49,152 | `transformer.blocks.0.mlp.activation` |
| 18 | 062 | contract | `M_0.mm[B,T,D] = Ga_0[B,T,F] · W_pr_0[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.0.mlp.projection` |
| 19 | 067 | mul | `N1_1.squared[B,T,D] = N1_1.centered[B,T,D] × N1_1.centered[B,T,D]` | 12,288 | `transformer.blocks.1.ln_1` |
| 20 | 071 | mul | `N1_1.normalized[B,T,D] = N1_1.centered[B,T,D] × N1_1.rstd[B,T]` | 12,288 | `transformer.blocks.1.ln_1` |
| 21 | 072 | mul | `N1_1.scaled[B,T,D] = N1_1.normalized[B,T,D] × g1_1[D]` | 12,288 | `transformer.blocks.1.ln_1` |
| 22 | 074 | contract | `Hq_1.mm[B,T,QKV] = N1_1[B,T,D] · W_qkv_1[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.1.attention.qkv_projection` |
| 23 | 088 | contract | `S_1.qk[B,H,q,k] = Q_1[B,H,q,Dh] · K_1[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.1.attention` |
| 24 | 097 | contract | `C_1[B,H,q,Dh] = A_1[B,H,q,k] · V_1[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.1.attention` |
| 25 | 101 | contract | `O_1.mm[B,T,D] = C_1.merged[B,T,Dc] · W_o_1[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.1.attention.output_projection` |
| 26 | 106 | mul | `N2_1.squared[B,T,D] = N2_1.centered[B,T,D] × N2_1.centered[B,T,D]` | 12,288 | `transformer.blocks.1.ln_2` |
| 27 | 110 | mul | `N2_1.normalized[B,T,D] = N2_1.centered[B,T,D] × N2_1.rstd[B,T]` | 12,288 | `transformer.blocks.1.ln_2` |
| 28 | 111 | mul | `N2_1.scaled[B,T,D] = N2_1.normalized[B,T,D] × g2_1[D]` | 12,288 | `transformer.blocks.1.ln_2` |
| 29 | 113 | contract | `G_1.mm[B,T,F] = N2_1[B,T,D] · W_fc_1[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.1.mlp.fc` |
| 30 | 115 | mul | `Ga_1.squared[B,T,F] = G_1[B,T,F] × G_1[B,T,F]` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 31 | 116 | mul | `Ga_1.cubic[B,T,F] = Ga_1.squared[B,T,F] × G_1[B,T,F]` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 32 | 117 | mul | `Ga_1.cubic_a[B,T,F] = Ga_1.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 33 | 119 | mul | `Ga_1.inner[B,T,F] = Ga_1.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 34 | 122 | mul | `Ga_1.half_x[B,T,F] = G_1[B,T,F] × c_half` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 35 | 123 | mul | `Ga_1[B,T,F] = Ga_1.half_x[B,T,F] × Ga_1.one_plus[B,T,F]` | 49,152 | `transformer.blocks.1.mlp.activation` |
| 36 | 124 | contract | `M_1.mm[B,T,D] = Ga_1[B,T,F] · W_pr_1[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.1.mlp.projection` |
| 37 | 129 | mul | `N1_2.squared[B,T,D] = N1_2.centered[B,T,D] × N1_2.centered[B,T,D]` | 12,288 | `transformer.blocks.2.ln_1` |
| 38 | 133 | mul | `N1_2.normalized[B,T,D] = N1_2.centered[B,T,D] × N1_2.rstd[B,T]` | 12,288 | `transformer.blocks.2.ln_1` |
| 39 | 134 | mul | `N1_2.scaled[B,T,D] = N1_2.normalized[B,T,D] × g1_2[D]` | 12,288 | `transformer.blocks.2.ln_1` |
| 40 | 136 | contract | `Hq_2.mm[B,T,QKV] = N1_2[B,T,D] · W_qkv_2[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.2.attention.qkv_projection` |
| 41 | 150 | contract | `S_2.qk[B,H,q,k] = Q_2[B,H,q,Dh] · K_2[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.2.attention` |
| 42 | 159 | contract | `C_2[B,H,q,Dh] = A_2[B,H,q,k] · V_2[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.2.attention` |
| 43 | 163 | contract | `O_2.mm[B,T,D] = C_2.merged[B,T,Dc] · W_o_2[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.2.attention.output_projection` |
| 44 | 168 | mul | `N2_2.squared[B,T,D] = N2_2.centered[B,T,D] × N2_2.centered[B,T,D]` | 12,288 | `transformer.blocks.2.ln_2` |
| 45 | 172 | mul | `N2_2.normalized[B,T,D] = N2_2.centered[B,T,D] × N2_2.rstd[B,T]` | 12,288 | `transformer.blocks.2.ln_2` |
| 46 | 173 | mul | `N2_2.scaled[B,T,D] = N2_2.normalized[B,T,D] × g2_2[D]` | 12,288 | `transformer.blocks.2.ln_2` |
| 47 | 175 | contract | `G_2.mm[B,T,F] = N2_2[B,T,D] · W_fc_2[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.2.mlp.fc` |
| 48 | 177 | mul | `Ga_2.squared[B,T,F] = G_2[B,T,F] × G_2[B,T,F]` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 49 | 178 | mul | `Ga_2.cubic[B,T,F] = Ga_2.squared[B,T,F] × G_2[B,T,F]` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 50 | 179 | mul | `Ga_2.cubic_a[B,T,F] = Ga_2.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 51 | 181 | mul | `Ga_2.inner[B,T,F] = Ga_2.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 52 | 184 | mul | `Ga_2.half_x[B,T,F] = G_2[B,T,F] × c_half` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 53 | 185 | mul | `Ga_2[B,T,F] = Ga_2.half_x[B,T,F] × Ga_2.one_plus[B,T,F]` | 49,152 | `transformer.blocks.2.mlp.activation` |
| 54 | 186 | contract | `M_2.mm[B,T,D] = Ga_2[B,T,F] · W_pr_2[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.2.mlp.projection` |
| 55 | 191 | mul | `N1_3.squared[B,T,D] = N1_3.centered[B,T,D] × N1_3.centered[B,T,D]` | 12,288 | `transformer.blocks.3.ln_1` |
| 56 | 195 | mul | `N1_3.normalized[B,T,D] = N1_3.centered[B,T,D] × N1_3.rstd[B,T]` | 12,288 | `transformer.blocks.3.ln_1` |
| 57 | 196 | mul | `N1_3.scaled[B,T,D] = N1_3.normalized[B,T,D] × g1_3[D]` | 12,288 | `transformer.blocks.3.ln_1` |
| 58 | 198 | contract | `Hq_3.mm[B,T,QKV] = N1_3[B,T,D] · W_qkv_3[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.3.attention.qkv_projection` |
| 59 | 212 | contract | `S_3.qk[B,H,q,k] = Q_3[B,H,q,Dh] · K_3[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.3.attention` |
| 60 | 221 | contract | `C_3[B,H,q,Dh] = A_3[B,H,q,k] · V_3[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.3.attention` |
| 61 | 225 | contract | `O_3.mm[B,T,D] = C_3.merged[B,T,Dc] · W_o_3[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.3.attention.output_projection` |
| 62 | 230 | mul | `N2_3.squared[B,T,D] = N2_3.centered[B,T,D] × N2_3.centered[B,T,D]` | 12,288 | `transformer.blocks.3.ln_2` |
| 63 | 234 | mul | `N2_3.normalized[B,T,D] = N2_3.centered[B,T,D] × N2_3.rstd[B,T]` | 12,288 | `transformer.blocks.3.ln_2` |
| 64 | 235 | mul | `N2_3.scaled[B,T,D] = N2_3.normalized[B,T,D] × g2_3[D]` | 12,288 | `transformer.blocks.3.ln_2` |
| 65 | 237 | contract | `G_3.mm[B,T,F] = N2_3[B,T,D] · W_fc_3[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.3.mlp.fc` |
| 66 | 239 | mul | `Ga_3.squared[B,T,F] = G_3[B,T,F] × G_3[B,T,F]` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 67 | 240 | mul | `Ga_3.cubic[B,T,F] = Ga_3.squared[B,T,F] × G_3[B,T,F]` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 68 | 241 | mul | `Ga_3.cubic_a[B,T,F] = Ga_3.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 69 | 243 | mul | `Ga_3.inner[B,T,F] = Ga_3.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 70 | 246 | mul | `Ga_3.half_x[B,T,F] = G_3[B,T,F] × c_half` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 71 | 247 | mul | `Ga_3[B,T,F] = Ga_3.half_x[B,T,F] × Ga_3.one_plus[B,T,F]` | 49,152 | `transformer.blocks.3.mlp.activation` |
| 72 | 248 | contract | `M_3.mm[B,T,D] = Ga_3[B,T,F] · W_pr_3[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.3.mlp.projection` |
| 73 | 253 | mul | `N1_4.squared[B,T,D] = N1_4.centered[B,T,D] × N1_4.centered[B,T,D]` | 12,288 | `transformer.blocks.4.ln_1` |
| 74 | 257 | mul | `N1_4.normalized[B,T,D] = N1_4.centered[B,T,D] × N1_4.rstd[B,T]` | 12,288 | `transformer.blocks.4.ln_1` |
| 75 | 258 | mul | `N1_4.scaled[B,T,D] = N1_4.normalized[B,T,D] × g1_4[D]` | 12,288 | `transformer.blocks.4.ln_1` |
| 76 | 260 | contract | `Hq_4.mm[B,T,QKV] = N1_4[B,T,D] · W_qkv_4[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.4.attention.qkv_projection` |
| 77 | 274 | contract | `S_4.qk[B,H,q,k] = Q_4[B,H,q,Dh] · K_4[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.4.attention` |
| 78 | 283 | contract | `C_4[B,H,q,Dh] = A_4[B,H,q,k] · V_4[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.4.attention` |
| 79 | 287 | contract | `O_4.mm[B,T,D] = C_4.merged[B,T,Dc] · W_o_4[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.4.attention.output_projection` |
| 80 | 292 | mul | `N2_4.squared[B,T,D] = N2_4.centered[B,T,D] × N2_4.centered[B,T,D]` | 12,288 | `transformer.blocks.4.ln_2` |
| 81 | 296 | mul | `N2_4.normalized[B,T,D] = N2_4.centered[B,T,D] × N2_4.rstd[B,T]` | 12,288 | `transformer.blocks.4.ln_2` |
| 82 | 297 | mul | `N2_4.scaled[B,T,D] = N2_4.normalized[B,T,D] × g2_4[D]` | 12,288 | `transformer.blocks.4.ln_2` |
| 83 | 299 | contract | `G_4.mm[B,T,F] = N2_4[B,T,D] · W_fc_4[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.4.mlp.fc` |
| 84 | 301 | mul | `Ga_4.squared[B,T,F] = G_4[B,T,F] × G_4[B,T,F]` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 85 | 302 | mul | `Ga_4.cubic[B,T,F] = Ga_4.squared[B,T,F] × G_4[B,T,F]` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 86 | 303 | mul | `Ga_4.cubic_a[B,T,F] = Ga_4.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 87 | 305 | mul | `Ga_4.inner[B,T,F] = Ga_4.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 88 | 308 | mul | `Ga_4.half_x[B,T,F] = G_4[B,T,F] × c_half` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 89 | 309 | mul | `Ga_4[B,T,F] = Ga_4.half_x[B,T,F] × Ga_4.one_plus[B,T,F]` | 49,152 | `transformer.blocks.4.mlp.activation` |
| 90 | 310 | contract | `M_4.mm[B,T,D] = Ga_4[B,T,F] · W_pr_4[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.4.mlp.projection` |
| 91 | 315 | mul | `N1_5.squared[B,T,D] = N1_5.centered[B,T,D] × N1_5.centered[B,T,D]` | 12,288 | `transformer.blocks.5.ln_1` |
| 92 | 319 | mul | `N1_5.normalized[B,T,D] = N1_5.centered[B,T,D] × N1_5.rstd[B,T]` | 12,288 | `transformer.blocks.5.ln_1` |
| 93 | 320 | mul | `N1_5.scaled[B,T,D] = N1_5.normalized[B,T,D] × g1_5[D]` | 12,288 | `transformer.blocks.5.ln_1` |
| 94 | 322 | contract | `Hq_5.mm[B,T,QKV] = N1_5[B,T,D] · W_qkv_5[D,QKV]   (Σ D)` | 28,311,552 | `transformer.blocks.5.attention.qkv_projection` |
| 95 | 336 | contract | `S_5.qk[B,H,q,k] = Q_5[B,H,q,Dh] · K_5[B,H,k,Dh]   (Σ Dh)` | 98,304 | `transformer.blocks.5.attention` |
| 96 | 345 | contract | `C_5[B,H,q,Dh] = A_5[B,H,q,k] · V_5[B,H,k,Dh]   (Σ k)` | 98,304 | `transformer.blocks.5.attention` |
| 97 | 349 | contract | `O_5.mm[B,T,D] = C_5.merged[B,T,Dc] · W_o_5[Dc,D]   (Σ Dc)` | 9,437,184 | `transformer.blocks.5.attention.output_projection` |
| 98 | 354 | mul | `N2_5.squared[B,T,D] = N2_5.centered[B,T,D] × N2_5.centered[B,T,D]` | 12,288 | `transformer.blocks.5.ln_2` |
| 99 | 358 | mul | `N2_5.normalized[B,T,D] = N2_5.centered[B,T,D] × N2_5.rstd[B,T]` | 12,288 | `transformer.blocks.5.ln_2` |
| 100 | 359 | mul | `N2_5.scaled[B,T,D] = N2_5.normalized[B,T,D] × g2_5[D]` | 12,288 | `transformer.blocks.5.ln_2` |
| 101 | 361 | contract | `G_5.mm[B,T,F] = N2_5[B,T,D] · W_fc_5[D,F]   (Σ D)` | 37,748,736 | `transformer.blocks.5.mlp.fc` |
| 102 | 363 | mul | `Ga_5.squared[B,T,F] = G_5[B,T,F] × G_5[B,T,F]` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 103 | 364 | mul | `Ga_5.cubic[B,T,F] = Ga_5.squared[B,T,F] × G_5[B,T,F]` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 104 | 365 | mul | `Ga_5.cubic_a[B,T,F] = Ga_5.cubic[B,T,F] × c_gelu_a` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 105 | 367 | mul | `Ga_5.inner[B,T,F] = Ga_5.inner_sum[B,T,F] × c_gelu_b` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 106 | 370 | mul | `Ga_5.half_x[B,T,F] = G_5[B,T,F] × c_half` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 107 | 371 | mul | `Ga_5[B,T,F] = Ga_5.half_x[B,T,F] × Ga_5.one_plus[B,T,F]` | 49,152 | `transformer.blocks.5.mlp.activation` |
| 108 | 372 | contract | `M_5.mm[B,T,D] = Ga_5[B,T,F] · W_pr_5[F,D]   (Σ F)` | 37,748,736 | `transformer.blocks.5.mlp.projection` |
| 109 | 377 | mul | `Xf.squared[B,T,D] = Xf.centered[B,T,D] × Xf.centered[B,T,D]` | 12,288 | `transformer.final_layer_norm` |
| 110 | 381 | mul | `Xf.normalized[B,T,D] = Xf.centered[B,T,D] × Xf.rstd[B,T]` | 12,288 | `transformer.final_layer_norm` |
| 111 | 382 | mul | `Xf.scaled[B,T,D] = Xf.normalized[B,T,D] × gf[D]` | 12,288 | `transformer.final_layer_norm` |
| 112 | 384 | contract | `L[B,T,V] = Xf[B,T,D] · E[V,D]   (Σ D)` | 617,558,016 | `lm_head` |

Per block (all six are identical in structure):

| block | contractions | elementwise × | contraction MACs |
|---|---|---|---|
| 0 | 6 | 12 | 113,442,816 |
| 1 | 6 | 12 | 113,442,816 |
| 2 | 6 | 12 | 113,442,816 |
| 3 | 6 | 12 | 113,442,816 |
| 4 | 6 | 12 | 113,442,816 |
| 5 | 6 | 12 | 113,442,816 |

## 4. The other tensor operations

Not multiplications: sums and differences (bias, residual, eps, mask, softmax shift, GELU terms), division (score scale, softmax normalisation), the nonlinearities `exp`, `tanh`, `rsqrt`, `log`, reductions, and data movement. They are in the full listing (`artifacts/tensor_program.txt`).

| primitive | fn | count |
|---|---|---|
| Elementwise | add | 81 |
| Elementwise | sub | 19 |
| Elementwise | rsqrt | 13 |
| Elementwise | div | 12 |
| Elementwise | log | 6 |
| Elementwise | exp | 6 |
| Elementwise | tanh | 6 |
| Gather |  | 2 |
| Permute |  | 24 |
| Reduce | mean | 26 |
| Reduce | max | 6 |
| Reduce | sum | 6 |
| Rename |  | 24 |
| Reshape |  | 24 |
| Slice |  | 18 |

The embedding lookup is a Gather; as a multiplication it is `one_hot(ids)[B,T,V] · E[V,D]` (617,611,328 MACs at B=2, T=8), which selects the same rows exactly.

## 5. Tensors

### Parameters and buffers (checkpoint keys, original layout)

| symbol | role | checkpoint key | axes | shape | count |
|---|---|---|---|---|---|
| `E` | parameter | `transformer.wte.weight` | [V,D] | 50257 × 768 | 1 |
| `Wp` | parameter | `transformer.wpe.weight` | [P,D] | 1024 × 768 | 1 |
| `g1_{l}` | parameter | `transformer.h.{l}.ln_1.weight` | [D] | 768 | 6 blocks |
| `b1_{l}` | parameter | `transformer.h.{l}.ln_1.bias` | [D] | 768 | 6 blocks |
| `W_qkv_{l}` | parameter | `transformer.h.{l}.attn.c_attn.weight` | [D,QKV] | 768 × 2304 | 6 blocks |
| `b_qkv_{l}` | parameter | `transformer.h.{l}.attn.c_attn.bias` | [QKV] | 2304 | 6 blocks |
| `Mb_{l}` | buffer | `transformer.h.{l}.attn.bias` | [q,k] | 1 × 1 × P × P, cut [0,0,:T,:T] | 6 blocks |
| `W_o_{l}` | parameter | `transformer.h.{l}.attn.c_proj.weight` | [Dc,D] | 768 × 768 | 6 blocks |
| `b_o_{l}` | parameter | `transformer.h.{l}.attn.c_proj.bias` | [D] | 768 | 6 blocks |
| `g2_{l}` | parameter | `transformer.h.{l}.ln_2.weight` | [D] | 768 | 6 blocks |
| `b2_{l}` | parameter | `transformer.h.{l}.ln_2.bias` | [D] | 768 | 6 blocks |
| `W_fc_{l}` | parameter | `transformer.h.{l}.mlp.c_fc.weight` | [D,F] | 768 × 3072 | 6 blocks |
| `b_fc_{l}` | parameter | `transformer.h.{l}.mlp.c_fc.bias` | [F] | 3072 | 6 blocks |
| `W_pr_{l}` | parameter | `transformer.h.{l}.mlp.c_proj.weight` | [F,D] | 3072 × 768 | 6 blocks |
| `b_pr_{l}` | parameter | `transformer.h.{l}.mlp.c_proj.bias` | [D] | 768 | 6 blocks |
| `gf` | parameter | `transformer.ln_f.weight` | [D] | 768 | 1 |
| `bf` | parameter | `transformer.ln_f.bias` | [D] | 768 | 1 |

Tied weight: `lm_head.weight = E` (the same tensor, used as the embedding table and as the output matrix).

### Inputs and constants

| tensor | role | axes / value |
|---|---|---|
| `ids` | input | [B,T] int64 |
| `pos` | input | [T] int64 |
| `c_eps` | constant | 1e-05 |
| `c_sqrt_dh` | constant | 8.0 |
| `c_gelu_a` | constant | 0.044715 |
| `c_gelu_b` | constant | 0.7978845608028654 |
| `c_half` | constant | 0.5 |
| `c_one` | constant | 1.0 |

### Intermediates and their PyTorch counterparts

Every symbol below is a tensor in this program **and** a tensor in `model(input_ids, return_debug=True).debug`, compared on every run.

| symbol | PyTorch debug name | axes | shape (B=2, T=8) |
|---|---|---|---|
| `X_0` | `embedding.output` | [B,T,D] | 2 × 8 × 768 |
| `N1_{l}` | `block.{l}.ln_1.output` | [B,T,D] | 2 × 8 × 768 |
| `Hq_{l}` | `block.{l}.attention.qkv` | [B,T,QKV] | 2 × 8 × 2304 |
| `Q_{l}` | `block.{l}.attention.q` | [B,H,q,Dh] | 2 × 12 × 8 × 64 |
| `K_{l}` | `block.{l}.attention.k` | [B,H,k,Dh] | 2 × 12 × 8 × 64 |
| `V_{l}` | `block.{l}.attention.v` | [B,H,k,Dh] | 2 × 12 × 8 × 64 |
| `S_{l}` | `block.{l}.attention.scores` | [B,H,q,k] | 2 × 12 × 8 × 8 |
| `A_{l}` | `block.{l}.attention.probs` | [B,H,q,k] | 2 × 12 × 8 × 8 |
| `C_{l}` | `block.{l}.attention.context` | [B,H,q,Dh] | 2 × 12 × 8 × 64 |
| `O_{l}` | `block.{l}.attention.output` | [B,T,D] | 2 × 8 × 768 |
| `Xm_{l}` | `block.{l}.attention_residual.output` | [B,T,D] | 2 × 8 × 768 |
| `N2_{l}` | `block.{l}.ln_2.output` | [B,T,D] | 2 × 8 × 768 |
| `G_{l}` | `block.{l}.mlp.fc.output` | [B,T,F] | 2 × 8 × 3072 |
| `Ga_{l}` | `block.{l}.mlp.activation.output` | [B,T,F] | 2 × 8 × 3072 |
| `M_{l}` | `block.{l}.mlp.output` | [B,T,D] | 2 × 8 × 768 |
| `X_{l+1}` | `block.{l}.output` | [B,T,D] | 2 × 8 × 768 |
| `Xf` | `model.final_hidden` | [B,T,D] | 2 × 8 × 768 |
| `L` | `model.logits` | [B,T,V] | 2 × 8 × 50257 |

93 tensors are compared in total; the rest are finer intermediates (`<symbol>.<part>`) such as `N1_0.centered` or `A_0.exp`.

## 6. The program: embeddings, block 0, head

Blocks 1 to 5 repeat block 0 with the index changed (`_1` … `_5`). All 385 operations: `artifacts/tensor_program.txt`.

```text
000  Gather            X_0.token[B,T,D] = gather(E[V,D], ids[B,T])   (rows of V)
                       [transformer.embeddings.token_embedding]
001  Gather            X_0.position[T,D] = gather(Wp[P,D], pos[T])   (rows of P)
                       [transformer.embeddings.position_embedding]
002  Elementwise.add   X_0[B,T,D] = X_0.token[B,T,D] + X_0.position[T,D]
                       [transformer.embeddings]
003  Reduce.mean       N1_0.mean[B,T] = mean_D(X_0[B,T,D])
                       [transformer.blocks.0.ln_1]
004  Elementwise.sub   N1_0.centered[B,T,D] = X_0[B,T,D] - N1_0.mean[B,T]
                       [transformer.blocks.0.ln_1]
005  Elementwise.mul   N1_0.squared[B,T,D] = N1_0.centered[B,T,D] × N1_0.centered[B,T,D]
                       [transformer.blocks.0.ln_1]
006  Reduce.mean       N1_0.var[B,T] = mean_D(N1_0.squared[B,T,D])
                       [transformer.blocks.0.ln_1]
007  Elementwise.add   N1_0.var_eps[B,T] = N1_0.var[B,T] + c_eps
                       [transformer.blocks.0.ln_1]
008  Elementwise.rsqrt N1_0.rstd[B,T] = rsqrt(N1_0.var_eps[B,T])
                       [transformer.blocks.0.ln_1]
009  Elementwise.mul   N1_0.normalized[B,T,D] = N1_0.centered[B,T,D] × N1_0.rstd[B,T]
                       [transformer.blocks.0.ln_1]
010  Elementwise.mul   N1_0.scaled[B,T,D] = N1_0.normalized[B,T,D] × g1_0[D]
                       [transformer.blocks.0.ln_1]
011  Elementwise.add   N1_0[B,T,D] = N1_0.scaled[B,T,D] + b1_0[D]
                       [transformer.blocks.0.ln_1]
012  Contraction       Hq_0.mm[B,T,QKV] = N1_0[B,T,D] · W_qkv_0[D,QKV]   (Σ D)
                       [transformer.blocks.0.attention.qkv_projection]
013  Elementwise.add   Hq_0[B,T,QKV] = Hq_0.mm[B,T,QKV] + b_qkv_0[QKV]
                       [transformer.blocks.0.attention.qkv_projection]
014  Slice             Q_0.slice[B,T,Dc] = Hq_0[QKV 0:768]
                       [transformer.blocks.0.attention]
015  Reshape           Q_0.heads[B,T,H,Dh] = reshape(Q_0.slice[B,T,Dc])
                       [transformer.blocks.0.attention]
016  Permute           Q_0.perm[B,H,T,Dh] = permute(Q_0.heads[B,T,H,Dh])
                       [transformer.blocks.0.attention]
017  Rename            Q_0[B,H,q,Dh] = rename(Q_0.perm[B,H,T,Dh], T→q)
                       [transformer.blocks.0.attention]
018  Slice             K_0.slice[B,T,Dc] = Hq_0[QKV 768:1536]
                       [transformer.blocks.0.attention]
019  Reshape           K_0.heads[B,T,H,Dh] = reshape(K_0.slice[B,T,Dc])
                       [transformer.blocks.0.attention]
020  Permute           K_0.perm[B,H,T,Dh] = permute(K_0.heads[B,T,H,Dh])
                       [transformer.blocks.0.attention]
021  Rename            K_0[B,H,k,Dh] = rename(K_0.perm[B,H,T,Dh], T→k)
                       [transformer.blocks.0.attention]
022  Slice             V_0.slice[B,T,Dc] = Hq_0[QKV 1536:2304]
                       [transformer.blocks.0.attention]
023  Reshape           V_0.heads[B,T,H,Dh] = reshape(V_0.slice[B,T,Dc])
                       [transformer.blocks.0.attention]
024  Permute           V_0.perm[B,H,T,Dh] = permute(V_0.heads[B,T,H,Dh])
                       [transformer.blocks.0.attention]
025  Rename            V_0[B,H,k,Dh] = rename(V_0.perm[B,H,T,Dh], T→k)
                       [transformer.blocks.0.attention]
026  Contraction       S_0.qk[B,H,q,k] = Q_0[B,H,q,Dh] · K_0[B,H,k,Dh]   (Σ Dh)
                       [transformer.blocks.0.attention]
027  Elementwise.div   S_0[B,H,q,k] = S_0.qk[B,H,q,k] / c_sqrt_dh
                       [transformer.blocks.0.attention]
028  Elementwise.log   Mb_0.log[q,k] = log(Mb_0[q,k])
                       [transformer.blocks.0.attention]
029  Elementwise.add   S_0.masked[B,H,q,k] = S_0[B,H,q,k] + Mb_0.log[q,k]
                       [transformer.blocks.0.attention]
030  Reduce.max        A_0.max[B,H,q] = max_k(S_0.masked[B,H,q,k])
                       [transformer.blocks.0.attention]
031  Elementwise.sub   A_0.shifted[B,H,q,k] = S_0.masked[B,H,q,k] - A_0.max[B,H,q]
                       [transformer.blocks.0.attention]
032  Elementwise.exp   A_0.exp[B,H,q,k] = exp(A_0.shifted[B,H,q,k])
                       [transformer.blocks.0.attention]
033  Reduce.sum        A_0.sum[B,H,q] = sum_k(A_0.exp[B,H,q,k])
                       [transformer.blocks.0.attention]
034  Elementwise.div   A_0[B,H,q,k] = A_0.exp[B,H,q,k] / A_0.sum[B,H,q]
                       [transformer.blocks.0.attention]
035  Contraction       C_0[B,H,q,Dh] = A_0[B,H,q,k] · V_0[B,H,k,Dh]   (Σ k)
                       [transformer.blocks.0.attention]
036  Rename            C_0.rename[B,H,T,Dh] = rename(C_0[B,H,q,Dh], q→T)
                       [transformer.blocks.0.attention]
037  Permute           C_0.perm[B,T,H,Dh] = permute(C_0.rename[B,H,T,Dh])
                       [transformer.blocks.0.attention]
038  Reshape           C_0.merged[B,T,Dc] = reshape(C_0.perm[B,T,H,Dh])
                       [transformer.blocks.0.attention]
039  Contraction       O_0.mm[B,T,D] = C_0.merged[B,T,Dc] · W_o_0[Dc,D]   (Σ Dc)
                       [transformer.blocks.0.attention.output_projection]
040  Elementwise.add   O_0[B,T,D] = O_0.mm[B,T,D] + b_o_0[D]
                       [transformer.blocks.0.attention.output_projection]
041  Elementwise.add   Xm_0[B,T,D] = X_0[B,T,D] + O_0[B,T,D]
                       [transformer.blocks.0]
042  Reduce.mean       N2_0.mean[B,T] = mean_D(Xm_0[B,T,D])
                       [transformer.blocks.0.ln_2]
043  Elementwise.sub   N2_0.centered[B,T,D] = Xm_0[B,T,D] - N2_0.mean[B,T]
                       [transformer.blocks.0.ln_2]
044  Elementwise.mul   N2_0.squared[B,T,D] = N2_0.centered[B,T,D] × N2_0.centered[B,T,D]
                       [transformer.blocks.0.ln_2]
045  Reduce.mean       N2_0.var[B,T] = mean_D(N2_0.squared[B,T,D])
                       [transformer.blocks.0.ln_2]
046  Elementwise.add   N2_0.var_eps[B,T] = N2_0.var[B,T] + c_eps
                       [transformer.blocks.0.ln_2]
047  Elementwise.rsqrt N2_0.rstd[B,T] = rsqrt(N2_0.var_eps[B,T])
                       [transformer.blocks.0.ln_2]
048  Elementwise.mul   N2_0.normalized[B,T,D] = N2_0.centered[B,T,D] × N2_0.rstd[B,T]
                       [transformer.blocks.0.ln_2]
049  Elementwise.mul   N2_0.scaled[B,T,D] = N2_0.normalized[B,T,D] × g2_0[D]
                       [transformer.blocks.0.ln_2]
050  Elementwise.add   N2_0[B,T,D] = N2_0.scaled[B,T,D] + b2_0[D]
                       [transformer.blocks.0.ln_2]
051  Contraction       G_0.mm[B,T,F] = N2_0[B,T,D] · W_fc_0[D,F]   (Σ D)
                       [transformer.blocks.0.mlp.fc]
052  Elementwise.add   G_0[B,T,F] = G_0.mm[B,T,F] + b_fc_0[F]
                       [transformer.blocks.0.mlp.fc]
053  Elementwise.mul   Ga_0.squared[B,T,F] = G_0[B,T,F] × G_0[B,T,F]
                       [transformer.blocks.0.mlp.activation]
054  Elementwise.mul   Ga_0.cubic[B,T,F] = Ga_0.squared[B,T,F] × G_0[B,T,F]
                       [transformer.blocks.0.mlp.activation]
055  Elementwise.mul   Ga_0.cubic_a[B,T,F] = Ga_0.cubic[B,T,F] × c_gelu_a
                       [transformer.blocks.0.mlp.activation]
056  Elementwise.add   Ga_0.inner_sum[B,T,F] = G_0[B,T,F] + Ga_0.cubic_a[B,T,F]
                       [transformer.blocks.0.mlp.activation]
057  Elementwise.mul   Ga_0.inner[B,T,F] = Ga_0.inner_sum[B,T,F] × c_gelu_b
                       [transformer.blocks.0.mlp.activation]
058  Elementwise.tanh  Ga_0.tanh[B,T,F] = tanh(Ga_0.inner[B,T,F])
                       [transformer.blocks.0.mlp.activation]
059  Elementwise.add   Ga_0.one_plus[B,T,F] = Ga_0.tanh[B,T,F] + c_one
                       [transformer.blocks.0.mlp.activation]
060  Elementwise.mul   Ga_0.half_x[B,T,F] = G_0[B,T,F] × c_half
                       [transformer.blocks.0.mlp.activation]
061  Elementwise.mul   Ga_0[B,T,F] = Ga_0.half_x[B,T,F] × Ga_0.one_plus[B,T,F]
                       [transformer.blocks.0.mlp.activation]
062  Contraction       M_0.mm[B,T,D] = Ga_0[B,T,F] · W_pr_0[F,D]   (Σ F)
                       [transformer.blocks.0.mlp.projection]
063  Elementwise.add   M_0[B,T,D] = M_0.mm[B,T,D] + b_pr_0[D]
                       [transformer.blocks.0.mlp.projection]
064  Elementwise.add   X_1[B,T,D] = Xm_0[B,T,D] + M_0[B,T,D]
                       [transformer.blocks.0]
375  Reduce.mean       Xf.mean[B,T] = mean_D(X_6[B,T,D])
                       [transformer.final_layer_norm]
376  Elementwise.sub   Xf.centered[B,T,D] = X_6[B,T,D] - Xf.mean[B,T]
                       [transformer.final_layer_norm]
377  Elementwise.mul   Xf.squared[B,T,D] = Xf.centered[B,T,D] × Xf.centered[B,T,D]
                       [transformer.final_layer_norm]
378  Reduce.mean       Xf.var[B,T] = mean_D(Xf.squared[B,T,D])
                       [transformer.final_layer_norm]
379  Elementwise.add   Xf.var_eps[B,T] = Xf.var[B,T] + c_eps
                       [transformer.final_layer_norm]
380  Elementwise.rsqrt Xf.rstd[B,T] = rsqrt(Xf.var_eps[B,T])
                       [transformer.final_layer_norm]
381  Elementwise.mul   Xf.normalized[B,T,D] = Xf.centered[B,T,D] × Xf.rstd[B,T]
                       [transformer.final_layer_norm]
382  Elementwise.mul   Xf.scaled[B,T,D] = Xf.normalized[B,T,D] × gf[D]
                       [transformer.final_layer_norm]
383  Elementwise.add   Xf[B,T,D] = Xf.scaled[B,T,D] + bf[D]
                       [transformer.final_layer_norm]
384  Contraction       L[B,T,V] = Xf[B,T,D] · E[V,D]   (Σ D)
                       [lm_head]
```

## 7. Verification

The program is executed from the checkpoint's own tensors (original HF layout, no transposition and no use of the pure model's weights) and every tensor that has a PyTorch counterpart is compared, with error relative to that tensor's max magnitude (tolerance 1e-5).

| sample | input shape | tensors compared | worst rel. error | worst tensor | logits max abs err | argmax equal | result |
|---|---|---|---|---|---|---|---|
| sample_single | [1, 8] | 93 | 2.53e-06 | `O_1` | 6.87e-05 | True | PASS |
| sample_batch | [2, 8] | 93 | 3.54e-06 | `O_1` | 1.07e-04 | True | PASS |
| sample_repeated_tokens | [2, 8] | 93 | 3.72e-06 | `C_3` | 9.92e-05 | True | PASS |
| sample_short_sequence | [2, 4] | 93 | 4.70e-06 | `O_2` | 6.87e-05 | True | PASS |
| sample_attention_debug | [1, 8] | 93 | 2.86e-06 | `A_2` | 7.63e-05 | True | PASS |

## 8. Cross-check with TEL's emitted program

TEL's importer (`tensorparser`, `packages/model-import`) compiled the same checkpoint into 591 primitive ops (`artifacts/tel/model.json`, produced by `node dist/cli.js artifacts/model.safetensors artifacts/tel --model distilbert/distilgpt2 --config test/fixtures/distilgpt2.config.json`). Same-named primitives, side by side:

| primitive | fn | this program | TEL | note |
|---|---|---|---|---|
| Broadcast |  | 0 | 144 | named-axis alignment is implicit in Elementwise here; TTIR makes it an explicit op |
| Constant |  | 6 | 43 | constants are declared tensors here, not ops (shared, so fewer) |
| Contraction |  | 37 | 37 | equal |
| Convert |  | 0 | 6 | TEL converts the integer `D_h` to fp32 |
| Elementwise | add | 81 | 81 | equal |
| Elementwise | div | 12 | 12 | equal |
| Elementwise | exp | 6 | 6 | equal |
| Elementwise | log | 6 | 6 | equal |
| Elementwise | mul | 75 | 63 | GELU's x³ is two multiplications here, one `pow` in TEL |
| Elementwise | pow | 0 | 6 | see mul |
| Elementwise | rsqrt | 13 | 13 | equal |
| Elementwise | sqrt | 0 | 6 | √Dh is the constant 8.0 here; TEL computes sqrt(D_h) at run time |
| Elementwise | sub | 19 | 19 | equal |
| Elementwise | tanh | 6 | 6 | equal |
| ExternalRef |  | 0 | 6 | TEL reads `D_h` as a runtime dimension tensor |
| Gather |  | 2 | 2 | equal |
| Permute |  | 24 | 31 | `K.T` and `E.T` need no transpose with named axes (7 in TEL) |
| Reduce | max | 6 | 6 | equal |
| Reduce | mean | 26 | 26 | equal |
| Reduce | sum | 6 | 6 | equal |
| Rename |  | 24 | 24 | equal |
| Reshape |  | 24 | 24 | equal |
| Slice |  | 18 | 18 | equal |
