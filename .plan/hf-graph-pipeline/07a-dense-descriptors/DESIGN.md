# 07a DESIGN

## Chosen keys (decoder_dense, catalog rows in scope, safetensors + GGUF)

| key | rows | descriptor | differences from `llama` | fixture |
| --- | --: | --- | --- | --- |
| llama | 1533 | llama (06a) | — | HuggingFaceTB/SmolLM2-135M |
| qwen3 | 955 | **qwen3** | per-head `q_norm` / `k_norm` RMSNorm over Dh before RoPE; no bias; a tied head may still be stored (class tied, shape-checked) | Qwen/Qwen3-0.6B |
| qwen2 | 817 | qwen2 (06a) | q/k/v bias | Qwen/Qwen2.5-7B-Instruct |
| mistral | 392 | mistral (06a) | — | mistralai/Mistral-7B-Instruct-v0.3 |
| hf_olmo | 354 | **hf_olmo** | AI2 OLMo (DataDecide): `model.transformer.blocks.N`, fused `att_proj` (q, k, v) and `ff_proj` (stored up then gate) bound through the overlay split; RMS norms with affine; padded vocab `embedding_size` | allenai/DataDecide-c4-10M |
| gemma2 | 101 | **gemma2** (real; the arch.mjs spelling is `gemma2_arch`) | (1+w) RMSNorms, post-attention / pre- and post-feed-forward norms, GeGLU (gelu_tanh), embedding × √D, attention and final soft-capping, query_pre_attn_scalar | unsloth/gemma-2-2b-it |
| gemma3_text | 95 | **gemma3_text** | gemma2 + q/k norms, no soft-capping, every 6th layer global (rope_theta), the rest local (rope_local_base_freq) | unsloth/gemma-3-1b-it |
| phi3 | 75 | phi3 (06a/06d) | fused qkv / gate_up (overlay split) | microsoft/Phi-3.5-mini-instruct |

Predicted decoder_dense rows unlocked by the new descriptors: qwen3 955 + hf_olmo 354 + gemma2 101 + gemma3_text 95 =
**1,505** (the GGUF rows of llama / qwen2 / qwen3 / mistral / gemma also bind now that 06d generates every GGUF scheme).
Expected misses: repos that are adapters only, remote-code repos whose architectures string matches none, and hf_olmo
variants with `layer_norm_type: default` (LayerNorm, not RMSNorm).

Not chosen: (unknown) 202 (no key: R3 fingerprint), gemma3 82 and gemma4 80 (VLM wrappers: 07d), granite 77 (llama tensors;
R3 proposes llama, `ok_assumed`, multipliers not drawn), the long tail under 80 rows (path B, 06f).

## Conformance runner

`python -m tools.conformance ID... | --all` (C1 fixture binding, C3 gguf-py names, C4 totals vs golden or the per-block
closed form, C5 hostile inputs, C6 determinism); `tests/test_conformance.py` runs it for every descriptor. The closed form
(`block_counts`) is checked against every golden it applies to.
