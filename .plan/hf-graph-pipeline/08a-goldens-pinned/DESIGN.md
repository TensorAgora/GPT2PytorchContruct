# 08a DESIGN

## Goldens

Pinned 2026-11-21 in `csv/goldens.csv`; fixtures in `tests/fixtures/headers/<org>__<repo>@<sha>/`.

| golden | repo | format | revision | spelled config |
| --- | --- | --- | --- | --- |
| distilgpt2 | distilbert/distilgpt2 | safetensors | 2290a626 | tests/fixtures/spelled/distilgpt2.config.json |
| qwen2.5-7b-instruct | Qwen/Qwen2.5-7B-Instruct | safetensors | a09a3545 | …/qwen2.5-7b-instruct.config.json |
| mistral-7b-instruct-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | safetensors | c170c708 | …/mistral-7b-instruct-v0.3.config.json |
| phi-3.5-mini-instruct | microsoft/Phi-3.5-mini-instruct | safetensors | 2fe19245 | …/phi-3.5-mini-instruct.config.json |
| gemma-2-2b-it | google/gemma-2-2b-it (gated, manual) | safetensors | 299a8560 | …/gemma-2-2b-it.config.json |
| deepseek-v3 | deepseek-ai/DeepSeek-V3 | safetensors | e815299b | …/deepseek-v3.config.json |
| mixtral-8x7b-instruct-q4km | mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF (B-02) | gguf | 92bb790b | …/mixtral-8x7b-instruct-q4km.config.json |
| qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | safetensors | c3a462cc | …/qwen3.8-27b-fp8.config.json (the real one) |

The mixtral repo was confirmed by its header: stacked `blk.N.ffn_{gate,up,down}_exps.weight`, 323 tensors = the golden's
323 keyed parameters. gemma-2: no HF_TOKEN with accepted terms, so no fixture; its test skips with a message.

## Config deltas

Real config.json against the config each golden was spelled from (keys of the spelled config only):

| golden | delta | cause |
| --- | --- | --- |
| distilgpt2 | none | |
| qwen2.5-7b-instruct | rms_norm_eps 1e-5 → 1e-6; head_dim 128 → absent | arch.mjs uses eps 1e-5 for every llama-family model; head_dim is implicit (D / Hq = 128) |
| mistral-7b-instruct-v0.3 | rope_theta 500000 → 1e6; head_dim 128 → absent | arch.mjs THETA maps family `llama` to 500000; head_dim implicit |
| phi-3.5-mini-instruct | rope_theta 500000 → 10000; head_dim 96 → absent | same THETA table; head_dim implicit |
| gemma-2-2b-it | not checked (gated) | |
| deepseek-v3 | none | |
| mixtral-8x7b-instruct-q4km | GGUF repo, no config.json; the GGUF KV gives the same sizes, eps 1e-5 (float32) and freq_base 1e6 | |
| qwen3.8-27b-fp8 | none (the spelled config is the real one) | |

Constants (eps, rope tables) are not part of §8.1 structural equality, so none of these deltas changes a comparison.

## Remaining differences (pipeline from pinned headers vs json/*.json)

| golden | result | cause |
| --- | --- | --- |
| distilgpt2, qwen2.5, mistral, mixtral, qwen3.8 | 0 differences | qwen3.8 includes the FP8 scale views and declares `model.visual` (333) and `mtp` (22) as components: status partial |
| deepseek-v3 | 0 differences except `view` on the FP8 weights outside the routed experts | the arch.mjs golden names the FP8 scales only inside its expert views; §5.3 asks every weight stored in another form to name it, as qwen3.8's golden does. MTP layer 61 is a component (partial); `e_score_correction_bias` is aux |
| phi-3.5-mini-instruct | before 06d: no_graph (160 q/k/v and gate/up keys missing, 64 fused tensors unclaimed); after 06d: 0 differences except `checkpoint_key`/`view` on the 160 split parameters | the checkpoint stores fused `qkv_proj` and `gate_up_proj`; the golden names separate keys that do not exist. The 06d overlay binds them as row slices (`[0:3072,:]` of `qkv_proj`) |
| gemma-2-2b-it | skipped (gated) | |
