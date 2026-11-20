# 06e DESIGN

From design doc §7.4. Sidecar rules run before binding (`bind.verify` calls `sidecars.apply`): packed principals are
replaced by their logical shape, missing logical principals are synthesized, and every sidecar that fits a rule becomes
`aux` attached to its principal. A companion that fits no rule stays a stored tensor, so the verifier reports it
unclaimed. Nothing is dropped. Bindings are recorded, never decoded.

| scheme | announced by | pattern → principal | shape relation (rule binds only if it holds) | graph `view` of the principal |
| --- | --- | --- | --- | --- |
| fp8_block | `quantization_config.quant_method: fp8` (or an F8 principal) | `X.weight_scale_inv` → `X.weight` (F8_E4M3) | scale = ceil(weight / block), block = `weight_block_size` (default 128 × 128) | `dequantized: F8_E4M3 x X.weight_scale_inv [a, b] (block-wise)` (model_graph.py's text) |
| fp8_mistral | Mistral layout | `X.qscale_weight`, `X.qscale_act` → `X.weight` | scale has ≤ 1 element | `dequantized: F8 x X.qscale_weight (per tensor)` |
| mxfp4 | `quant_method: mxfp4` | `X_blocks` U8 `[…, R, C/32, 16]` + `X_scales` U8 `[…, R, C/32]` → `X` | blocks[:-1] == scales, blocks[-1] == 16; logical `[…, R, C]` with C = 32 · blocks[-2] | `mxfp4: X_blocks + X_scales (E2M1, block 32)` |
| gptq | `quant_method: gptq` | `P.qweight` I32 `[in·b/32, out]`, `P.qzeros`, `P.scales` `[in/g, out]`, `P.g_idx` → `P.weight` | qweight[1] == scales[1]; logical `[out, in]`, in = qweight[0] · 32/b | `gptq int{b}: P.qweight / .qzeros / .scales / .g_idx (group g)` |
| awq | `quant_method: awq` | `P.qweight` I32 `[in, out·b/32]`, `P.qzeros`, `P.scales` `[in/g, out]` → `P.weight` | scales[1] == qweight[1] · 32/b; logical `[out, in]` | `awq int{b}: P.qweight / .qzeros / .scales (group g)` |
| mlx | `quantization {bits, group_size}` in config.json | `P.weight` U32 `[out, in·b/32]` + `P.scales`, `P.biases` `[out, in/g]` → `P.weight` | scales == biases, scales[0] == weight[0]; logical `[out, in]`, in = weight[1] · 32/b | `mlx int{b}: P.weight + .scales / .biases (group g)` |
| bitsandbytes | `quant_method: bitsandbytes` or the suffixes | `P.weight.absmax`, `.quant_map`, `.nested_absmax`, `.nested_quant_map`, `.quant_state.*` → `P.weight` | the logical shape lives in the quant_state payload, which is not read: the principal binds by name, shape unchecked | `bitsandbytes 4-bit: logical shape in quant_state (not read)` |
| gguf | the ggml type | none (no sidecar tensors) | — | none (dtype in the manifest) |

When no `quant_method` is announced, the relations themselves select gptq vs awq. b = `bits` (default 4), g =
`group_size` (default 128).

`apply(manifest, cfg) -> Sidecars(stored, aux {sidecar: principal}, views {principal: text}, records [{tensor,
principal, scheme}])`: `stored` is the manifest's tensors with logical principals in place of packed or missing ones.
report.json `sidecars` = `records`.
