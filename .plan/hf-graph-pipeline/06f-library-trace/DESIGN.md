# 06f DESIGN

## Build

`registry/trace.py`. `make_model(library, cls, cfg, kind)` builds the class on the `meta` device from the
files' config, with no weights and nothing remote:
- timm: `timm.create_model(<architecture>, pretrained=False)`
- diffusers: `getattr(diffusers, _class_name).from_config(cfg)`, with the denoiser folder first (`UNet*`, `*Transformer*`).
  Every other folder goes into `components` as `declared_without_graph`.
- transformers: `AutoConfig.for_model(model_type)` with `AutoModelForCausalLM`, `AutoModel` or `AutoModelForSeq2SeqLM`.
  `trust_remote_code=False` is always set. A config that needs remote code (an `auto_map` without a native class) is
  skipped, never run.

Export uses `torch.export.export(model, args, strict=False)` on meta example inputs (B=1, T=8, image 3×input_size,
latents 1×C×min(sample,16)²). torch.fx symbolic tracing was not used: it breaks on HF control flow. If export fails
(data-dependent guards and the like), the run records a near-miss `path B: TraceError` and nothing is guessed.

## ATen → Program (ops a tiny resnet18 and a tiny UNet2DConditionModel produce)

| ATen | Program | note |
| --- | --- | --- |
| linear, addmm, mm (weight operand) | Rename weight → Contraction over x's last axis → Rename → Elementwise add (bias) | the weight keeps one declaration (ties); `mm`/`addmm` read `[in,out]` (GPT-2 Conv1D) |
| matmul (weight left, e.g. Mamba dt) | Rename → Contraction over x axis −2 → Permute | |
| add, sub, mul, div (tensor or scalar) | Elementwise add/sub/mul/div | scalars become constants `c_<v>` |
| relu, sigmoid, tanh, exp, log, rsqrt, erf | Elementwise unary | |
| view, reshape, _unsafe_view, flatten | Reshape | |
| embedding | Gather | |
| dropout, clone, contiguous, alias, detach, _to_copy, to, type_as, _assert_tensor_metadata | identity | dtypes are the checkpoint's |
| conv2d, batch_norm, group_norm, layer_norm, max_pool2d, adaptive_avg_pool2d, upsample_nearest2d, scaled_dot_product_attention, silu, gelu, sin, cos, cat, slice, chunk/getitem, unsqueeze, expand, permute, transpose, arange | `Opaque` (D5), `fn` = ATen name, ports typed by input names, output axes from the meta shape | |

Axes are named right-aligned by size (`r<pos>_<size>`), so broadcasting is TEL's subsequence rule. The graph output is
renamed `L`. Ops L does not depend on (masks, side outputs) are pruned.

## Binding and provenance

Parameters and buffers keep their placeholder names with `checkpoint_key = state_dict_key = ` the export signature's
target, which is the stored key. `bind.verify` runs as on path A (ignored: `*.num_batches_tracked`, `*.position_ids`).
The descriptor is `trace_<lib>_<cls>@<installed library version>` with certainty `traced`, path `B`.
`resolve_and_bind` falls back to path B only when no path-A candidate binds and no descriptor was forced.

## C2

`tools/conformance.py --c2 <ids|all>` takes each path-A descriptor whose fixture config names a transformers
`model_type`. GGUF fixtures use the spelled HF config. The config is cut to 2 layers (`first_k_dense_replace` ≤ 1,
`rope_scaling` dropped), then both paths are built. They are compared on weight-contraction FLOPs per token
(contractions with a parameter operand or a Rename/Permute/Reshape view of one; `experts` and `pooler` modules are left
out, so dense-vs-routed MoE does not count).
Result: PASS for all 19 descriptors with an upstream class; 11 are n/a (VLM assemblies, golden-only, no fixture config).

## Files

`registry/trace.py` (new), `registry/resolve.py` (`_path_b`, `Resolution.components`), `hub/pipeline.py` (path and
components from the resolution), `tensorprog/program.py` (Opaque equation), `schemas/viz-graph-v1.schema.json`
(`Opaque` enum), `registry/descriptor.py` (index_sets clipped to L), `tools/conformance.py` (`--c2`),
`tests/test_trace.py`, `tests/test_pipeline.py` (`path_b`), `tests/test_model.py` (trace.py exempt, D-21).

## Denoiser inputs (06f.C, 2026-11-21)

The first catalog pass traced only VAEs for most DiT repos (466 `partial` rows were AutoencoderKL graphs) and failed
555 SDXL UNets (`added_cond_kwargs` missing). `trace.denoiser_inputs` holds one row per class: UNet2D* (SDXL
`text_time` ids, class labels), ControlNetModel, FluxControlNetModel, T2IAdapter, Flux / Flux2, SD3, Wan, QwenImage,
LTX-Video; each traces in 3-10 s at < 0.8 GB per worker with every stored parameter bound. Root `config.json`
single-model repos (ControlNet, T2I-Adapter, bare UNet) are path-B candidates. Still "no example inputs":
MiniMax-H3, Z-Image (data-dependent index tensors / tensor lists), LTX-2, CogVideoX, HiDream, Cosmos, Krea2,
video UNets. A registry change re-queues `partial` rows (D-20), so these reach the VAE-only rows.
