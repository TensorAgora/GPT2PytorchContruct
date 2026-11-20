# Graph ⇔ TEL: design and proof

How a `tensormorph-viz-graph` v1 graph written by the hf-graph pipeline becomes Tensor Expression Language (TEL), and
the round-trip check that holds it to that. Design only, plus the proof of
`prompts/hf-weights-to-graph-json-pipeline.md` §9.6; the batch converter (`model.tel` plus `bindings.json` per repo) is
the next stage.

Checked against TensormorphDesignUI `8755b2c` (2026-11-20) and tensorexpressionlanguage `553b7bae` (2026-10-23).
Paths: `OPS` = `tensorexpressionlanguage/docs/operators`, `ADR` = `…/docs/adr`, `PKG` = `…/packages`.

## Where TEL sits

The graph is already at TEL's primitive level. Every viz-graph `primitive` (Contraction, Elementwise, Reduce, Gather,
Slice, Reshape, Permute, Rename) is one of TEL's TensorIR primitives (`OPS/61-tensorir-primitives.md` §2), so
graph → TEL is printing, not lowering. The target is **Quick TEL source plus out-of-band bindings** (T-3), the shape the
shipped importer returns (`PKG/model-import/src/program/emit.ts:190-201`: `{source, bindings: [{name, uri, declared}],
viewBindings, layerSpans, dialect: 'quick'}`). Quick TEL is the canonical saved source (ADR-0025, Accepted); emitting
TTIR (ADR-0009) directly is a later option.

The printed form is **T♭**: one statement per op, every tensor declared (TensorNodes `nodes/tel.js` `printFlat`), whose
`build()`/`decodeExpr()` rebuild the graph by symbol. Fields TEL never spells (`module`, `module_type`, `region`,
`equation`, `checkpoint_key`, `view`, `pytorch_name`) travel in Σ, the symbol table, joined by output symbol; `edges`,
`groups`, `multiplications` and `totals` are re-derived.

## Mapping

| viz-graph op | in the 8 goldens | TEL | catalog lowering | note |
| --- | --: | --- | --- | --- |
| Contraction `over` | 6,719 | `@`, or `contract(a, b, over=X)` | `P:Contraction` | `contract` when the right operand has a batch axis the left lacks (fix 1) |
| Elementwise add, sub, mul, div | 9,865 | infix | `P:Elementwise` (div `D:`) | |
| Elementwise exp, log, tanh | 696 | builtins | `P:Elementwise` | |
| Elementwise rsqrt, sigmoid | 1,419 | builtins | `P:` / `D:` | missing in `tools/viz_to_tel.py` |
| Reduce mean, sum, max | 1,575 | `mean`, `sum`, `max` | `D:` / `P:Reduce` | missing in `tools/viz_to_tel.py` |
| Rename | 1,434 | `relabel` | `P:Rename` | |
| Reshape, Permute | 2,222 | `reshape`, `permute` | `D:` / `P:Permute` | |
| Slice `to_axis` | 560 | `slice` + `relabel` | `P:Slice` | |
| Gather | 9 | `gather` / embedding | `P:Gather` | |
| region `softmax`, `layernorm` | 246 | composite call | expands in the compiler | |
| RMSNorm, SwiGLU, attention | — | primitives | the compiler emits `tm.math` with no expansion (`PKG/tel/src/ir/lower.ts:553-565`) | |

**Rule T-1.** A composite prints as a call only when the compiler expands it (`softmax`, `layernorm`, the compiler's
spellings per DR-43, not the catalog's `layer_norm`/`rms_norm`); every other composite prints as its primitives.

## Bindings

- **URI (ADR-0027, Accepted).** `tensor:` is the only scheme (U1, lines 37-41); `hf:`, `safetensors:` and `gguf:` are
  rejected. The authority is "the store — a checkpoint file or a model name — with an optional `@revision`" (table,
  lines 56-60), escaped only where `/ @ ? # %` would read as structure; the path is the store's own name for the tensor,
  opaque to the grammar (U3, lines 71-83).
- **Store name for an HF repo.** The model name at its pinned revision, with `/` escaped:
  `tensor://Qwen%2FQwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28/model.layers.0.self_attn.q_proj.weight`.
  The revision is the manifest's `revision_sha`; the path is the manifest's `tensors[].key` verbatim (llama.cpp names for
  GGUF).
- **Sharded repos are one store.** `bindStorage(requests, store)` binds against one `Store` with one header
  (`PKG/tel/src/storage/bind.ts:42-47`) and looks each path up in `store.header.entries` (`:94`). A sharded repo
  therefore resolves to a single store whose header is the union of its shard headers (the manifest's `tensors[]`,
  each entry keeping its `file`); a per-file store (`tensor://model-00001-of-00004.safetensors/…`) is legal but ties
  identity to the storage layout, which U1 argues against.
- **dtype.** The binder compares the declared dtype with the header's (`bind.ts:125`, `STORAGE_DTYPE_MISMATCH`), so a
  declaration carries the manifest dtype (BF16, F8_E4M3, …), not today's blanket `float32`.
- **Quantized storage** (FP8 block scales, GGUF ggml types) has no TEL dtype or `QuantScheme` record yet
  (`OPS/41-quantization.md`): those parameters are counted exclusions.
- **Keyless parameters** (`view` without `checkpoint_key`, e.g. DeepSeek's per-expert stacks) need TEL families or
  `BindRequest.substitutions` (ADR-0029, Proposed): counted exclusions until then.
- **Model paths** (`model m = "tensor://…"`, `m.layers[3]…`) are ADR-0029 (Proposed) and give TM1107 today: not emitted.

## Exclusions

The spellable subset is printed; everything else is a marked read-only line that is counted, never passed silently.
`tools/tel_coverage.py` counts these per graph.

| construct | why | source |
| --- | --- | --- |
| routed MoE (`Sel_*` runtime buffers, top-k) | catalog names exist, nothing is built | decoder_moe; D8 `execution.kind = routed` |
| unrolled or scanned recurrence | no `Scan` surface; graphs unroll at T = 8 | decoder_hybrid; D8 `execution.kind = scan` |
| tensor-valued constants (masks, rope `cos`/`sin`/`R`, `conv_shift`) | no array literal, no `-inf` literal | every non-GPT-2 golden; bound as buffers instead (fix 2) |
| concrete `B`, `T` with no symbol | a symbolic program cannot come from a fixed-T graph | arch.mjs goldens; pipeline graphs keep `axes[].symbol` (R-52) |
| ops with no TEL spelling | `tm.extern` (ADR-0030) is Proposed | path-B `opaque` ops |

## Round trip

The claim: **graph ⇔ T♭ + Σ is 1:1 on the spellable subset, provided tensor constants are bound and MoE contractions
use `contract`.** It is not a claim about arbitrary TEL: hand-written or inlined TEL does not decode back, `decodeExpr`
reads `contract(…, over=…)` as one axis (all 8 goldens contract one axis; path-B convolutions may not), and there is no
TTIR → graph converter.

`node tools/tel_roundtrip.cjs <graph|dir>… [--const-as-buffer] [--fix-contract]` prints `printFlat` → `build`
differences by kind, the compile result and D3 (TEL's `tm.core` counts per primitive equal the graph's), one JSON line
per graph and a summary line. The two fixes are flags of that script; `nodes/tel.js` is not modified (T-2: the Python
port, `gpt2_pytorch/tensorprog/tel_flat.py`, carries them natively).

1. `--fix-contract`: a Contraction printed with `@` whose right operand has a batch axis (all but its last two) that is
   not a batch axis of the left is respelled `contract(a, b, over=X)` (`Hg = N2[B,T,D] @ Wge[E,Fe,D].T`,
   `Y = Hh[B,T,E,Fe] @ Wde[E,D,Fe].T`: 96 in mixtral, 174 in deepseek-v3). `x[B,T,D] @ W[F,D].T` keeps `@`.
2. `--const-as-buffer`: tensor-valued constants become declared buffers, as if bound through a `tensor://` sidecar store.

Expected on the 8 goldens (§9.5): with `--const-as-buffer`, 0 differences everywhere and 6 of 8 pass D3 (mixtral and
deepseek-v3 fail TM2205); adding `--fix-contract`, 8 of 8 have 0 differences, compile clean and pass D3. The baseline
`tools/viz_to_tel.py` → `tools/tel_compile.mjs` on distilgpt2 still prints PASS. Results are recorded in this file's
last section by the proof run.

## Measured

2026-11-21, `node tools/tel_roundtrip.cjs json/*.json …` (09a proof):

| golden | ops | list constants | `--const-as-buffer`: differences, compile, D3 | `--fix-contract` respelled | both flags: differences, compile, D3 |
| --- | --: | --: | --- | --: | --- |
| deepseek-v3 | 5,453 | 6 | 0, TM2205 ×2, fail | 174 | 0, clean, pass |
| distilgpt2 | 385 | 0 | 0, clean, pass | 0 | 0, clean, pass |
| gemma-2-2b-it | 1,464 | 6 | 0, clean, pass | 0 | 0, clean, pass |
| mistral-7b-instruct-v0.3 | 1,800 | 6 | 0, clean, pass | 0 | 0, clean, pass |
| mixtral-8x7b-instruct-q4km | 2,024 | 6 | 0, TM2205 ×2, fail | 96 | 0, clean, pass |
| phi-3.5-mini-instruct | 1,800 | 6 | 0, clean, pass | 0 | 0, clean, pass |
| qwen2.5-7b-instruct | 1,660 | 6 | 0, clean, pass | 0 | 0, clean, pass |
| qwen3.8-27b-fp8 | 9,913 | 23 | 0, clean, pass | 0 | 0, clean, pass |

Summary: `--const-as-buffer` 8/8 zero differences, 6/8 D3; with `--fix-contract` 8/8 zero differences, 8/8 compile,
8/8 D3, matching §9.5. `tools/viz_to_tel.py` → `tools/tel_compile.mjs` on distilgpt2: PASS.

`python -m tools.tel_coverage json/*.json` (counted exclusions, without binding constants):

| golden | ops | spellable | routed MoE | recurrence | ops on list constants | concrete B/T |
| --- | --: | --: | --: | --: | --: | --: |
| deepseek-v3 | 5,453 | 3,808 | 1,218 | 0 | 427 | 2 |
| distilgpt2 | 385 | 385 | 0 | 0 | 0 | 0 |
| gemma-2-2b-it | 1,464 | 1,282 | 0 | 0 | 182 | 2 |
| mistral-7b-instruct-v0.3 | 1,800 | 1,576 | 0 | 0 | 224 | 2 |
| mixtral-8x7b-instruct-q4km | 2,024 | 1,384 | 416 | 0 | 224 | 2 |
| phi-3.5-mini-instruct | 1,800 | 1,576 | 0 | 0 | 224 | 2 |
| qwen2.5-7b-instruct | 1,660 | 1,464 | 0 | 0 | 196 | 2 |
| qwen3.8-27b-fp8 | 9,913 | 3,273 | 0 | 6,480 | 2,416 | 0 |

The list-constant counts equal tel_roundtrip's `list_constants` for every golden.
