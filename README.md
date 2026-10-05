# GPT2PytorchContruct

Pure-PyTorch **DistilGPT2** built to be *looked at*, not to be fast. No `transformers` anywhere in `src/`.

## Purpose

This is not another GPT-2 training project. It is an inspectable reference workload for Tensormorph:

```text
PyTorch → state_dict → tensors → modules → operators → execution → tracing → Safetensors → TensorGraph → Tensormorph
```

Every tensor in the forward pass is a named local variable, attention is spelled out (no `nn.MultiheadAttention`, no SDPA, no FlashAttention), inputs are hard-coded token ids (no tokenizer, no network), and everything runs on CPU in `eval()` mode by default. Nothing here depends on Tensormorph.

## Quick start

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '.[dev]'          # torch, safetensors, pytest
uv pip install --python .venv/bin/python -e '.[reference]'    # optional: transformers, only for reference/
source .venv/bin/activate                                      # the commands below use plain `python` / `pytest`

python -m tools.inspect_checkpoint        # keys, shapes, tying, inferred architecture, tree  (--full: all blocks)
python -m tools.print_model               # module tree + parameter counts + logical/unique accounting
python -m tools.tensor_inventory          # every parameter/buffer -> artifacts/tensor_inventory.{json,csv}
python -m tools.tensor_program            # every layer as tensor operations (all operands tensors) -> artifacts/tensor_program.{md,json,txt}, tensor_multiplications.csv
python -m examples.forward_fixed          # one deterministic forward, all intermediate shapes + logits stats
python -m examples.dataloader_forward     # Dataset -> DataLoader -> batch -> model -> logits
python -m examples.inspect_attention      # Q/K/V, scores, mask, probabilities of one block
python -m examples.forward_hooks          # register_forward_pre_hook / register_forward_hook execution order
python -m examples.execution_trace        # artifacts/execution_trace.json  (--include-values [MAX_NUMEL])
python -m examples.fx_trace               # torch.fx graph -> artifacts/fx_graph.{txt,json}
python -m examples.export_model           # torch.export -> artifacts/export/
python -m examples.causal_lm_loss         # logits, shift_logits, shift_labels, loss
python -m examples.single_train_step      # forward -> loss -> backward -> gradients -> one SGD step
python -m examples.autograd_inspect       # grad_fn / is_leaf / .grad and the backward graph
python -m tools.convert_to_safetensors    # weights/pytorch_model.bin -> artifacts/model.safetensors
python -m tools.inspect_safetensors       # header-only inspection (names, shapes, dtypes, offsets)
python -m reference.compare_huggingface --check-gelu-variants   # optional parity vs transformers
pytest
python -m tools.run_pipeline         # everything above, in order -> artifacts/pipeline_report.md (logs in artifacts/logs/)
```

Next steps with TEL (Tensor Expression Language): [`docs/TEL_NEXT_STEPS.md`](docs/TEL_NEXT_STEPS.md).

Every example/tool takes `--checkpoint` (`.bin` or `.safetensors`) and, where a model is loaded, `--device cpu|cuda|mps`. CPU is the default and what the tests use; the model-loading examples and tools were also run once on Apple `mps` (logits agree with CPU to ~3e-5). `cuda` is untested.

The checkpoints in this repo live in `weights/` (git-ignored), not the project root: `DEFAULT_CHECKPOINT = weights/pytorch_model.bin`. `weights/model.safetensors` is the Hugging Face file and is never overwritten; converted output goes to `artifacts/model.safetensors`.

## Architecture

```text
Token IDs [B,T]
    ↓
Token Embedding ─────────┐
                         +   (+ dropout)
Position Embedding ──────┘
    ↓  [B,T,C]
Transformer Block × 6
    ↓
Final LayerNorm
    ↓
LM Head (weight tied to token embedding)
    ↓
Vocabulary logits [B,T,50257]
```

Verified against the checkpoint: vocab 50257, 1024 positions, hidden C=768, 6 layers, 12 heads (D=64), MLP 3072. The head count is the one value not recoverable from any tensor shape (`infer_config` assumes head_dim 64 and asserts every shape it *can* check); parity with Hugging Face confirms 12.

```text
DistilGPT2LMHeadModel
├── transformer: GPT2Model
│   ├── embeddings: GPT2Embeddings        (TokenEmbedding, PositionEmbedding, Dropout)
│   ├── blocks: ModuleList × 6 of TransformerBlock
│   │   ├── ln_1: LayerNorm
│   │   ├── attention: CausalSelfAttention (qkv_projection, output_projection)
│   │   ├── ln_2: LayerNorm
│   │   └── mlp: GPT2MLP (fc → GPT2GELU → projection)
│   └── final_layer_norm: LayerNorm
└── lm_head: Linear (no bias, same Parameter as the token embedding)
```

Block: `x = x + Attn(LN1(x))`, then `x = x + MLP(LN2(x))` (pre-LayerNorm). MLP uses GPT-2's `gelu_new` (tanh approximation), written out in `GPT2GELU`.

### Attention flow

```text
X [B,T,C]
 ↓ qkv_projection (Linear C→3C)
[B,T,3C]
 ↓ chunk(3)
 ├─ Q [B,H,T,D]   (each: [B,T,C] → [B,T,H,D] → [B,H,T,D])
 ├─ K [B,H,T,D]
 └─ V [B,H,T,D]
 ↓
Q Kᵀ / √D          [B,H,T,T]    "attention_scores"
 ↓ masked_fill(~causal_mask, -inf)
 ↓ softmax(dim=-1)              "attention_probs", each row sums to 1, future = exactly 0
 ↓ dropout (off in eval)
probs @ V          [B,H,T,D]    "context"
 ↓ merge heads: [B,T,H,D] → [B,T,C]
 ↓ output_projection (Linear C→C), dropout
[B,T,C]
```

`attention(x, return_debug=True)` returns `(output, {qkv, query, key, value, attention_scores, causal_mask, attention_probs, context})`. `model(ids, return_debug=True)` returns a `CausalLMOutput` whose `.debug` holds every tensor of the whole model under Tensormorph names (below). A plain `model(ids)` returns just the logits and keeps nothing alive. `model(ids, labels=labels)` adds `.loss`.

The causal mask is a non-persistent boolean buffer (`register_buffer(..., persistent=False)`), so it is not part of `state_dict()`; it still appears in the tensor inventory as a buffer.

## Checkpoint tensors explained

83 tensors in `pytorch_model.bin`. Hugging Face `Conv1D` stores weights as `[in, out]`; `nn.Linear` stores `[out, in]`. The mapping layer (`checkpoint/mapping.py`, the only place that knows HF names) transposes exactly the four projection **weights** per block. Biases, embeddings and LayerNorm parameters are never transposed.

| Checkpoint name | Meaning / where used | Shape (rank) | in → out | Our module | Transposed |
|---|---|---|---|---|---|
| `transformer.wte.weight` | token embedding table; row *i* is the vector for token *i*; also the LM head | `[50257, 768]` (2) | vocab → C | `embeddings.token_embedding` | no |
| `transformer.wpe.weight` | learned absolute position embedding | `[1024, 768]` (2) | position → C | `embeddings.position_embedding` | no |
| `h.N.ln_1.weight/bias` | scale/shift of the LayerNorm before attention | `[768]` (1) | C → C | `blocks.N.ln_1` | no |
| `h.N.attn.bias` | HF's precomputed causal mask buffer | `[1,1,1024,1024]` (4) | — | verified `== tril`, then ignored (we build our own mask) | — |
| `h.N.attn.c_attn.weight` | fused Q,K,V projection | `[768, 2304]` (2) | C → 3C | `attention.qkv_projection` | **yes** → `[2304, 768]` |
| `h.N.attn.c_attn.bias` | its bias, order Q then K then V | `[2304]` (1) | | same | no |
| `h.N.attn.c_proj.weight` | attention output projection | `[768, 768]` (2) | C → C | `attention.output_projection` | **yes** |
| `h.N.attn.c_proj.bias` | its bias | `[768]` (1) | | same | no |
| `h.N.ln_2.weight/bias` | LayerNorm before the MLP | `[768]` (1) | C → C | `blocks.N.ln_2` | no |
| `h.N.mlp.c_fc.weight` | MLP expansion | `[768, 3072]` (2) | C → 4C | `mlp.fc` | **yes** → `[3072, 768]` |
| `h.N.mlp.c_fc.bias` | its bias | `[3072]` (1) | | same | no |
| `h.N.mlp.c_proj.weight` | MLP contraction | `[3072, 768]` (2) | 4C → C | `mlp.projection` | **yes** → `[768, 3072]` |
| `h.N.mlp.c_proj.bias` | its bias | `[768]` (1) | | same | no |
| `transformer.ln_f.weight/bias` | final LayerNorm | `[768]` (1) | C → C | `final_layer_norm` | no |
| `lm_head.weight` | output projection, **same storage as `wte`** | `[50257, 768]` (2) | C → vocab | `lm_head` (tied) | no |

(`h.N` = `transformer.h.N`, N = 0…5.) Loader report for the real checkpoint:

```text
loaded 76 (of which transposed 24), tied 1, ignored 6, missing 0, unexpected 0
```

Any unmapped checkpoint key, missing model key, wrong shape, an `attn.bias` that is not the causal mask, or an `lm_head` that differs from `wte` raises `CheckpointError` before anything is copied.

## Everything as tensor operations

`python -m tools.tensor_program` lowers every `nn.Module` into a flat list of tensor operations where **every operand is a tensor**: `y = x·W + b` is the tensors `x`, `W`, `b` and the operations `x·W`, `+ b`; LayerNorm, softmax and GELU are spelled out the same way, constants such as `eps` and `0.044715` are rank-0 tensors, and axes are named (`B,T,D,H,Dh,...`) so no transposes are needed. For DistilGPT2: 475 tensors, 385 operations, **112 multiplications** (37 contractions + 75 elementwise products); the 37 contractions equal what TEL's importer emits for the same checkpoint.

The program is executed straight from the checkpoint's original Hugging Face tensors (Conv1D `[in,out]`, no transposition) and 93 intermediates are compared with the pure model's `return_debug` tensors on all five samples (worst error 4.7e-6 of each tensor's max magnitude). Output: `artifacts/tensor_program.md` (report with the full multiplication list), `.json` (machine-readable), `.txt` (all 385 operations), `tensor_multiplications.csv`. Implementation: `src/gpt2_pytorch/tensorprog/` (one lowering rule per module type; an unknown module type raises).

## Parameter accounting

| Category | Logical references | Unique parameters |
|---|---:|---:|
| token embeddings | 38,597,376 | 38,597,376 |
| position embeddings | 786,432 | 786,432 |
| attention (6 blocks) | 14,174,208 | 14,174,208 |
| MLP (6 blocks) | 28,334,592 | 28,334,592 |
| LayerNorm (13) | 19,968 | 19,968 |
| LM head | 38,597,376 | 0 (tied) |
| **total** | **120,509,952** | **81,912,576** |

The 6 × 1024² `attn.bias` buffers add 6,291,456 elements; (81,912,576 + 6,291,456) × 4 bytes = 352,816,128 B = 336.47 MiB, the checkpoint size, counted once per storage.

## Weight tying and Safetensors

`self.lm_head.weight = self.transformer.token_embedding.weight`: one `Parameter`, one storage, one gradient, listed under two names (`named_parameters(remove_duplicate=False)` shows both; `parameters()` once).

Safetensors cannot store aliased storage, so `tools.convert_to_safetensors` **omits `lm_head.weight`** (recorded in the file metadata under `tied`) and keeps the six `attn.bias` buffers. The result has the same 82 keys and bit-identical values as Hugging Face's own `weights/model.safetensors` (tested). Loading re-ties the head by construction; the loader reports it as `lm_head.weight -> transformer.embeddings.token_embedding.weight (reconstructed)`. `.bin` and Safetensors logits are `torch.equal`.

`tools.inspect_safetensors` parses only the JSON header (8-byte length prefix + JSON) and cross-checks `shape × dtype size == end − begin` for each tensor.

## Tracing and Tensormorph naming

Parameters (persistent state) and activations (runtime tensors) are separate everywhere.

| What | Name pattern |
|---|---|
| input | `input.input_ids` |
| embeddings | `embedding.token.output`, `embedding.position.output`, `embedding.output` |
| block | `block.N.input`, `block.N.ln_1.output`, `block.N.attention_residual.output`, `block.N.ln_2.output`, `block.N.output` |
| attention | `block.N.attention.{qkv,q,k,v,scores,mask,probs,context,output}` |
| MLP | `block.N.mlp.fc.output`, `block.N.mlp.activation.output`, `block.N.mlp.output` |
| head | `model.final_hidden`, `model.logits` |
| loss (with labels) | `loss.shift_logits`, `loss.shift_labels`, `loss.value` |

Three levels:

1. **Hooks** (`ActivationRecorder`, `examples.forward_hooks`): per module call, in order, with name, type, input/output tensors, and shape, dtype, device, min, max, mean, std, numel, bytes, strides, storage offset. No model edits needed.
2. **Internal attention tensors**: hooks cannot see locals, so `CausalSelfAttention` has `return_debug=True` and a `debug_sink` the recorder installs for you. Stats only unless `keep_tensors=True`.
3. **Graphs**: FX and `torch.export` (below).

`artifacts/execution_trace.json` (`format: tensormorph-pytorch-trace`, `version: 1`):

```json
{"format": "tensormorph-pytorch-trace", "version": 1, "model": {...},
 "inputs":     [{"name": "input.input_ids", "shape": [2, 8], "dtype": "int64", "device": "cpu", "numel": 16}],
 "parameters": [{"name": "lm_head.weight", "kind": "parameter", "shape": [50257, 768], "stride": [768, 1],
                 "storage_id": "storage:0", "tied_with": ["transformer.embeddings.token_embedding.weight"], "...": "..."}],
 "buffers":    [{"name": "transformer.blocks.0.attention.causal_mask", "kind": "buffer", "...": "..."}],
 "events":     [{"index": 9, "exit_index": 5, "depth": 4, "module": "transformer.blocks.0.attention.qkv_projection",
                 "module_type": "Linear", "input_shapes": [[2, 8, 768]], "output_shapes": [[2, 8, 2304]],
                 "dtype": "float32", "device": "cpu",
                 "inputs": [...], "outputs": [...], "parameters": ["...qkv_projection.weight", "...qkv_projection.bias"],
                 "internals": []}],
 "outputs":    [{"name": "model.final_hidden", "...": "..."}, {"name": "model.logits", "...": "..."}]}
```

`index` is call order (pre-hook), `exit_index` completion order. Attention events carry their q/k/v/scores/mask/probs/context in `internals`. Tied parameters share a `storage_id` (raw addresses are not written; they are not stable). Values are omitted unless `--include-values [MAX_NUMEL]` (only tensors up to that size).

`ActivationRecorder`, `storage_info` and the inventory report `shape, stride, storage_offset, is_contiguous, is_view, dtype, device, requires_grad, numel, element_size, bytes` and shared-storage groups.

### FX and `torch.export`

* `torch.fx.symbolic_trace(model, concrete_args={"labels": None, "position_ids": None, "return_debug": False})` traces the **unmodified** model into 335 nodes (Linear/LayerNorm/Embedding/Dropout stay `call_module` leaves; attention maths and GELU are inlined). Output equals eager bit-for-bit, and the graph works for other sequence lengths. Three constructs had to be written FX-friendly in the reference code without changing the maths: `x.size(i)` instead of unpacking `x.shape`, `-inf` instead of `torch.finfo(dtype).min` for the mask fill, and `torch.narrow` instead of slicing a buffer with a symbolic length. `concrete_args` freezes the optional arguments so the dataclass-returning debug path is never traced. `print_tabular` needs the third-party `tabulate`, so the example prints its own table (same columns plus shape) to `artifacts/fx_graph.txt` and `.json`.
* `torch.export.export(model, (FIXED_INPUT_IDS,))` works on the unmodified model: 346 graph nodes, graph inputs 77 parameters (the tied head is listed under both names), 7 buffers, 1 user input. Saved as `artifacts/export/exported_program.txt` and `graph.json` (nodes with shapes/dtypes plus the input signature). The `.pt2` is not written by default (it embeds all 336 MiB of weights).

## Tests

`pytest` (108 tests, ~10 s, offline) covers: checkpoint inspection and config inference, the key mapping (every category, transposition, and each failure mode), embeddings, attention (shapes, QKV order, SDPA cross-check *in tests only*, causality incl. "changing future tokens cannot change past logits"), MLP/GELU variant, residual structure, tied weights and gradients, the fixed DataLoader, golden logits, `.bin` ↔ Safetensors equality, the recorder, the trace format, and FX/export on a tiny seeded model. Tests that need the real weights skip if `weights/pytorch_model.bin` is absent; the Hugging Face parity test skips without `transformers`; one test asserts nothing under `src/` imports `transformers`.

## Hugging Face parity

`python -m reference.compare_huggingface --check-gelu-variants` builds the HF model from a locally constructed config (no network), loads the same `.bin`, and compares logits on the same ids in `eval()`:

```text
tanh GELU (ours)   max_abs=7.629e-05  mean_abs=9.686e-06  max_rel=1.067e-06  allclose(rtol=1e-4, atol=1e-3)=True
erf GELU (wrong)   max_abs=7.070e-02  mean_abs=1.966e-02  max_rel=1.388e-03  allclose=False
```

The ~1e-5 residual is float32 summation order (`addmm` vs `Linear`); the erf variant being ~1000× worse is what pins down `gelu_new` as the checkpoint's activation (also stated in the HF `config.json`: `activation_function: gelu_new`).

## Layout

```text
src/gpt2_pytorch/{config,outputs}.py
src/gpt2_pytorch/model/        embeddings attention mlp block backbone causal_lm
src/gpt2_pytorch/checkpoint/   inspect infer_config mapping pytorch_loader safetensors_loader loader
src/gpt2_pytorch/data/         fixed_samples dataset
src/gpt2_pytorch/tracing/      tensor_stats hooks activation_recorder execution_trace
src/gpt2_pytorch/utils/        model_tree param_count tensor_inventory seed cli
examples/ tools/ reference/ tests/ artifacts/
```

## Limitations

* No tokenizer (raw ids only; `tokenizer/` is intentionally not included). The fixed ids are valid GPT-2 ids for the sentences in `fixed_samples.py`, not produced by running a tokenizer here.
* Attention is the explicit O(T²) path only; there is no optimized SDPA variant.
* `cuda` is untested; `mps` was run once by hand (all model-loading commands), the pytest suite runs on CPU only.
* Number of heads is assumed from head_dim 64 (see above).
