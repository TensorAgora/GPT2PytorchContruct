# Build a TypeScript Model Importer: Any Weight / Model Format → Tensormorph Metadata + Compute Graph + Ordered Matrix-Multiplication Graph

You are a senior **model-format engineer, TypeScript systems engineer, ML compiler engineer, and graph-visualization tooling engineer**.

Your task is to build a **TypeScript library and CLI** that opens a model file (or a bundle of files) in any of the popular weight / model formats, and converts it into **one internal Tensormorph representation**:

1. a **tensor inventory** (every stored tensor: name, dtype, shape, byte location, quantization, aliasing),
2. a **compute graph** in canonical operations, in execution order (when a graph can be obtained),
3. a **matrix-multiplication graph**: every tensor multiplication (contraction) of the forward pass, in a well-defined sequence order, with the weights it reads, its contraction dimensions, its FLOPs, and its data dependencies on earlier multiplications,

so that a viewer can **visualize every tensor multiplication of a model in a suitable sequence order**.

The project must favor **correctness, explicit diagnostics, deterministic output, and inspectability** over coverage claims. A format that cannot be read faithfully must say so; it must never produce a guessed graph.

---

# 0. How to Read This Prompt

* Sections 1–3 explain the goal and the one fact that shapes everything (most weight files contain **no graph**).
* Sections 4–7 are the **contract**: inputs, outputs, the pipeline, and the exact ordering and dependency rules.
* Section 8 is the **per-format requirements**. Section 9 maps every format's ops to the canonical op set.
* Sections 10–13 cover architecture templates, the matmul extraction algorithm, visualization, and security.
* Sections 14–19 cover performance, layout, dependencies, testing, phases, and the Definition of Done.
* Section 20 lists open decisions. Record your choice in `DECISIONS.md`; do not block on them.

Wherever this prompt says **"source of truth: …"**, verify the detail there (and with a fixture file) before writing code. Do not implement format details from memory.

---

# 1. Context: What Already Exists

Project root (Python reference project, the numerical and structural oracle):

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct
```

Relevant pieces:

| Path | What it is | Why it matters here |
|---|---|---|
| `weights/pytorch_model.bin` | Hugging Face DistilGPT2, PyTorch zip pickle, 83 tensors, 336 MiB | golden input #1 |
| `weights/model.safetensors` | Hugging Face DistilGPT2, Safetensors, 82 tensors (`lm_head.weight` omitted: tied) | golden input #2 |
| `src/gpt2_pytorch/tracing/matmul_graph.py` | **state_dict + PyTorch model code → matmul graph** via `torch.export(...).run_decompositions()` | defines the output schema `tensormorph-matmul-graph` v1 and the dependency rules this importer must reproduce |
| `tools/matmul_graph.py` | CLI: `python -m tools.matmul_graph [--model pkg.mod:build] [--checkpoint PATH]` → `artifacts/matmul_graph.{json,md}` | the bridge for "trained PyTorch weights + my own model code" (Section 3.3) |
| `artifacts/matmul_graph.json` | golden output for DistilGPT2, B=2, T=8 | acceptance test (Section 17) |
| `artifacts/tensor_multiplications.csv` | 112 multiplications (37 contractions + 75 elementwise products) from the tensor program | golden for the optional elementwise mode |
| `src/gpt2_pytorch/checkpoint/mapping.py` | HF checkpoint names ↔ this repo's module names, Conv1D transposition rule | translates golden names to HF names |
| `docs/TEL_NEXT_STEPS.md` | TEL (Tensor Expression Language) analysis; TEL's TypeScript `tensorparser` already reads a GPT-2 Safetensors header (82 tensors, 0 payload bytes read) | possible home for this code (Open Decision D1) |
| `netron/` | vendored **Netron 9.3.2** (`lutzroeder/netron`, MIT license), untracked in git | reference parsers for every format in this prompt |

Netron is a model **viewer**. It parses all these formats, but it does not compute matmul dependency graphs, contraction dimensions, or FLOPs, and its graph model is built for display. Use `netron/source/<format>.js` as the **source of truth for file layout**. You may port parsing code with attribution, keeping the MIT notice. **Do not port `netron/source/python.js`** (22,799 lines, an emulated Python interpreter for TorchScript and arbitrary pickles). `netron/source/pickle.js` (437 lines) is the right size of reference for the restricted unpickler you need.

---

# 2. Goal

```text
model file(s) ──► detect format (magic bytes) ──► parse container ──► tensor inventory
                                                        │
                       ┌────────────────────────────────┼─────────────────────────────────┐
                       ▼                                ▼                                 ▼
              embedded graph                  architecture template              external graph JSON
       (ONNX, TFLite, Core ML, Keras      (GGUF general.architecture,       (tools/matmul_graph.py output,
        config, TF .meta/SavedModel,        HF config.json model_type,        i.e. PyTorch code + weights)
        torch.export .pt2)                  user --architecture)
                       └────────────────────────────────┼─────────────────────────────────┘
                                                        ▼
                         canonical compute graph (topologically ordered, shapes inferred)
                                                        ▼
                         matmul graph (nodes = contractions, edges = data dependencies)
                                                        ▼
                         JSON (+ Mermaid, DOT, CSV) ──► viewer: every multiplication, in order
```

User-visible outcome: for any supported input, one command

```bash
tm-import model.onnx --dims batch=1,sequence=8 --out out/
```

writes `out/model.tensormorph.json` and, on request, `matmul_graph.mmd`, `matmul_graph.dot` and `tensor_multiplications.csv`. Together they list every matrix multiplication in execution order, with its weight, its dimensions, its cost, and what it depends on.

---

# 3. The Fact That Shapes Everything: Weights Are Not a Graph

A state_dict / Safetensors / GGUF file is a dictionary `name → tensor`. It does not record **which operations run, in which order, on which tensors**. The order of multiplications lives in the **model code** or in a **graph-carrying format**. The importer must therefore classify every input by where its graph comes from.

## 3.1 Format classification

| Format | Typical extensions | Detection (verify in netron source) | Weights | Graph | Order comes from |
|---|---|---|---|---|---|
| Safetensors | `.safetensors`, `*.safetensors.index.json` | u64 LE header length, then JSON `{` (`netron/source/safetensors.js`) | yes | **no** | template / external |
| GGUF | `.gguf` | ASCII `GGUF` + u32 version (`netron/source/gguf.js`) | yes (often quantized) | **no** (only `general.architecture` + hyperparameters) | template |
| PyTorch state_dict / training checkpoint | `.pt`, `.pth`, `.bin`, `.ckpt` (Lightning) | zip `PK\x03\x04` containing `*/data.pkl`; or legacy pickle (`netron/source/pytorch.js`, `pickle.js`) | yes | **no** | template / external |
| PyTorch TorchScript | `.pt`, `.pth` | zip with `code/` and `constants.pkl` (`pytorch.js`) | yes | yes, but as TorchScript source code | out of scope v1 → re-export (Section 8.4) |
| torch.export | `.pt2` | zip with serialized ExportedProgram (`pytorch.js`, `pytorch-schema.js`) | yes | **yes** | file |
| ONNX | `.onnx` (+ external data files) | protobuf `ModelProto` (`onnx.js`, `onnx-proto.js`) | yes | **yes** | file (spec requires topological order; still validate) |
| TensorFlow checkpoint | `<prefix>.index` + `<prefix>.data-NNNNN-of-NNNNN` (+ TF1 `<prefix>.meta`) | SSTable table + BundleEntryProto (`tf.js`, `tf-proto.js`) | yes | only with `.meta` (TF1 MetaGraphDef) or a SavedModel `saved_model.pb` | `.meta` / SavedModel, else template |
| HDF5 / Keras 2 | `.h5`, `.hdf5`, `.keras` (Keras 3 = zip + `config.json` + `model.weights.h5`) | HDF5 signature `\x89HDF\r\n\x1a\n` (`hdf5.js`, `keras.js`) | yes | only if `model_config` / `config.json` is present (full-model save); `save_weights` files have none | Keras config (layer order + inbound nodes) |
| TensorFlow Lite | `.tflite` | FlatBuffer, file identifier `TFL3` (`tflite.js`, `tflite-schema.js`) | yes | **yes** | operator order in subgraph |
| TensorRT | `.engine`, `.plan`, `.trt` | `tensorrt.js` (Netron itself reports "undocumented TensorRT engine data") | opaque | opaque, version/GPU-specific | best-effort only (Section 8.10) |
| Core ML | `.mlmodel`, `.mlpackage/` | protobuf `Model` (`coreml.js`, `coreml-proto.js`) | yes (`mlProgram`: in package blob file) | **yes** (`neuralNetwork` layers or `mlProgram` MIL ops) | file |
| "Binary" | `.bin` | **ambiguous**, sniff contents and companions (Section 8.11) | depends | depends | depends |

TensorFlow Lite appears twice in the original requirement list. It is one format.

## 3.2 "No graph available" is a normal result

If the input carries only weights and no template matches, the importer **succeeds**. It emits the tensor inventory, `graph: null`, `matmuls: null`, and a diagnostic of severity `info` explaining which companion would unlock the graph (`config.json`, `--architecture`, an ONNX export, or `tools/matmul_graph.py` JSON). It must **never infer a graph from tensor names alone** without a declared template.

## 3.3 Trained PyTorch weights + the user's own model code

TypeScript cannot run Python model code. For "I trained my own `nn.Module` and uploaded the `.pt`", the supported path is:

```bash
# Python side (this repo): the model code supplies the order
python -m tools.matmul_graph --model my_pkg.models:build --checkpoint trained.pt --out-dir out/
# TypeScript side: import the graph, bind it to the weight file for dtype / byte offsets / inventory
tm-import trained.pt --graph out/matmul_graph.json --out out/
```

An alternative is `torch.onnx.export` / `torch.export.save`, then import the `.onnx` / `.pt2`. The importer must document both paths in its CLI help and in the `info` diagnostic of Section 3.2.

---

# 4. Scope

## In scope (v1)

* Readers for: Safetensors (single + sharded index), PyTorch zip pickle and legacy pickle (state_dict, nested training checkpoints, Lightning `.ckpt`), GGUF, ONNX (+ external data), TFLite, Keras HDF5 / `.keras`, TF checkpoint (+ `.meta` / SavedModel `.pb` when present), Core ML `.mlmodel` / `.mlpackage`, torch.export `.pt2`, the external `tensormorph-matmul-graph` JSON (v1 and v2).
* Detection by content, multi-file bundles, companion files (`config.json`, `*.index.json`, `.meta`, `model.json`).
* Architecture templates: `gpt2` (covers DistilGPT2), `llama`. The template contract (Section 10) must make adding one a single file.
* Canonical compute graph, shape inference for the ops on any path to a matmul, topological ordering, matmul extraction, FLOPs.
* Emitters: JSON (primary), Mermaid, DOT, CSV.
* CLI + library API, Node ≥ 20 and modern browsers.

## Best-effort (v1, clearly labelled)

* TensorRT engines: detection + metadata + optional companion layer-info JSON (Section 8.10).
* `.bin` companions for OpenVINO IR (`.xml` + `.bin`) and TF.js (`model.json` + shards).

## Out of scope (v1)

* Executing models or dequantizing weights for computation (no numerics; Python is the numeric oracle).
* TorchScript code parsing, arbitrary pickles (full `torch.save(model)`), control-flow sub-graphs (`If`/`Loop`/`WHILE`/`cond`/`scan`) beyond reporting them.
* Training graphs / backward pass.
* The visual viewer itself. This project delivers the data contract and text/graph exports the viewer consumes.

---

# 5. Inputs

## 5.1 Library API

```ts
export interface ByteSource {
  readonly name: string;          // file name or path, used in diagnostics
  readonly size: number;          // bytes; files > 2^32 must work (use number up to 2^53, never 32-bit math)
  read(offset: number, length: number): Promise<Uint8Array>;  // random access; never read the whole file implicitly
}

export interface FileBundle {
  primary: ByteSource;                              // the file the user pointed at
  resolve(relativePath: string): Promise<ByteSource | null>;  // siblings: external data, shards, config.json, .meta
}

export interface ImportOptions {
  dims?: Record<string, number>;       // bind symbolic dims, e.g. { batch: 2, sequence: 8 }
  architecture?: string;               // force a template, e.g. "gpt2"
  config?: Record<string, unknown>;    // hyperparameters when no config.json exists
  graph?: ByteSource;                  // external tensormorph-matmul-graph JSON (Section 3.3)
  checkpointKey?: string;              // which sub-dict of a training checkpoint holds the weights
  includeElementwise?: boolean;        // also list elementwise multiplications (default false)
  fuseBias?: boolean;                  // MatMul + Add(bias vector) → one node with a bias operand (default true)
  readPayload?: boolean;               // default false: header/metadata only (Section 14)
}

export function importModel(bundle: FileBundle, options?: ImportOptions): Promise<ModelPackage>;
```

Provide two `ByteSource` implementations: Node (`fs.promises.open` + positional `read`) and browser (`File`/`Blob.slice().arrayBuffer()`). Directory inputs (`.mlpackage/`, SavedModel dirs, sharded checkpoints) are bundles whose `resolve` walks the directory.

## 5.2 CLI

```text
tm-import <path> [--dims k=v,...] [--architecture NAME] [--config config.json] [--graph matmul_graph.json]
                 [--checkpoint-key KEY] [--elementwise] [--no-fuse-bias]
                 [--emit json,mermaid,dot,csv] [--out DIR] [--strict]
tm-import <path> --inspect         # detection + inventory summary only, no graph work
```

`--strict` turns `warning` diagnostics into a non-zero exit. Exit codes: `0` success (graph may be null), `1` unreadable / corrupt / unsafe input, `2` strict-mode warnings.

## 5.3 Example dims

Shapes in graph formats are often symbolic (ONNX `dim_param`, `-1` batch in TFLite/Keras). Defaults (Open Decision D4): every unbound symbolic dim = 1, except a dim whose name looks like a sequence length (`sequence`, `seq`, `tokens`, `T`) = 8. The Python golden uses B=2, T=8, so golden tests pass `--dims batch=2,sequence=8` explicitly. Always record the bindings used in the output (`inputs[].shape` and `dims`). If a dim stays unbound, keep it symbolic in shapes, set `flops: null` and fill `flops_expr` (Section 6.4).

---

# 6. Output: `ModelPackage` (format `tensormorph-model`, version 1)

One JSON document per import. Keys are **snake_case** on disk (matching the Python schema); TypeScript interfaces mirror them. Output must be **byte-for-byte deterministic** for the same input and options: stable key order, stable node order, no timestamps unless `--timestamp`.

```ts
interface ModelPackage {
  format: "tensormorph-model";
  version: 1;
  source: {
    files: { path: string; size: number; role: "primary" | "external_data" | "shard" | "config" | "graph" | "companion" }[];
    detected_format: FormatId;            // "safetensors" | "gguf" | "pytorch_zip" | "pytorch_legacy" | "onnx" | ...
    format_version?: string;              // e.g. GGUF v3, ONNX ir_version + opsets, TFLite schema version
    producer?: string;                    // producer_name / converter / framework version when recorded
    metadata: Record<string, unknown>;    // raw file metadata (safetensors __metadata__, GGUF kv, ONNX metadata_props)
  };
  architecture: {
    name: string | null;                  // "gpt2", "llama", or null
    origin: "file_graph" | "template" | "external_graph" | "none";
    config: Record<string, unknown>;      // resolved hyperparameters (n_layer, hidden, heads, ...)
  };
  dims: Record<string, number | string>;  // symbolic dim bindings actually used
  tensors: TensorInfo[];                  // the inventory, always present
  tied: string[][];                       // groups of tensor names sharing one storage
  graph: ComputeGraph | null;
  matmuls: MatmulGraph | null;            // tensormorph-matmul-graph v2, Section 6.4
  diagnostics: Diagnostic[];
  stats: { bytes_read: number; header_bytes: number; payload_bytes_read: number; tensors: number; parse_ms: number };
}

interface TensorInfo {
  name: string;                           // original key, exactly as stored (never rewritten)
  canonical_name?: string;                // template / mapping name, e.g. HF key for a GGUF tensor
  role: "parameter" | "buffer" | "constant" | "input" | "output";
  dtype: DType;                           // "f32" | "f16" | "bf16" | "f64" | "i64" | "i32" | "i16" | "i8" | "u8" | "bool" | "f8_e4m3" | "f8_e5m2" | quantized ids
  shape: (number | string)[];             // row-major, outermost first, ALWAYS normalized to this convention
  source_shape?: number[];                // as written in the file when it differs (GGUF dim order, TFLite OHWI, ...)
  layout_note?: string;                   // e.g. "Conv1D [in,out]", "TFLite filter OHWI", "Keras kernel [in,out]"
  storage?: { file: string; offset: number; length: number; encoding: "raw" | "quantized" | "packed" };
  quant?: { scheme: string; block_size?: number; scale?: string; zero_point?: string; axis?: number };  // names of the scale/zp tensors, or GGUF type
  strides?: number[];                     // PyTorch views: non-contiguous storage
  storage_key?: string;                   // PyTorch storage id / TFLite buffer index / ONNX initializer name: aliasing evidence
}

interface Diagnostic {
  severity: "error" | "warning" | "info";
  code: string;                           // stable, e.g. "E_TRUNCATED", "W_OPAQUE_COMPOSITE", "I_NO_GRAPH"
  message: string;                        // one sentence a user can act on
  where?: string;                         // tensor / node / file offset
}
```

## 6.1 Compute graph

```ts
interface ComputeGraph {
  inputs: string[];                       // value names of user inputs
  outputs: string[];
  values: Record<string, { shape: (number | string)[]; dtype: DType; tensor?: string }>;  // tensor = inventory name for weights/constants
  nodes: GraphNode[];                     // topologically ordered (Section 7.4)
}

interface GraphNode {
  id: string;                             // unique, stable: source node name if unique, else "<op>_<n>"
  index: number;                          // position in topological order
  op: CanonicalOp;                        // Section 9
  source_op: string;                      // exactly as in the file, e.g. "MatMul", "FULLY_CONNECTED", "innerProduct", "aten.addmm.default"
  source_domain?: string;                 // ONNX domain, Core ML spec type, ...
  inputs: string[];                       // value names, positional ("" for omitted optional inputs)
  outputs: string[];
  attrs: Record<string, unknown>;
  module: string;                         // grouping path, e.g. "transformer.h.0.attn.c_attn" or "blk.3.ffn_up" ("" if unknown)
}
```

## 6.2 Module / group paths

`module` is what the viewer clusters on. Derive it from the source:

* ONNX: node name path (`/transformer/h.0/attn/c_attn/MatMul` → `transformer.h.0.attn.c_attn`), else the weight's name prefix.
* TFLite: output tensor name prefix. Keras: layer name (+ nested model path). Core ML: layer / op name.
* Templates: the template declares it. External JSON: its `module` field.

## 6.3 Naming rules

* Never rename a stored tensor. `name` is always the original key.
* `canonical_name` is set only by a template or an explicit mapping table, e.g. GGUF `blk.0.attn_qkv.weight` → HF `transformer.h.0.attn.c_attn.weight`. Cross-format comparison (Section 17) uses it.
* Training-checkpoint unwrapping strips `module.` (DataParallel/DDP) and `_orig_mod.` (torch.compile) prefixes **into `canonical_name` only**, and records the stripping in a diagnostic.

## 6.4 Matmul graph: `tensormorph-matmul-graph` version 2

Version 2 is a **strict superset** of the v1 JSON written by `src/gpt2_pytorch/tracing/matmul_graph.py`. Keep every v1 field name and meaning; add only the fields marked *new*. The importer must **read v1 and v2** and **write v2**.

```ts
interface MatmulGraph {
  format: "tensormorph-matmul-graph";
  version: 2;
  inputs: { name: string; shape: (number | string)[]; dtype: string }[];
  nodes: MatmulNode[];
  edges: [src: string, dst: string, role: "left" | "right" | "bias"][];   // one per (operand, upstream matmul or "input")
  outputs: string[];                      // matmuls (or "input") reaching the graph outputs
  tied_parameters: string[][];
  total_flops: number | null;
  max_rel_diff_vs_eager: number | null;   // v1 field; null when nothing was executed (always null from TypeScript)
  source: "python_export" | "file_graph" | "template";   // new
}

interface MatmulNode {
  id: string;
  index: number;                          // sequence order (Section 7.4)
  op: string;                             // v1: ATen name (mm, bmm, addmm, baddbmm, convolution); v2: = source_op for compatibility
  kind: "matmul" | "conv" | "conv_transpose" | "einsum";   // new: canonical kind, what viewers and comparisons use
  source_op: string;                      // new
  module: string;
  operands: {
    left: Operand; right: Operand; bias?: Operand;
  };
  out_shape: (number | string)[];
  contraction: { batch: number[]; m: number; n: number; k: number } | null;   // new; null if dims unbound
  flops: number | null;                   // 2 · out_numel · k
  flops_expr?: string;                    // new; symbolic form when dims are unbound, e.g. "2*batch*sequence*768*2304"
  deps: string[];                         // union of operands' `from`, in index order
  depth: number;                          // new: level in the matmul DAG (Section 7.4)
  quantized?: { dtype: string; scheme: string };   // new: weight is quantized (no dequantization performed)
}

interface Operand {
  source: string | null;                  // inventory tensor name if it resolves to a stored tensor/input, else null (activation)
  kind: "parameter" | "buffer" | "constant" | "input" | "activation";
  via: string[];                          // view-like ops looked through, nearest first, e.g. ["Transpose"], ["permute"]
  shape: (number | string)[];
  from: string[];                         // upstream matmul ids (or "input") reaching this operand, in index order
}
```

v1 → v2 upgrade when reading Python output: `kind` from `op` (`mm|bmm|addmm|baddbmm` → `matmul`, `convolution` → `conv`), `source_op = op`, `contraction` recomputed from operand shapes, `depth` computed, `source = "python_export"`.

---

# 7. Pipeline and Exact Rules

## 7.1 Stages

1. **Detect**: sniff magic bytes of the primary file (Section 3.1); extensions are hints only. `.ckpt` can be a PyTorch Lightning zip **or** a TF checkpoint prefix; `.bin` has many meanings (Section 8.11). Ambiguity → diagnostic listing candidates, then try them in a fixed order.
2. **Bundle resolution**: collect companions (external data, shards, `config.json`, `.meta`, `model.json`, `.xml`). Missing companion → `warning`, continue with what is readable.
3. **Container parse**: header/metadata only by default. Build the tensor inventory with byte ranges. Validate ranges (Section 13).
4. **Graph source**: embedded graph → else template (from `--architecture`, HF `config.json` `model_type`, GGUF `general.architecture`) → else external `--graph` → else `graph: null`.
5. **Canonicalize**: map source ops to canonical ops (Section 9). Expand known fused composites (Section 9.3). Keep `source_op`.
6. **Shape inference**: for every value on any path into a matmul operand; bind dims.
7. **Order**: validate/produce topological order (Section 7.4).
8. **Matmul extraction**: Section 11.
9. **Emit**: JSON + requested exports; diagnostics; stats.

## 7.2 Dependency semantics (must equal the Python tool)

Copy these rules from `matmul_graph.py` exactly:

* Walk nodes in topological order and keep, for each value, `reach(value)` = the set of matmul ids (or the sentinel `"input"`) whose results flow into it **without passing through another matmul**.
* User inputs: `reach = {"input"}`. Parameters, buffers, constants: `reach = {}`.
* Non-matmul node: `reach(out) = ∪ reach(inputs)`.
* Matmul node: each operand records `from = sorted(reach(operand))`; `deps = sorted(∪ from)`; then `reach(out) = {this id}`.
* Graph outputs: `outputs = sorted(∪ reach(graph outputs))`.
* Sort key: matmul `index`, with `"input"` first.
* Edges: one `[src, dst, role]` per element of every operand's `from`.
* Residual connections therefore produce edges that skip blocks. This is correct; do not "simplify" it with transitive reduction.

**Additional rule (not needed in Python, required here):** shape-only dependencies do not propagate `reach`. Inputs that a node consumes only for their **shape** (ONNX `Shape`, `Size`, `ConstantOfShape` fed by `Shape`, `Range` whose bounds come from `Shape`, TF `Shape`, TFLite `SHAPE`) contribute `{}`. Without it, an ONNX GPT-2 gets false `input → every matmul` edges through position-id computation.

## 7.3 FLOPs and contraction dims (must equal the Python tool)

* `flops = 2 · numel(out) · k`.
* matmul (`MatMul`, `Gemm`, `FullyConnected`, `BatchMatMul`, `innerProduct`, `linear`, `Dense`, `mm`, `bmm`, `addmm`): `k` = contracted length (last dim of the effective left operand after transpose flags); `m`, `n` from the output; `batch` = broadcast batch dims of the output.
* conv: `k = prod(weight.shape[1:])` in **OIHW** terms, i.e. `(C_in / groups) · prod(kernel)`. Normalize TFLite OHWI / depthwise 1HWO and Keras HWIO filters before computing. Depthwise conv = conv with `groups = C_in`.
* conv_transpose: `flops = 2 · numel(input) · (C_out / groups) · prod(kernel)`. Document the formula in the node (`flops_expr`).
* einsum (two operands): contracted letters = in both inputs and not in output; `flops = 2 · prod(size of every letter)`. More than two operands: decompose left-to-right into pairwise contractions, one node each, and emit `W_EINSUM_SPLIT`.
* Quantized ops count the same FLOPs (they are multiply-accumulates); record `quantized`.

## 7.4 Sequence order ("suitable sequence order", defined)

* **`index`**: position in a deterministic topological order of the **compute graph**, matmuls numbered in that order. Use Kahn's algorithm with priority = the node's position in the source file (ONNX node list, TFLite operator list, Keras layer order, Core ML layer/op order, template declaration order). For formats whose node order is already topological (ONNX by spec, TFLite operators), this reproduces the file order. **Validate anyway.** For formats whose order is not guaranteed (TF `GraphDef`), sort, and emit `I_RESORTED` when the order changed.
* **`depth`**: longest path from `"input"` in the matmul DAG (`depth = 1 + max(depth(dep))`, `"input"` = 0). Independent multiplications share a depth, so LLaMA's separate Q/K/V projections or gate/up projections line up side by side, while `index` still gives one serial schedule.
* Cycles: error `E_CYCLE`, no matmul graph.

## 7.5 Operand resolution (weights through views)

Follow an operand backwards through **view-like / metadata ops**: `Transpose`, `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Expand`, `Identity`, `Cast`, `Contiguous`, PyTorch `view|_unsafe_view|reshape|permute|t|transpose|expand|unsqueeze|squeeze|clone|alias`, and **dequantize** ops (`DequantizeLinear`, TFLite `DEQUANTIZE`, fp16→fp32 dequantize). Stop at a stored tensor (→ `source` = its name, `kind` from role), a user input, or any other op (→ `activation`). Record the ops passed in `via`, nearest first. A dequantize passed through sets `MatmulNode.quantized`.

`fuseBias` (default on): a matmul whose only consumer is an `Add` with a 1-D (or broadcast-compatible `[N]`) stored tensor gets that tensor as its `bias` operand. The `Add` is then absorbed. This makes ONNX `MatMul + Add` comparable with PyTorch `addmm`.

---

# 8. Per-Format Requirements

Each format section lists: detection, what to read, layout pitfalls, graph extraction, and its acceptance fixture. **Source of truth for byte layout: the named Netron file plus a real fixture.** Write the fixture first, then the parser.

## 8.1 Safetensors (`.safetensors`, sharded `model.safetensors.index.json`)

* Layout: 8-byte little-endian u64 header length `N`, then `N` bytes of UTF-8 JSON, then the data buffer. JSON maps names to `{dtype, shape, data_offsets: [begin, end]}` relative to the start of the data buffer; optional `__metadata__` (string→string). Source of truth: `netron/source/safetensors.js` and the Safetensors spec.
* Validate: `N` ≤ min(file size − 8, 100 MiB); offsets in range, non-overlapping, `end − begin == numel · dtype_size`. This repo's `tools/inspect_safetensors.py` performs the same check.
* Sharded: read `weight_map` from the index JSON, open each shard as a bundle file.
* Graph: none. Template from `config.json` (`model_type`) or `--architecture`; otherwise `I_NO_GRAPH`.
* Tied weights cannot be stored aliased. This repo's file omits `lm_head.weight` and records `tied` in `__metadata__`. Templates re-tie (Section 10).
* Fixture: `weights/model.safetensors` (82 tensors).

## 8.2 PyTorch zip pickle (`.pt`, `.pth`, `.bin`, Lightning `.ckpt`)

* Container: zip (entries are usually stored uncompressed) with `<archive>/data.pkl`, `<archive>/data/<storage key>`, `<archive>/version`, optional `<archive>/byteorder`. Source of truth: `netron/source/pytorch.js`, `zip.js`, `pickle.js`.
* **Restricted unpickler**: implement the pickle opcodes needed for state_dicts, with a **whitelist of globals** (e.g. `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch._utils._rebuild_parameter`, the `torch.*Storage` classes, `torch.Size`). It must handle the persistent-id storage tuples. Any other global → error `E_UNSAFE_PICKLE` naming the global, the same policy as `torch.load(weights_only=True)`. **Never execute anything.** Build the opcode list from `netron/source/pickle.js` and verify each against fixtures.
* Tensors: dtype from the storage type; `shape`, `strides`, `storage_offset`; tied tensors share a storage key → `tied` groups.
* **Training checkpoints**: the top-level object is often `{ "state_dict" | "model" | "model_state_dict": {...}, "optimizer": ..., "epoch": ..., ... }`. Find every nested dict whose values are all tensors. Choose by `--checkpoint-key`, else the single largest. List the alternatives in an `info` diagnostic. Optimizer state is inventoried under role `buffer` only if `--include-optimizer` is set; otherwise it is counted and skipped.
* Full-model pickles (`torch.save(model)`, globals from user modules) → `E_UNSAFE_PICKLE` with a hint: save `state_dict()`, or use `tools/matmul_graph.py` / ONNX.
* Legacy (pre-1.6, non-zip) pickle format: support read-only per `pytorch.js`.
* HF sharded `pytorch_model.bin.index.json`: like Safetensors shards.
* Graph: none (template / external).
* Fixture: `weights/pytorch_model.bin` (83 tensors, `lm_head.weight` tied to `transformer.wte.weight`, 6 `attn.bias` causal-mask buffers).

## 8.3 torch.export (`.pt2`)

* Zip containing a serialized `ExportedProgram` (graph JSON + weights). Source of truth: `netron/source/pytorch.js`, `pytorch-schema.js`, upstream `torch/_export/serde/schema.py`. The schema is versioned; read its version field and reject unknown major versions with `E_UNSUPPORTED_VERSION`.
* Graph: ATen ops, already in execution order. `graph_signature` maps placeholders to state_dict keys (the same mechanism the Python tool uses).
* Map ATen ops via Section 9. Non-decomposed `aten.scaled_dot_product_attention` is a fused composite (Section 9.3).
* Fixture: generate with `torch.export.save` from this repo's DistilGPT2 (Section 17.2).

## 8.4 TorchScript (`.pt`, `.pth` with `code/`)

* Detect and inventory the weights (`constants.pkl` / `data.pkl`). The graph is Python-subset source code: **out of scope in v1**.
* Emit `I_TORCHSCRIPT_GRAPH_UNSUPPORTED` with the re-export hint: `torch.export` → `.pt2`, `torch.onnx.export` → `.onnx`, or `tools/matmul_graph.py`.

## 8.5 GGUF (`.gguf`)

* Layout: magic `GGUF`, version, tensor count, metadata key/value pairs (typed), tensor infos (name, dims, type, offset), alignment padding (`general.alignment`, default 32), data section. Source of truth: `netron/source/gguf.js` and ggml's `gguf.h` / `ggml.c` type traits.
* **Dims order**: GGUF lists dims fastest-varying first. Normalize to row-major in `shape` and keep the raw order in `source_shape`. Verify on a fixture whose PyTorch shapes are known (Section 17.2).
* **Quantized types** (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, ...): compute byte length from the type's block size and bytes per block. Take the table from ggml's type traits, not from memory. Record `quant.scheme`. Unknown type → `E_UNKNOWN_DTYPE` for that tensor, continue with the others.
* Graph: none. Template from `general.architecture` (`gpt2`, `llama`, ...), hyperparameters from `<arch>.*` keys (block count, embedding length, head count, KV head count, feed-forward length, context length). Read the exact key names from the fixture and `gguf.js`.
* Tied output: if no output projection tensor exists, the template ties the head to the token embedding and records it in `tied`.
* Name mapping per architecture (GGUF names ↔ HF names) lives in the template's mapping table and is verified against a converted fixture.

## 8.6 ONNX (`.onnx` + external data)

* Protobuf `ModelProto` → `GraphProto` (`node`, `initializer`, `input`, `output`, `value_info`), `opset_import`. Source of truth: `netron/source/onnx.js`, `onnx-proto.js`, upstream `onnx.proto`.
* **External data**: tensors with `data_location = EXTERNAL` and `external_data` entries (`location`, `offset`, `length`) live in sibling files. Resolve through the bundle. Models over 2 GB always use this.
* `Constant` nodes and initializers become inventory entries (`constant` / `parameter`). An initializer that is also a graph input is a parameter with a default.
* Shapes: use `value_info` when present, otherwise infer (Section 7.1, step 6).
* Domains: `ai.onnx` default. `com.microsoft` contrib ops (`Attention`, `MultiHeadAttention`, `MatMulNBits`, `FusedMatMul`, `QAttention`) are fused composites or quantized matmuls (Section 9.3).
* Sub-graphs (`If`, `Loop`, `Scan`): inventory their weights; matmuls inside are reported with `W_SUBGRAPH_NOT_WALKED` (v1).
* Fixture: `torch.onnx.export` of this repo's DistilGPT2 (Section 17.2).

## 8.7 TensorFlow Lite (`.tflite`)

* FlatBuffer with identifier `TFL3`: `Model { subgraphs, buffers, operator_codes, metadata }`. Source of truth: `netron/source/tflite.js`, `tflite-schema.js`, upstream `schema.fbs`.
* Operators run in list order within a subgraph. Subgraph 0 is the main graph; others belong to control flow (`WHILE`, `IF`, `CALL_ONCE`) → `W_SUBGRAPH_NOT_WALKED`.
* Weights: tensors whose `buffer` has data. Large models may store buffer data outside the flatbuffer via offset/size fields (check the schema version). The same buffer index used twice → tied.
* Quantization: per-tensor or per-axis `scale` / `zero_point` / `quantized_dimension` → `quant`. fp16 weights appear behind `DEQUANTIZE`.
* Layout: `FULLY_CONNECTED` weights `[out, in]` (with `keep_num_dims`); `CONV_2D` filters OHWI; `DEPTHWISE_CONV_2D` 1HWO with depth multiplier; `BATCH_MATMUL` `adj_x` / `adj_y`. Normalize for FLOPs (Section 7.3); keep `source_shape`.
* Fixture: a tiny Keras model (Dense + Conv2D + DepthwiseConv2D + attention via BatchMatMul) converted with `TFLiteConverter`, float and int8 variants, with a hand-written expected matmul list.

## 8.8 Keras HDF5 (`.h5`, `.hdf5`) and Keras 3 (`.keras`)

* HDF5 is a complex container (superblock versions, object headers, B-trees, chunked + filtered datasets). Source of truth: `netron/source/hdf5.js`, `keras.js`. Choose between porting a pure-JS reader and depending on a WASM HDF5 library (Open Decision D3).
* Keras 2 full model: root attribute `model_config` (JSON string) holds the architecture; `model_weights/<layer>/.../<var>` datasets hold weights. Weights-only (`save_weights`): no `model_config` → template / external / `I_NO_GRAPH`.
* Keras 3 `.keras`: zip with `config.json`, `metadata.json`, `model.weights.h5`.
* Graph from config: `Sequential` = layer order; Functional = `layers[].inbound_nodes` + `input_layers` / `output_layers`; nested models expand with prefixed module paths. Map layers via Section 9 (`Dense` kernel is `[in, out]`; `Conv2D` HWIO; `MultiHeadAttention` and `Attention` are composites).
* Fixture: the same tiny Keras model as 8.7, saved as `.h5` (full + weights-only) and `.keras`.

## 8.9 TensorFlow checkpoint (`<prefix>.index` + `<prefix>.data-*-of-*`, TF1 `.meta`) and SavedModel

* `.index` is a table of `BundleEntryProto` records (dtype, shape, shard id, offset, size, crc). Data lives in shard files. Source of truth: `netron/source/tf.js`, `tf-proto.js`, upstream `tensor_bundle.proto`. Verify sizes against shard files.
* TF2 object-based checkpoints use keys like `layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE`, plus a serialized object graph entry. Inventory them; an object graph is **not** an op graph.
* Graph: only from a TF1 `.meta` (MetaGraphDef → GraphDef) or a SavedModel `saved_model.pb` in the bundle. `GraphDef` node order is **not** guaranteed topological → sort (Section 7.4). Ops: `MatMul`, `BatchMatMulV2/V3`, `Conv2D`, `DepthwiseConv2dNative`, `Einsum`, `BiasAdd` (bias fusion). Variables (`VarHandleOp` / `ReadVariableOp` / `VariableV2`) resolve to checkpoint keys.
* `.ckpt` disambiguation: a `.ckpt` file starting with `PK\x03\x04` is a PyTorch Lightning checkpoint (Section 8.2), not TF.
* Fixture: the tiny Keras model saved as a TF checkpoint and as a SavedModel.

## 8.10 TensorRT (`.engine`, `.plan`, `.trt`), best-effort

* Engines are serialized, **undocumented**, TensorRT-version- and GPU-specific. Netron's own `tensorrt.js` reports "undocumented TensorRT engine data". Do not reverse-engineer the layout.
* v1 behavior: detect (per `tensorrt.js`), report what is safely readable (size, any recognizable header/version string) with `W_OPAQUE_FORMAT`, `graph: null`.
* Optional companion: a layer-information JSON exported by NVIDIA's `trtexec` at build time with detailed profiling verbosity. If supplied via `--graph`, map its layers best-effort into the compute graph (`source: "file_graph"`, every node flagged `W_TRT_LAYER_INFO`). Fused regions that hide matmuls stay opaque nodes.
* Recommended path, printed in the diagnostic: import the **ONNX model the engine was built from**.

## 8.11 "Binary" (`.bin`): sniff, never assume

`.bin` resolves, in this order, to:

1. PyTorch zip or legacy pickle (`PK\x03\x04` / pickle magic) → 8.2 (HF `pytorch_model.bin` is this).
2. OpenVINO IR weights when a sibling `.xml` with the same stem exists → graph from the XML (`netron/source/openvino.js`), weights by offset/size from the `.bin`. Best-effort.
3. TF.js weight shard when a `model.json` references it → `weightsManifest` gives names, dtypes, shapes, and shard order; the `modelTopology` gives the graph when present. Best-effort.
4. ONNX external data when referenced by an `.onnx` in the bundle → 8.6.
5. Legacy GGML/GGJT files (pre-GGUF magic) → `E_UNSUPPORTED_VERSION` with "convert to GGUF".
6. Anything else (raw bytes, no header) → `E_UNKNOWN_FORMAT`. A headerless buffer has no names, dtypes, or shapes; refuse rather than guess.

## 8.12 Core ML (`.mlmodel`, `.mlpackage/`)

* Protobuf `Model` (`specificationVersion`, `description`, and one of `neuralNetwork` / classifier / regressor / `mlProgram` / `pipeline`). Source of truth: `netron/source/coreml.js`, `coreml-proto.js`, coremltools `mlmodel/format/*.proto`.
* `neuralNetwork`: layers in order. `innerProduct` (weights `[out, in]`), `batchedMatMul`, `convolution`, `embedding`. Weights in `WeightParams` (float / float16 / raw / quantized).
* `mlProgram` (MIL): functions → blocks → operations (`linear`, `matmul`, `conv`, `einsum`, ...). Constant weights reference a **blob file** inside the `.mlpackage` (`Data/com.apple.CoreML/weights/weight.bin`) by offset. Verify the blob storage header format with coremltools' storage reader and a fixture. A bare `.mlmodel` of type `mlProgram` without its package → weights missing → `W_MISSING_COMPANION`.
* `pipeline`: walk each sub-model in order, prefix module paths.
* Fixture: the tiny model converted with coremltools to both `neuralnetwork` and `mlprogram`.

## 8.13 External matmul graph JSON

* Accept `tensormorph-matmul-graph` v1 (Python) and v2. Validate against a JSON Schema shipped with the package.
* When given together with a weight file, bind `source` names to inventory tensors. An unknown name → `W_UNBOUND_OPERAND`; a shape mismatch → `E_SHAPE_MISMATCH`.

---

# 9. Canonical Operations

## 9.1 Canonical op set

Only these ops need full semantics. Everything else is `opaque` (inputs → outputs, shape from file metadata if available, otherwise unknown → its downstream matmuls get `contraction: null` + `W_SHAPE_UNKNOWN`).

| Canonical op | Semantics needed |
|---|---|
| `matmul` | batched, broadcast; `trans_a`, `trans_b`, `alpha`, `beta`, optional fused bias |
| `conv`, `conv_transpose` | N-d, strides, pads, dilations, groups; normalized OIHW weight |
| `einsum` | equation parse, two operands (Section 7.3) |
| `gather` | embedding lookup (not a multiplication; reach passes from the index input) |
| `add`, `sub`, `mul`, `div`, `pow`, `where` | broadcasting elementwise (`mul` is listed in elementwise mode) |
| `unary` | exp, tanh, erf, sqrt, rsqrt, log, relu, gelu, silu, sigmoid, ... (`attrs.fn`) |
| `reduce` | mean/sum/max/min over axes, keepdims |
| `softmax`, `layernorm`, `rmsnorm`, `batchnorm` | shape-preserving composites, not multiplications |
| view-like | `reshape`, `transpose`, `squeeze`, `unsqueeze`, `flatten`, `expand`, `slice`, `split`, `concat`, `cast`, `identity`, `dequantize`, `quantize` |
| `shape_of`, `range`, `constant_of_shape` | shape-only (Section 7.2) |
| `opaque` | anything else; `source_op` keeps the original |

## 9.2 Source → canonical (matmul family)

| Source | Ops → `matmul` | → `conv` / `conv_transpose` | → `einsum` |
|---|---|---|---|
| ONNX | `MatMul`, `Gemm`, `MatMulInteger`, `QLinearMatMul`, `com.microsoft.FusedMatMul`, `com.microsoft.MatMulNBits` | `Conv`, `ConvInteger`, `QLinearConv` / `ConvTranspose` | `Einsum` |
| TFLite | `FULLY_CONNECTED`, `BATCH_MATMUL` | `CONV_2D`, `DEPTHWISE_CONV_2D`, `CONV_3D` / `TRANSPOSE_CONV` | — |
| Keras | `Dense`, `EinsumDense` (→ einsum) | `Conv1D/2D/3D`, `DepthwiseConv2D`, `SeparableConv2D` (= depthwise + pointwise: two nodes) / `Conv2DTranspose` | `EinsumDense` |
| TF GraphDef | `MatMul`, `BatchMatMul*` | `Conv2D`, `Conv3D`, `DepthwiseConv2dNative` / `Conv2DBackpropInput` | `Einsum` |
| Core ML NN | `innerProduct`, `batchedMatMul` | `convolution` (`isDeconvolution` → transpose) | — |
| Core ML MIL | `linear`, `matmul` | `conv` / `conv_transpose` | `einsum` |
| torch.export ATen | `mm`, `bmm`, `addmm`, `baddbmm`, `matmul`, `linear` | `convolution` (`transposed` flag), `conv1d/2d/3d` | `einsum` |
| Templates | declared directly | declared | declared |

## 9.3 Fused composites: expand or flag, never drop

Ops that contain matmuls internally (ONNX `com.microsoft.Attention` / `MultiHeadAttention` / `QAttention`, ATen `scaled_dot_product_attention`, Keras `MultiHeadAttention` / `Attention`, Core ML MIL `scaled_dot_product_attention`, TensorRT fused regions):

* If a rule exists, **expand** into canonical ops (projections when the composite owns weights; `scores = Q·Kᵀ`, `softmax`, `context = P·V`), keeping `module` = the composite and `source_op` = the composite name on each produced node.
* Otherwise emit one `opaque` node plus `W_OPAQUE_COMPOSITE` naming the op, so the viewer shows a marked gap instead of a silently shorter list.
* v1 must ship expansion rules for ATen SDPA, ONNX contrib `Attention` / `MultiHeadAttention`, and Keras `MultiHeadAttention`.

---

# 10. Architecture Templates (weight-only inputs)

A template turns a tensor inventory + hyperparameters into a compute graph. One file per architecture, pure data + a small builder:

```ts
interface ArchitectureTemplate {
  name: string;                                         // "gpt2"
  matches(ctx: { config: Record<string, unknown>; tensors: TensorInfo[]; format: FormatId }): boolean;
  resolveConfig(ctx): Record<string, unknown>;          // from config.json / GGUF kv / shapes; assert every shape it can check
  nameMaps: Record<string, Record<string, string>>;     // per source naming scheme (hf, gguf, ...) → canonical template names
  build(b: GraphBuilder, config): void;                 // emits canonical ops in execution order, with module paths
}
```

Rules:

* **Verify before trusting**: every weight the template binds must exist with the expected shape (after layout normalization), and every stored tensor must be bound or explicitly ignored (like this repo's `attn.bias` causal-mask buffers). Otherwise `E_TEMPLATE_MISMATCH` listing missing / unexpected / mis-shaped tensors, and no graph. This mirrors `checkpoint/loader.py`.
* Hyperparameters that cannot be recovered from shapes (for example GPT-2's head count) must come from `config.json`, GGUF metadata, or `--config`. If the template has to assume one, it records the assumption as `W_ASSUMED_HYPERPARAMETER`. This repo assumes head_dim 64 and checks it.
* Layout: the template knows HF GPT-2 `Conv1D` weights are `[in, out]` (no transpose op is emitted; `layout_note` records it), while Linear-style weights are `[out, in]` (a `transpose` view is emitted so `via` shows it).
* Ties: the template declares tied pairs (GPT-2 `lm_head` ↔ `wte`). A missing tied copy is not an error.

## 10.1 `gpt2` template (DistilGPT2 is gpt2 with 6 layers)

Per block, in this order, with HF module paths: `ln_1` → `attn.c_attn` (matmul + bias) → split Q/K/V → heads → `Q·Kᵀ` (matmul, activation × activation) → scale → causal mask → softmax → `P·V` (matmul) → merge heads → `attn.c_proj` (matmul + bias) → residual add → `ln_2` → `mlp.c_fc` (matmul + bias) → gelu (tanh approximation) → `mlp.c_proj` (matmul + bias) → residual add. Then `ln_f` → `lm_head` (matmul with `wte`, transposed view). The reference implementation is `src/gpt2_pytorch/model/` and `src/gpt2_pytorch/tensorprog/lower.py`. The template must produce the same 6 matmuls per block + 1 head, in the same order, as `artifacts/matmul_graph.json`.

## 10.2 `llama` template

Embedding → per block: `attn_norm` (rmsnorm) → Q, K, V projections (three independent matmuls, same `depth`) → RoPE (elementwise) → grouped-query attention (`Q·Kᵀ` and `P·V` with `n_head / n_head_kv` sharing) → output projection → residual → `ffn_norm` → gate and up projections (independent, same `depth`) → SiLU · up (elementwise) → down projection → residual. Then final norm → output projection (or tied embedding). Hyperparameters and name maps from GGUF metadata / HF `config.json`, verified on a fixture.

---

# 11. Matmul Extraction Algorithm

```text
reach = {}; order = {"input": -1}; nodes = []
for node in graph.nodes (topological):
    if node is a graph input placeholder:  reach[out] = {"input"}
    if node is a stored tensor / constant: reach[out] = {}
    for each input i of node:
        r_i = {} if node uses i only for its shape (Section 7.2) else reach[i]
    if node.op in {matmul, conv, conv_transpose, einsum}:
        for role, operand in (left, right, bias?):
            src, kind, via, quantized = resolve(operand)            # Section 7.5
            from = sort(reach[operand], key=order)
        record MatmulNode(id, index=len(nodes), kind, source_op, module, operands, out_shape,
                          contraction, flops, deps = sort(∪ from), depth = 1 + max(depth[d]), quantized)
        reach[node.out] = {id}; order[id] = index
    else:
        reach[each out] = ∪ r_i
outputs = sort(∪ reach[graph outputs])
edges   = [(f, node.id, role) for node, role, f in every operand.from]
```

With `includeElementwise`, `mul` nodes (and `div` by a constant is **not** a multiplication) are also listed. They appear as `kind: "elementwise_mul"` in a separate `elementwise` array with the same operand / `from` structure, matching the 75 elementwise products of `artifacts/tensor_multiplications.csv` for DistilGPT2. Contraction nodes and edges are unaffected.

---

# 12. Visualization Contract

The viewer is out of scope, but the data must make these views trivial to build:

* **Sequence view**: matmuls sorted by `index`, one row each: `index`, `module`, `kind`, `left` / `right` source names (or "activation"), `M×K · K×N` (+ batch), `flops`, `deps`.
* **DAG view**: nodes = matmuls (+ an `input` node and an `output` node), edges from `edges` labelled by role, layered by `depth`, clustered by `module` prefix (block level).
* **Repeated-block folding**: compute a per-block signature (sequence of `kind` + contraction dims + relative operand wiring). Consecutive identical blocks can be shown as "×N" (`blocks: [{ module_prefix, node_ids, signature }]` in the JSON).
* **Cost view**: `flops` per node and `total_flops`; share per module.
* Exports:
  * **Mermaid** `graph TD`, edges `a -- role --> b` (same as the Python `to_markdown`).
  * **DOT** with `rank=same` per `depth`, `subgraph cluster_<block>`.
  * **CSV** with the columns of `artifacts/tensor_multiplications.csv` (`n,op,kind,output,left,right,summed_axes,cost,module`) so existing tooling reads it.

---

# 13. Security and Robustness (uploaded files are untrusted)

* **No code execution**, ever: restricted unpickler with a global whitelist; no `eval`; no dynamic import based on file content.
* **Bounds**: every offset/length is validated against the file size before reading. Integer math uses `number` with explicit checks below 2^53 (or `bigint` for 64-bit fields), never 32-bit bitwise ops on sizes.
* **Resource limits** (configurable): JSON header size (default 100 MiB), metadata entry count, string lengths, protobuf / flatbuffer recursion depth, pickle memo size, zip entry count, decompressed-size ratio (zip bombs), graph node count, tensor rank (≤ 16).
* **Malformed input** → a typed `error` diagnostic and exit 1, never an unhandled exception, never partial output that looks complete. Fuzz every parser with truncated, bit-flipped and oversized inputs (Section 17.4).
* **Paths**: companion resolution never escapes the bundle root (`..`, absolute paths in ONNX `location`, zip entry names).
* Payloads are not read by default (Section 14); `readPayload` is opt-in.

---

# 14. Performance

* **Header-only by default**: importing `weights/model.safetensors` must read under 64 KiB, and `stats.payload_bytes_read` must be 0. TEL's importer reads 8,376 header bytes for this file; match that order of magnitude.
* Instrument every `ByteSource` read and report `bytes_read`, `header_bytes`, `payload_bytes_read` in `stats`. Tests assert on them.
* Files larger than 4 GiB (GGUF, sharded checkpoints, ONNX external data) must work in Node without loading them into memory.
* Graph work is linear in nodes + edges; `reach` sets are bounded by the number of matmuls (L² growth from residuals is expected: DistilGPT2 has 127 edges for 37 matmuls).
* Target: DistilGPT2 from Safetensors + template, full pipeline, < 200 ms in Node on a laptop.

---

# 15. Project Structure

Proposed (adapt to the chosen home, Open Decision D1):

```text
tensormorph-model-import/
├── src/
│   ├── index.ts                 # importModel, types re-export
│   ├── types.ts                 # ModelPackage, TensorInfo, ComputeGraph, MatmulGraph (Section 6)
│   ├── io/                      # ByteSource (node, browser), bundle, counting reader, binary reader,
│   │                            # zip, protobuf wire decoder, flatbuffer reader, restricted pickle, hdf5
│   ├── detect.ts                # magic-byte sniffing, candidate ordering, .bin/.ckpt disambiguation
│   ├── formats/                 # one folder per format: safetensors, pytorch, pt2, gguf, onnx, tflite,
│   │                            # keras, tf, coreml, tensorrt, openvino, tfjs, matmul-json
│   ├── graph/                   # canonical ops, builder, shape inference, toposort, bias fusion, composites
│   ├── templates/               # gpt2.ts, llama.ts (+ name maps)
│   ├── matmul/                  # extraction, FLOPs, depth, block signatures
│   ├── emit/                    # json, mermaid, dot, csv
│   └── cli.ts
├── schema/                      # JSON Schemas: tensormorph-model v1, tensormorph-matmul-graph v1 + v2
├── fixtures/                    # small generated fixtures (+ scripts that generate them, Section 17.2)
├── test/
├── THIRD_PARTY_NOTICES.md       # Netron MIT notice for any ported code
└── DECISIONS.md
```

Each format folder exposes the same two functions: `detect(head: Uint8Array, name: string): Confidence` and `parse(bundle, options): Promise<FormatResult>` (inventory + optional source graph + diagnostics). No format code outside its folder.

---

# 16. Dependencies

* TypeScript `strict`, ESM, no `any` in public types, Node ≥ 20 and evergreen browsers. Tests with `node:test` (built in).
* **Runtime dependencies: zero by default.** Each addition needs a line in `DECISIONS.md` (why, size, license). Candidates where hand-writing is unreasonable:
  * zip inflate: prefer the platform `DecompressionStream("deflate-raw")` where available; PyTorch zips are usually stored uncompressed; otherwise a small inflater (port `netron/source/zip.js` or a tiny library).
  * HDF5: port a pure-JS reader (`netron/source/hdf5.js`) or depend on a WASM HDF5 build (Open Decision D3).
  * protobuf / flatbuffers: a hand-written wire-format decoder for only the fields used (ONNX, Core ML, TF) is small. Generated code is acceptable if it is kept to the messages actually read.
* Fixture generation scripts are Python and live in this repo or `fixtures/scripts/`. Their dependencies (`onnx`, `tensorflow`, `coremltools`, llama.cpp's converter) are dev-only and optional; each fixture is committed in generated form so TypeScript tests never need Python.

---

# 17. Testing and Golden Validation

## 17.1 Golden: DistilGPT2 equals the Python oracle

`artifacts/matmul_graph.json` (Python, `torch.export` decomposition, B=2, T=8) is the reference. Measured values:

| Quantity | Value |
|---|---|
| matmuls | **37** (per block: qkv, Q·Kᵀ, P·V, attn out, fc, proj; × 6; + lm_head) |
| edges | **127** |
| total FLOPs | **2,596,429,824** |
| first node | `addmm`, weight `transformer.blocks.0.attention.qkv_projection.weight`, deps `["input"]` |
| last node | `mm`, weight `lm_head.weight` (tied to the token embedding) |

Comparison is by **index-aligned signature**, not by ids or ATen op names. For each node: `kind`; `contraction` (`prod(batch)`, `m`, `n`, `k`), which absorbs the shape flattening the Python decomposition does (`[16, 2304]` vs `[2, 8, 2304]`, `[24, 8, 8]` vs `[2, 12, 8, 8]`); `flops`; the canonical weight name of the parameter operand; operand `from` lists translated to indices. Python names are this repo's module names. Translate them to HF names with `src/gpt2_pytorch/checkpoint/mapping.py` (write a small Python script that emits `fixtures/golden/distilgpt2.signature.json` with both names).

These inputs must all produce that signature:

1. `weights/model.safetensors` + `gpt2` template (config from shapes + `n_head=12`).
2. `weights/pytorch_model.bin` + `gpt2` template. The inventory must also match the Safetensors one tensor-for-tensor (minus the stored `lm_head.weight` copy, reported as tied).
3. `artifacts/matmul_graph.json` imported as v1 → v2 (no weight file), and again bound to `weights/model.safetensors` via `--graph`.
4. ONNX export of the same model (Section 17.2). Bias fusion and the shape-only rule must make it match; any remaining difference is a documented, tested exception, not a tolerance.
5. GGUF conversion of DistilGPT2 + `gpt2` template.
6. `.pt2` from `torch.export.save`.

## 17.2 Fixtures to generate (scripts committed, outputs committed)

* DistilGPT2: ONNX (`torch.onnx.export`, static B=2, T=8 and a dynamic-axes variant), `.pt2`, GGUF (llama.cpp's HF→GGUF converter; f32 / f16 / q8_0), a training checkpoint `{"model": state_dict, "optimizer": ..., "epoch": 3}` with `module.` prefixes, a full-model pickle (must be refused).
* Tiny Keras model (Dense, Conv2D, DepthwiseConv2D, MultiHeadAttention, residual): `.h5` (full and weights-only), `.keras`, TF checkpoint, SavedModel, TFLite float and int8, Core ML `neuralnetwork` and `mlprogram`. Each comes with a **hand-written expected matmul list** (count, order, contraction dims, FLOPs, deps).
* Tiny LLaMA-shaped model (2 layers, GQA): HF Safetensors + `config.json`, and GGUF.
* TensorRT: any small engine if available on the build machine; otherwise a stub that tests detection and the `W_OPAQUE_FORMAT` path only.

Keep fixtures small (< 5 MB each) except files already in `weights/`. Tests that need `weights/` skip when it is absent, as this repo's pytest suite does.

## 17.3 Unit tests

Per parser: header fields, inventory, dtype/size validation, layout normalization (`source_shape` vs `shape`), tied detection. Graph: toposort determinism, shape-only rule, bias fusion, view-through resolution, composite expansion, FLOP formulas (matmul, Gemm with transposes, conv with groups, depthwise, conv_transpose, einsum, broadcast batch), `depth` on a diamond graph. Emitters: snapshot tests for JSON / Mermaid / DOT / CSV.

## 17.4 Negative and robustness tests

Truncated files at every structural boundary; header length > file size; overlapping or out-of-range offsets; dtype/size mismatch; unknown GGUF type; unsafe pickle global; full-model pickle; zip bomb; path traversal in ONNX `location` and zip entries; cyclic graph; missing companion; TensorRT engine. Each must give the documented diagnostic code. A small fuzz loop (random truncation + byte flips over every fixture, fixed seed) must never throw an unhandled exception or hang.

## 17.5 Determinism

Importing the same input twice yields byte-identical JSON. Node and browser `ByteSource`s yield identical JSON.

---

# 18. Phases

Each phase ends with its tests green and the Definition of Done items for that phase checked.

| Phase | Deliverable | Why first |
|---|---|---|
| **P0** | types + JSON Schemas, external JSON reader (v1→v2), matmul post-processing (`contraction`, `depth`, blocks), emitters (JSON, Mermaid, DOT, CSV), CLI skeleton | visualization works immediately on the Python output |
| **P1** | `ByteSource`, detection, Safetensors (+ shards), PyTorch zip/legacy + restricted unpickler + training-checkpoint unwrapping, HF `config.json`, `gpt2` template, golden 17.1 items 1–3 | covers "upload trained PyTorch weights" end to end |
| **P2** | ONNX (+ external data), canonical ops, shape inference, toposort, bias fusion, shape-only rule, composite expansion; golden item 4 | first general graph-carrying format; exercises the whole graph pipeline |
| **P3** | GGUF (quant type table, dim order), `gpt2` + `llama` templates, golden item 5; `.pt2`, golden item 6 | most common LLM distribution format |
| **P4** | TFLite, Keras `.h5` / `.keras`, TF checkpoint + `.meta` / SavedModel | TensorFlow family |
| **P5** | Core ML `.mlmodel` / `.mlpackage` | Apple family |
| **P6** | TensorRT best-effort, OpenVINO / TF.js `.bin` companions, fuzzing hardening, performance targets | last because least documented |

---

# 19. Definition of Done

* [ ] Every format of Section 3.1 is detected by content. Each one either imports (inventory always; graph where the format or a template provides one) or fails with a documented diagnostic code. No format silently produces a guessed graph.
* [ ] `ModelPackage` v1 and `tensormorph-matmul-graph` v2 JSON Schemas are published. Every fixture output validates. v1 Python output is read and upgraded.
* [ ] Golden 17.1: all six DistilGPT2 inputs give 37 matmuls, 127 edges and 2,596,429,824 FLOPs, with index-aligned signatures equal to `artifacts/matmul_graph.json` (after name mapping).
* [ ] Every Keras/TFLite/TF/Core ML/LLaMA fixture matches its hand-written expected matmul list.
* [ ] Dependency, ordering, FLOP and operand-resolution rules (Section 7) are implemented exactly, each with a unit test. The shape-only rule and bias fusion are tested on ONNX.
* [ ] Header-only import of `weights/model.safetensors` reads < 64 KiB with `payload_bytes_read == 0`. Files > 4 GiB work in Node.
* [ ] Restricted unpickler refuses non-whitelisted globals. Fuzz loop passes. Path traversal is blocked.
* [ ] Output is deterministic (byte-identical across runs and between Node and browser).
* [ ] Mermaid, DOT and CSV exports exist, and the CSV matches the column set of `artifacts/tensor_multiplications.csv`.
* [ ] README: supported formats table (with graph source per format), the "trained PyTorch weights + own code" path (Section 3.3), diagnostic code reference, and an example per format.
* [ ] `THIRD_PARTY_NOTICES.md` credits any code ported from Netron (MIT), and `DECISIONS.md` records every dependency and every Open Decision.

---

# 20. Open Decisions (record in `DECISIONS.md`, do not block)

* **D1: Where the code lives.** A standalone package (e.g. `tensormorph-model-import`), or an extension of TEL's TypeScript `packages/model-import` (`tensorparser`), which already reads GPT-2 Safetensors headers (see `docs/TEL_NEXT_STEPS.md`). Check TEL's importer first. If it is the Tensormorph TypeScript home, extend it and keep this prompt's schemas as its output contract.
* **D2: Port vs. dependency for protobuf / flatbuffers.** Hand-written decoders for the fields used vs. generated code.
* **D3: HDF5.** Port `netron/source/hdf5.js` (pure JS, read-only) vs. a WASM HDF5 library (larger, more complete).
* **D4: Default example dims** when a graph has symbolic dims (Section 5.3), and whether B=2, T=8 (the Python golden) should be the CLI default for language models.
* **D5: Elementwise multiplications** (Section 11): separate `elementwise` array (proposed) vs. mixed into `nodes` with a `kind` filter.
* **D6: TensorRT.** Accept `trtexec` layer-info JSON in v1, or defer entirely to "import the source ONNX".
