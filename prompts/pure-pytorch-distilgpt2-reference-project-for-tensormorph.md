# Build `GPT2PytorchContruct`: Pure PyTorch DistilGPT2 Reference Project for Tensormorph

You are a senior **PyTorch internals engineer, transformer architecture engineer, model-format engineer, and ML debugging/tooling engineer**.

Your task is to inspect an existing Hugging Face DistilGPT2 checkpoint and build a complete, educational, highly inspectable **pure PyTorch implementation of DistilGPT2**.

This project is intended primarily as a reference/demo model for **Tensormorph**, especially for:

- tensor inspection,
- parameter inspection,
- Safetensors inspection,
- PyTorch execution tracing,
- forward-pass debugging,
- module hierarchy visualization,
- tensor-shape visualization,
- intermediate activation capture,
- operator tracing,
- checkpoint loading,
- checkpoint conversion,
- state-dict analysis,
- TensorGraph generation,
- TensorView experiments,
- Tensor IR experiments,
- PyTorch workflow demonstrations.

The project must favor **clarity, explicit tensor operations, deterministic execution, and inspectability** over abstraction.

---

# 1. TARGET PROJECT

Project root:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct
```

An existing Hugging Face checkpoint has already been downloaded:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct/pytorch_model.bin
```

The original model is:

```text
distilbert/distilgpt2
```

Reference Hugging Face model:

```text
https://huggingface.co/distilbert/distilgpt2
```

The checkpoint originates from the architecture normally loaded using:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilbert/distilgpt2",
    device_map="auto",
)
```

However, the implementation produced by this project must **NOT depend on Hugging Face Transformers for the model architecture**.

The final model architecture must be built directly from:

```python
torch.nn.Module
torch.nn.Linear
torch.nn.Embedding
torch.nn.LayerNorm
torch.nn.Dropout
torch.nn.ModuleList
```

and explicit PyTorch tensor operations.

---

# 2. PRIMARY OBJECTIVE

Reconstruct DistilGPT2 as an explicit hierarchy of pure PyTorch modules.

Conceptually:

```text
DistilGPT2LMHeadModel
└── GPT2Model
    ├── token embedding
    ├── positional embedding
    ├── dropout
    ├── TransformerBlock × N
    │   ├── LayerNorm
    │   ├── CausalSelfAttention
    │   │   ├── QKV projection
    │   │   └── output projection
    │   ├── residual connection
    │   ├── LayerNorm
    │   ├── MLP
    │   │   ├── expansion projection
    │   │   ├── GELU
    │   │   └── projection
    │   └── residual connection
    ├── final LayerNorm
    └── LM head
```

Do not wrap the implementation around:

```python
GPT2Model
GPT2LMHeadModel
AutoModel
AutoModelForCausalLM
GPT2Config
```

from `transformers`.

The architecture itself must belong entirely to this repository.

---

# 3. FIRST: INSPECT THE EXISTING CHECKPOINT

Before implementing the architecture, inspect:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct/pytorch_model.bin
```

Do not begin by blindly hardcoding assumptions.

Create utilities that load and inspect the checkpoint.

Prefer safe checkpoint loading where supported:

```python
torch.load(
    path,
    map_location="cpu",
    weights_only=True,
)
```

Determine:

- all state-dict keys,
- tensor names,
- tensor shapes,
- dtypes,
- number of tensors,
- total scalar count,
- total parameter bytes,
- hierarchy implied by key names,
- number of transformer blocks,
- embedding dimensions,
- vocabulary size,
- positional embedding count,
- MLP expansion dimension,
- attention projection dimensions,
- LayerNorm dimensions.

Print information similar to:

```text
transformer.wte.weight
  shape = [50257, 768]
  dtype = torch.float32

transformer.wpe.weight
  shape = [1024, 768]

transformer.h.0.ln_1.weight
  shape = [768]

transformer.h.0.attn.c_attn.weight
  shape = [768, 2304]

transformer.h.0.attn.c_attn.bias
  shape = [2304]

...
```

Use the checkpoint itself to verify the architecture.

---

# 4. EXPECTED DISTILGPT2 CONFIGURATION

After inspecting the checkpoint, verify whether it corresponds to approximately:

```python
vocab_size = 50257
max_position_embeddings = 1024
hidden_size = 768
num_attention_heads = 12
num_hidden_layers = 6
intermediate_size = 3072
```

Calculate:

```python
head_dim = hidden_size // num_attention_heads
```

Expected:

```text
head_dim = 64
```

Do not silently assume these values.

Assert that the checkpoint confirms them.

Create an explicit configuration object such as:

```python
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_epsilon: float = 1e-5
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
```

Prefer names understandable outside Hugging Face internals.

---

# 5. IMPORTANT: HUGGING FACE GPT2 `Conv1D`

Pay special attention to Hugging Face GPT2's historical `Conv1D` implementation.

Despite its name, this is effectively a linear projection with an unusual weight layout.

For example:

```text
attn.c_attn.weight
[768, 2304]
```

is logically:

```text
input_dim = 768
output_dim = 2304
```

while:

```python
nn.Linear(768, 2304)
```

stores its weight as:

```text
[2304, 768]
```

Therefore checkpoint conversion may require:

```python
linear.weight.copy_(hf_weight.T)
```

Handle this explicitly.

Identify all affected GPT2 projections, likely including:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
mlp.c_proj
```

Never transpose embeddings or LayerNorm parameters.

Create a centralized checkpoint mapping layer rather than scattering transpose logic throughout the modules.

For example:

```text
checkpoint/
├── inspect.py
├── mapping.py
└── loader.py
```

---

# 6. ARCHITECTURE MODULES

Decompose the implementation into explicit modules.

Do not place the entire model inside one Python file.

At minimum implement:

## Embeddings

```text
TokenEmbedding
PositionEmbedding
GPT2Embeddings
```

Input:

```text
input_ids
[B, T]
```

Output:

```text
hidden_states
[B, T, C]
```

Explicitly show:

```python
token_embeddings = self.token_embedding(input_ids)
position_embeddings = self.position_embedding(position_ids)

hidden_states = token_embeddings + position_embeddings
```

---

# 7. CAUSAL SELF ATTENTION

Implement a dedicated:

```python
CausalSelfAttention(nn.Module)
```

Use regular:

```python
nn.Linear
```

for checkpoint-loaded projections.

Prefer explicit Q/K/V decomposition over hiding everything behind:

```python
torch.nn.MultiheadAttention
```

Do NOT use `nn.MultiheadAttention`.

The tensor flow should be obvious.

Starting from:

```text
hidden_states
[B, T, C]
```

compute:

```python
qkv = self.qkv_projection(hidden_states)
```

Shape:

```text
[B, T, 3C]
```

then:

```python
query, key, value = qkv.chunk(3, dim=-1)
```

Reshape explicitly:

```text
[B, T, C]

→

[B, T, H, D]

→

[B, H, T, D]
```

where:

```text
H = number of heads
D = head dimension
```

Calculate attention scores explicitly:

```python
attention_scores = query @ key.transpose(-2, -1)
attention_scores = attention_scores / sqrt(head_dim)
```

Shape:

```text
[B, H, T, T]
```

Apply causal masking explicitly.

Create a causal mask such as:

```text
[T, T]
```

with future token positions masked.

Then:

```python
attention_probs = softmax(attention_scores, dim=-1)
```

and:

```python
context = attention_probs @ value
```

Then convert:

```text
[B, H, T, D]

→

[B, T, H, D]

→

[B, T, C]
```

Finally apply:

```python
output_projection
```

Keep each important tensor accessible for tracing/debugging.

---

# 8. DO NOT HIDE ATTENTION BEHIND SDPA INITIALLY

For the primary implementation, do NOT use:

```python
torch.nn.functional.scaled_dot_product_attention
```

and do not use FlashAttention.

We want the individual tensors to remain visible:

```text
query
key
value
attention_scores
attention_mask
attention_probs
context
```

This project is for Tensormorph debugging and visualization, so transparency is more important than maximum performance.

A later optional optimized implementation may use SDPA, but it must not replace the inspectable reference path.

---

# 9. MLP

Create:

```python
GPT2MLP(nn.Module)
```

Explicitly represent:

```text
hidden_size
768

→ expansion

3072

→ activation

3072

→ projection

768
```

Use:

```python
nn.Linear(768, 3072)
```

followed by GPT2-compatible GELU.

Verify exactly which GELU formulation the checkpoint/model expects.

Implement the appropriate GPT2 GELU explicitly if necessary instead of assuming plain:

```python
F.gelu(x)
```

without checking compatibility.

Then:

```python
nn.Linear(3072, 768)
```

---

# 10. TRANSFORMER BLOCK

Create:

```python
TransformerBlock(nn.Module)
```

Make the residual structure explicit:

```python
residual = hidden_states

normalized = self.ln_1(hidden_states)

attention_output = self.attention(normalized)

hidden_states = residual + attention_output
```

Then:

```python
residual = hidden_states

normalized = self.ln_2(hidden_states)

mlp_output = self.mlp(normalized)

hidden_states = residual + mlp_output
```

This should intentionally expose intermediate tensors rather than compressing them into one-line expressions.

---

# 11. GPT2 BACKBONE

Create:

```python
GPT2Model(nn.Module)
```

It should contain:

```python
self.token_embedding
self.position_embedding
self.embedding_dropout

self.blocks = nn.ModuleList([...])

self.final_layer_norm
```

The forward flow should be straightforward:

```text
input_ids
    ↓
token embeddings
    +
position embeddings
    ↓
block 0
    ↓
block 1
    ↓
...
    ↓
block 5
    ↓
final LayerNorm
    ↓
hidden states
```

---

# 12. LM HEAD

Create:

```python
DistilGPT2LMHeadModel(nn.Module)
```

containing:

```python
self.transformer = GPT2Model(...)
self.lm_head = nn.Linear(
    hidden_size,
    vocab_size,
    bias=False,
)
```

Investigate whether the original model ties:

```text
lm_head.weight
```

to:

```text
transformer.wte.weight
```

If so, reproduce true weight tying rather than copying the tensor:

```python
self.lm_head.weight = self.transformer.token_embedding.weight
```

Document this behavior.

Output:

```text
logits
[B, T, vocab_size]
```

---

# 13. FIXED DETERMINISTIC INPUT

The main demo must NOT depend on a tokenizer.

Create a deterministic fixed token sample directly as tensor data.

Example:

```python
FIXED_INPUT_IDS = torch.tensor(
    [
        [15496, 11, 616, 1438, 318, 1332, 13],
        [1212, 318, 257, 1332, 6827, 284, 1332],
    ],
    dtype=torch.long,
)
```

The exact IDs can be changed if needed, but the final sample must:

- have fixed values,
- have fixed dimensions,
- be checked into source control,
- never be randomly generated,
- require no network access,
- be deterministic across runs.

For debugging, use a simple shape such as:

```text
batch = 2
sequence_length = 8
```

or another small fixed value.

Create multiple samples if useful:

```text
sample_single
sample_batch
sample_repeated_tokens
sample_short_sequence
sample_attention_debug
```

---

# 14. DATASET AND DATALOADER

Although the data is fixed, demonstrate the normal PyTorch workflow.

Implement:

```python
class FixedTokenDataset(torch.utils.data.Dataset):
    ...
```

and:

```python
DataLoader(
    dataset,
    batch_size=...,
    shuffle=False,
)
```

Do not introduce randomness.

The purpose is to demonstrate:

```text
Dataset
    ↓
DataLoader
    ↓
Batch
    ↓
nn.Module
    ↓
Forward
    ↓
Loss / logits
```

Create an optional fixed next-token target tensor.

For example:

```text
input_ids:
[B, T]

labels:
[B, T]
```

allowing demonstration of causal language modeling loss.

---

# 15. LOSS

Implement optional causal LM loss explicitly.

Perform the standard shift:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

Then:

```python
F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
)
```

Expose:

```text
logits
shift_logits
shift_labels
loss
```

for inspection.

---

# 16. FIXED FORWARD DEMO

Provide a command like:

```bash
python -m examples.forward_fixed
```

It should:

1. load config,
2. instantiate the pure PyTorch model,
3. load `pytorch_model.bin`,
4. load a deterministic DataLoader batch,
5. run `model.eval()`,
6. use `torch.no_grad()`,
7. run one forward pass,
8. print important tensor shapes,
9. print several deterministic values,
10. print logits statistics.

Example output:

```text
input_ids             [2, 8]
token_embeddings      [2, 8, 768]
position_embeddings   [2, 8, 768]

block.0.input          [2, 8, 768]
block.0.q              [2, 12, 8, 64]
block.0.k              [2, 12, 8, 64]
block.0.v              [2, 12, 8, 64]
block.0.attn_scores    [2, 12, 8, 8]
block.0.attn_probs     [2, 12, 8, 8]
block.0.output         [2, 8, 768]

...

final_hidden           [2, 8, 768]
logits                 [2, 8, 50257]
```

---

# 17. ACTIVATION CAPTURE

Because this project will be used by Tensormorph, provide a robust activation inspection mechanism.

Implement something such as:

```python
ActivationRecorder
```

Support capturing:

```text
module name
module type
input tensor
output tensor
shape
dtype
device
min
max
mean
std
numel
bytes
```

For important attention tensors, capture internal intermediates too:

```text
q
k
v
attention_scores
attention_probs
context
```

Do not require rewriting model code to inspect each layer.

Possible API:

```python
with ActivationRecorder(model) as recorder:
    logits = model(input_ids)
```

Then:

```python
for record in recorder.records:
    print(record)
```

---

# 18. FORWARD HOOK DEMO

Provide:

```bash
python -m examples.forward_hooks
```

Demonstrating:

```python
register_forward_pre_hook
register_forward_hook
```

Capture the execution order.

Produce output like:

```text
000 TokenEmbedding
001 PositionEmbedding
002 LayerNorm
003 Linear
004 CausalSelfAttention
...
```

This is important for TensorGraph/Tensormorph experiments.

---

# 19. PYTORCH FX TRACE

Provide an FX tracing example if the model can be made compatible:

```bash
python -m examples.fx_trace
```

Experiment with:

```python
torch.fx.symbolic_trace
```

If some operations prevent straightforward tracing, document why and provide a tracing-friendly wrapper.

Print:

```python
graph.print_tabular()
```

Optionally save:

```text
artifacts/fx_graph.txt
```

and:

```text
artifacts/fx_graph.json
```

if useful.

Do not distort the primary model architecture merely to make FX happy.

---

# 20. TORCH EXPORT

Provide an example using modern PyTorch:

```python
torch.export.export(...)
```

where possible.

Example:

```bash
python -m examples.export_model
```

Use the fixed input tensor as `example_inputs`.

Save the resulting graph information under:

```text
artifacts/export/
```

This is particularly relevant to Tensormorph's TensorGraph / Tensor IR work.

---

# 21. SAFETENSORS CONVERSION

Add support for converting:

```text
pytorch_model.bin
```

into:

```text
model.safetensors
```

without changing tensor values.

Create:

```bash
python -m tools.convert_to_safetensors
```

Use:

```python
safetensors.torch.save_file
```

Be aware of shared/tied parameters.

Safetensors cannot naively serialize arbitrary storage aliasing in the same way as a PyTorch state dict.

Handle tied embedding / LM-head weights deliberately.

Document whether:

```text
lm_head.weight
```

is omitted, duplicated, or reconstructed after load.

Then implement:

```bash
python -m tools.inspect_safetensors
```

which must inspect metadata without loading the entire model where possible.

Print:

```text
tensor name
shape
dtype
offset/range if available
numel
estimated bytes
```

---

# 22. SAFETENSORS LOADER

Support both:

```text
pytorch_model.bin
```

and:

```text
model.safetensors
```

through a unified API:

```python
load_checkpoint(
    model,
    checkpoint_path,
)
```

Detect format from extension.

Example:

```python
model = DistilGPT2LMHeadModel(config)

load_checkpoint(
    model,
    "./pytorch_model.bin",
)
```

or:

```python
load_checkpoint(
    model,
    "./model.safetensors",
)
```

---

# 23. STATE-DICT MAPPING

Create a clear state-dict mapping system.

For example:

```python
CHECKPOINT_MAPPING = {
    "transformer.wte.weight": ...,
    "transformer.wpe.weight": ...,
    ...
}
```

But do not manually list every block if a structured mapping loop is clearer.

Map:

```text
transformer.h.0
transformer.h.1
...
```

to:

```text
transformer.blocks.0
transformer.blocks.1
...
```

Keep mapping logic isolated from model implementation.

Include explicit mapping for:

```text
ln_1
attn.c_attn
attn.c_proj
ln_2
mlp.c_fc
mlp.c_proj
```

Validate every checkpoint key.

At the end report:

```text
loaded
transposed
tied
ignored
missing
unexpected
```

Fail loudly if important keys remain unmapped.

Do not silently ignore checkpoint parameters.

---

# 24. CHECKPOINT INSPECTION CLI

Provide:

```bash
python -m tools.inspect_checkpoint
```

Example:

```text
Checkpoint:
  pytorch_model.bin

Format:
  PyTorch state_dict

Tensor count:
  76

Parameter count:
  ...

Size:
  ...

Architecture inference:
  vocab_size: 50257
  hidden_size: 768
  positions: 1024
  layers: 6
  heads: 12
  mlp_size: 3072
```

Then print a hierarchical tree:

```text
transformer
├── wte
│   └── weight [50257,768]
├── wpe
│   └── weight [1024,768]
├── h
│   ├── 0
│   │   ├── ln_1
│   │   ├── attn
│   │   │   ├── c_attn
│   │   │   └── c_proj
│   │   ├── ln_2
│   │   └── mlp
│   └── ...
└── ln_f
```

---

# 25. MODEL TREE

Provide another utility:

```bash
python -m tools.print_model
```

which prints the pure PyTorch module hierarchy:

```text
DistilGPT2LMHeadModel
├── transformer: GPT2Model
│   ├── token_embedding: Embedding
│   ├── position_embedding: Embedding
│   ├── blocks: ModuleList
│   │   ├── 0: TransformerBlock
│   │   │   ├── ln_1
│   │   │   ├── attention
│   │   │   │   ├── qkv_projection
│   │   │   │   └── output_projection
│   │   │   ├── ln_2
│   │   │   └── mlp
│   │   └── ...
│   └── final_layer_norm
└── lm_head
```

Include parameter counts per module.

---

# 26. TENSOR INVENTORY

Implement an important Tensormorph-oriented command:

```bash
python -m tools.tensor_inventory
```

Generate a table containing:

```text
name
module
role
shape
rank
dtype
numel
bytes
requires_grad
shared/tied
```

For example:

```text
transformer.token_embedding.weight
embedding
parameter
[50257,768]
2
float32
38597376
154389504
true
shared-with-lm-head
```

Optionally export:

```text
artifacts/tensor_inventory.json
artifacts/tensor_inventory.csv
```

---

# 27. EXECUTION TRACE

Implement:

```bash
python -m examples.execution_trace
```

Capture the forward execution sequence.

Each trace event should include:

```json
{
  "index": 10,
  "module": "transformer.blocks.0.attention.qkv_projection",
  "module_type": "Linear",
  "input_shapes": [[2, 8, 768]],
  "output_shapes": [[2, 8, 2304]],
  "dtype": "float32",
  "device": "cpu"
}
```

Write:

```text
artifacts/execution_trace.json
```

This file should be simple enough for Tensormorph to consume later.

---

# 28. INTERNAL ATTENTION TRACE

Standard hooks cannot see local intermediate variables.

Therefore add an optional structured debug mode to:

```python
CausalSelfAttention.forward(...)
```

such as:

```python
output, debug = attention(
    hidden_states,
    return_debug=True,
)
```

where debug contains:

```python
{
    "qkv": qkv,
    "query": query,
    "key": key,
    "value": value,
    "attention_scores": attention_scores,
    "causal_mask": causal_mask,
    "attention_probs": attention_probs,
    "context": context,
}
```

Likewise allow the full model to optionally produce structured debug information.

Normal inference should remain:

```python
logits = model(input_ids)
```

Debug inference could be:

```python
result = model(
    input_ids,
    return_debug=True,
)
```

Avoid retaining huge tensors unless debug mode is explicitly enabled.

---

# 29. MODEL OUTPUT TYPES

Use clear dataclasses where useful.

For example:

```python
@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None
    debug: dict[str, Any] | None = None
```

But do not over-engineer the API.

For simple inference, returning a tensor should remain possible.

---

# 30. TOKENIZER

The main model and demo must not depend on a tokenizer.

The deterministic example uses raw token IDs.

However, optionally create:

```text
tokenizer/
```

for a minimal GPT2 BPE tokenizer implementation later.

If tokenizer assets such as:

```text
vocab.json
merges.txt
tokenizer.json
```

are not already present locally, do not make the primary project dependent on downloading them.

The core tests must run offline using fixed token IDs.

---

# 31. OPTIONAL REFERENCE VALIDATION

The primary source tree must have zero dependency on `transformers`.

However, an isolated OPTIONAL reference script may be created:

```text
reference/
└── compare_huggingface.py
```

This script may use:

```python
transformers
```

only to verify numerical parity with Hugging Face.

Nothing under:

```text
src/
```

may import `transformers`.

Nothing required for normal tests may require Transformers.

This distinction is important:

```text
PURE IMPLEMENTATION
src/

OPTIONAL EXTERNAL VALIDATOR
reference/
```

---

# 32. PARITY VALIDATION

If `transformers` is locally installed, compare:

```text
our logits
vs
Hugging Face logits
```

using the exact same:

```text
input_ids
checkpoint
eval mode
dtype
device
```

Report:

```text
max absolute error
mean absolute error
max relative error
allclose result
```

For example:

```python
torch.testing.assert_close(
    ours,
    reference,
    rtol=...,
    atol=...,
)
```

Investigate mismatches rather than increasing tolerance blindly.

Potential mismatch sources include:

- Conv1D weight orientation,
- GELU variant,
- attention scaling,
- causal masking,
- dropout not disabled,
- position IDs,
- LayerNorm epsilon,
- weight tying,
- softmax dimension,
- QKV ordering.

---

# 33. TESTS

Create comprehensive tests.

At minimum:

```text
tests/
├── test_checkpoint_inspection.py
├── test_config_inference.py
├── test_checkpoint_mapping.py
├── test_embeddings.py
├── test_attention.py
├── test_causal_mask.py
├── test_mlp.py
├── test_block.py
├── test_model.py
├── test_lm_head.py
├── test_weight_tying.py
├── test_dataloader.py
├── test_fixed_forward.py
├── test_safetensors.py
├── test_activation_recorder.py
└── test_execution_trace.py
```

Prefer deterministic assertions.

---

# 34. PARTICULAR ATTENTION TESTS

Test causal attention carefully.

Verify that token position:

```text
i
```

cannot attend to:

```text
j > i
```

Inspect the actual attention probability matrix.

For example:

```python
future = attention_probs[..., torch.triu(..., diagonal=1)]
```

and confirm effectively zero probability.

Also validate:

```text
attention_probs.sum(dim=-1) ≈ 1
```

---

# 35. PROJECT STRUCTURE

Create approximately the following structure, adapting it where implementation quality requires:

```text
GPT2PytorchContruct/
├── README.md
├── pyproject.toml
├── pytorch_model.bin
│
├── src/
│   └── gpt2_pytorch/
│       ├── __init__.py
│       │
│       ├── config.py
│       │
│       ├── outputs.py
│       │
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── attention.py
│       │   ├── mlp.py
│       │   ├── block.py
│       │   ├── backbone.py
│       │   └── causal_lm.py
│       │
│       ├── checkpoint/
│       │   ├── __init__.py
│       │   ├── inspect.py
│       │   ├── infer_config.py
│       │   ├── mapping.py
│       │   ├── pytorch_loader.py
│       │   ├── safetensors_loader.py
│       │   └── loader.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fixed_samples.py
│       │   └── dataset.py
│       │
│       ├── tracing/
│       │   ├── __init__.py
│       │   ├── activation_recorder.py
│       │   ├── execution_trace.py
│       │   ├── tensor_stats.py
│       │   └── hooks.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── model_tree.py
│           ├── tensor_inventory.py
│           └── seed.py
│
├── examples/
│   ├── forward_fixed.py
│   ├── dataloader_forward.py
│   ├── forward_hooks.py
│   ├── inspect_attention.py
│   ├── execution_trace.py
│   ├── fx_trace.py
│   ├── export_model.py
│   └── causal_lm_loss.py
│
├── tools/
│   ├── inspect_checkpoint.py
│   ├── print_model.py
│   ├── tensor_inventory.py
│   ├── convert_to_safetensors.py
│   └── inspect_safetensors.py
│
├── reference/
│   └── compare_huggingface.py
│
├── tests/
│   └── ...
│
└── artifacts/
    ├── tensor_inventory.json
    ├── execution_trace.json
    └── export/
```

Keep generated large artifacts ignored where appropriate.

---

# 36. NO UNNECESSARY FRAMEWORKS

The primary runtime should depend on as little as possible.

Prefer:

```text
torch
safetensors
pytest
```

Do not introduce:

```text
transformers
accelerate
datasets
lightning
deepspeed
```

to the primary package.

The optional reference parity tool may declare `transformers` as a development/optional dependency.

---

# 37. PYPROJECT

Use:

```toml
[project]
name = "gpt2-pytorch-construct"
```

Use modern Python.

Target at least:

```text
Python 3.11+
```

Organize optional dependencies such as:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
]

reference = [
    "transformers",
]
```

Do not make Transformers mandatory.

---

# 38. CPU FIRST

The default demo must run on CPU:

```python
device = torch.device("cpu")
```

This is important because the project is intended for inspection and reproducibility.

Support:

```text
cpu
cuda
mps
```

through CLI options if convenient, but do not require GPU hardware.

---

# 39. DETERMINISM

Set deterministic seeds where appropriate:

```python
torch.manual_seed(...)
```

But because the model uses pretrained weights and `eval()`, outputs should already be stable.

Make sure:

```python
model.eval()
```

is used for parity and inspection.

Never let dropout introduce accidental nondeterminism in the main examples.

---

# 40. README

Write a comprehensive `README.md`.

Explain:

## Purpose

This is not another general GPT2 training project.

It is an inspectable reference implementation designed for:

```text
PyTorch
→ state_dict
→ tensors
→ modules
→ operators
→ execution
→ tracing
→ Safetensors
→ TensorGraph
→ Tensormorph
```

## Architecture

Explain:

```text
Token IDs
    ↓
Token Embedding ─────────┐
                         +
Position Embedding ──────┘
    ↓
Transformer Block × 6
    ↓
LayerNorm
    ↓
LM Head
    ↓
Vocabulary Logits
```

Also document the attention flow:

```text
X [B,T,C]
 ↓
QKV Linear
 ↓
[B,T,3C]
 ↓
split
 ├─ Q [B,H,T,D]
 ├─ K [B,H,T,D]
 └─ V [B,H,T,D]
 ↓
QKᵀ / √D
 ↓
[B,H,T,T]
 ↓
causal mask
 ↓
softmax
 ↓
attention @ V
 ↓
[B,H,T,D]
 ↓
merge heads
 ↓
[B,T,C]
```

---

# 41. EXPLAIN EVERY CHECKPOINT TENSOR CATEGORY

The README should explain the semantic meaning of checkpoint names such as:

```text
transformer.wte.weight
transformer.wpe.weight

transformer.h.0.ln_1.weight
transformer.h.0.ln_1.bias

transformer.h.0.attn.c_attn.weight
transformer.h.0.attn.c_attn.bias

transformer.h.0.attn.c_proj.weight
transformer.h.0.attn.c_proj.bias

transformer.h.0.ln_2.weight
transformer.h.0.ln_2.bias

transformer.h.0.mlp.c_fc.weight
transformer.h.0.mlp.c_fc.bias

transformer.h.0.mlp.c_proj.weight
transformer.h.0.mlp.c_proj.bias

transformer.ln_f.weight
transformer.ln_f.bias
```

Explain:

- what each tensor represents,
- where it is used,
- its input dimension,
- output dimension,
- rank,
- expected shape,
- whether it needs transposition when converted to `nn.Linear`.

---

# 42. PARAMETER ACCOUNTING

Add a utility to calculate parameter counts.

Break down approximately:

```text
token embeddings
position embeddings
attention
MLP
LayerNorm
LM head
total
```

Correctly account for tied weights.

Differentiate:

```text
logical parameter references
```

from:

```text
unique underlying parameters
```

when weight tying exists.

---

# 43. TENSOR STORAGE ANALYSIS

Because Tensormorph will inspect models at the tensor/storage level, add storage information where PyTorch safely exposes it.

For tensors report:

```text
shape
stride
storage_offset
is_contiguous
dtype
device
requires_grad
numel
element_size
logical bytes
```

Where applicable also identify:

```text
shared storage
views
tied parameter identity
```

Do not rely on deprecated APIs unnecessarily.

---

# 44. TENSOR IDs

For trace JSON, assign stable hierarchical names rather than relying exclusively on Python memory addresses.

Examples:

```text
input.input_ids

embedding.token.output
embedding.position.output

block.0.ln_1.output

block.0.attention.q
block.0.attention.k
block.0.attention.v
block.0.attention.scores
block.0.attention.probs
block.0.attention.context
block.0.attention.output

block.0.mlp.fc.input
block.0.mlp.fc.output

model.final_hidden
model.logits
```

This naming convention will make later Tensormorph visualization easier.

---

# 45. TRACE JSON FORMAT

Design a simple versioned structure:

```json
{
  "format": "tensormorph-pytorch-trace",
  "version": 1,
  "model": {
    "name": "distilgpt2"
  },
  "inputs": [],
  "parameters": [],
  "events": [],
  "outputs": []
}
```

Each tensor record may contain:

```json
{
  "name": "block.0.attention.q",
  "shape": [2, 12, 8, 64],
  "dtype": "float32",
  "device": "cpu",
  "numel": 12288
}
```

Do not serialize the full tensor values by default.

Optionally support:

```text
--include-values
```

only for small debugging tensors.

---

# 46. SEPARATE PARAMETERS FROM ACTIVATIONS

This distinction is important for Tensormorph.

Parameters:

```text
persistent model state
```

Examples:

```text
embedding weight
Linear weight
Linear bias
LayerNorm weight
LayerNorm bias
```

Activations:

```text
runtime tensors
```

Examples:

```text
hidden_states
query
key
value
attention_scores
attention_probs
MLP activation
logits
```

Trace/export formats should distinguish these explicitly.

---

# 47. SIMPLE TRAINING WORKFLOW

Although this project is primarily for inspection, include one tiny deterministic training/debug example:

```bash
python -m examples.single_train_step
```

It should:

```text
fixed Dataset
→ DataLoader
→ model
→ logits
→ causal LM loss
→ loss.backward()
→ inspect gradients
```

Do NOT attempt to train the full model meaningfully.

The objective is only to expose the complete PyTorch lifecycle:

```text
parameter
→ forward
→ activation
→ loss
→ autograd
→ gradient
```

Print representative gradients from:

```text
embedding
attention projection
MLP
LayerNorm
```

---

# 48. AUTOGRAD INSPECTION

Provide an example demonstrating:

```python
tensor.grad_fn
parameter.grad
requires_grad
is_leaf
```

This will be useful for future Tensormorph autograd graph visualization.

Where practical, inspect the backward graph without relying on third-party graph libraries.

---

# 49. DO NOT OVER-ABSTRACT

Avoid generic frameworks like:

```python
BaseTransformer
BaseAttention
AbstractCheckpointAdapter
GenericModelFactory
```

unless they solve an immediate concrete problem.

This repository should make DistilGPT2 easy to read.

Prefer:

```python
class CausalSelfAttention
class GPT2MLP
class TransformerBlock
class GPT2Model
class DistilGPT2LMHeadModel
```

over unnecessary inheritance.

---

# 50. SOURCE CODE STYLE

Favor verbose intermediate variables.

Prefer:

```python
query_key_scores = torch.matmul(
    query,
    key.transpose(-2, -1),
)

scaled_scores = query_key_scores / math.sqrt(self.head_dim)

masked_scores = scaled_scores.masked_fill(
    causal_mask == 0,
    torch.finfo(scaled_scores.dtype).min,
)

attention_probs = torch.softmax(
    masked_scores,
    dim=-1,
)
```

instead of compressing everything into one expression.

The source code itself should serve as an educational representation of the tensor computation graph.

---

# 51. SHAPE COMMENTS

For important transformations add concise shape comments:

```python
# [B, T, C] -> [B, T, 3C]
qkv = self.qkv_projection(hidden_states)

# Three tensors, each [B, T, C]
query, key, value = qkv.chunk(3, dim=-1)

# [B, T, C] -> [B, H, T, D]
query = self._split_heads(query)
```

Do this especially for:

- embeddings,
- attention,
- head splitting,
- attention matrices,
- MLP,
- logits.

---

# 52. VERIFY, DO NOT JUST IMPLEMENT

Once implementation is complete:

1. inspect checkpoint,
2. infer configuration,
3. instantiate model,
4. load every parameter,
5. verify no unexpected mapping gaps,
6. run deterministic forward,
7. run tests,
8. inspect logits,
9. convert checkpoint to Safetensors,
10. reload Safetensors,
11. confirm `.bin` and Safetensors outputs match,
12. run optional HF parity if Transformers is available,
13. run FX/export demonstrations,
14. generate trace JSON,
15. generate tensor inventory.

Do not declare success merely because the Python modules import.

---

# 53. CHECKPOINT ROUND-TRIP TEST

Test:

```text
pytorch_model.bin
        ↓
pure PyTorch model
        ↓
fixed logits A
```

Then:

```text
pytorch_model.bin
        ↓
Safetensors conversion
        ↓
pure PyTorch model
        ↓
fixed logits B
```

Require:

```python
torch.testing.assert_close(logits_a, logits_b)
```

within appropriate numerical tolerance.

---

# 54. FINAL COMMAND SET

The repository should ultimately make commands like these work:

```bash
cd /Users/thanh/Tensormorph/GPT2PytorchContruct

python -m tools.inspect_checkpoint

python -m tools.print_model

python -m tools.tensor_inventory

python -m examples.forward_fixed

python -m examples.dataloader_forward

python -m examples.inspect_attention

python -m examples.forward_hooks

python -m examples.execution_trace

python -m examples.fx_trace

python -m examples.export_model

python -m examples.causal_lm_loss

python -m examples.single_train_step

python -m tools.convert_to_safetensors

python -m tools.inspect_safetensors

pytest
```

---

# 55. IMPORTANT IMPLEMENTATION BOUNDARIES

The following are mandatory:

### Core implementation

Must use:

```text
PyTorch
```

Must NOT use Hugging Face model classes.

### Checkpoint

Primary source:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct/pytorch_model.bin
```

Do not redownload it.

### Input

Primary demos use deterministic fixed token IDs.

Do not require tokenizer downloads.

### Attention

Must expose Q/K/V and attention matrices.

Do not hide primary implementation behind `MultiheadAttention`, FlashAttention, or SDPA.

### Checkpoint mapping

Must explicitly handle Hugging Face GPT2 Conv1D weight orientation.

### Safetensors

Must support inspection, conversion, loading, and output parity.

### Model hierarchy

Must be decomposed across meaningful source files and `nn.Module` objects.

### Tracing

Must support module hooks plus internal attention traces.

### Tensormorph

Design names and trace structures so they can later map naturally into:

```text
ITensor
IModel
TensorGraph
Tensor IR
TensorSession
TensorView
```

but do NOT add dependencies on Tensormorph itself to this sample project.

---

# 56. FIRST ACTION

Begin by inspecting:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct
```

and especially:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct/pytorch_model.bin
```

Print the checkpoint structure before designing the final model classes.

Use evidence from the checkpoint to determine the exact architecture.

Then create the project incrementally:

```text
checkpoint inspection
→ config inference
→ module architecture
→ checkpoint mapping
→ fixed DataLoader
→ deterministic forward
→ activation tracing
→ Safetensors
→ FX/export
→ tests
→ documentation
```

Do not modify or remove the original:

```text
pytorch_model.bin
```

Treat it as the authoritative reference checkpoint.

---

# 57. FINAL DELIVERABLE

At completion, provide a concise engineering report containing:

```text
1. Files created
2. Architecture discovered
3. Number of layers
4. Hidden size
5. Attention heads
6. Vocabulary size
7. Parameter count
8. Checkpoint tensor count
9. Conv1D → Linear transformations performed
10. Weight tying behavior
11. Fixed input shape
12. Forward output shape
13. Safetensors conversion result
14. .bin ↔ Safetensors numerical parity
15. Hugging Face parity result, if reference dependency is available
16. FX tracing status
17. torch.export status
18. Test results
19. Generated Tensormorph-oriented trace artifacts
20. Any remaining limitations
```

The completed repository should function as a small but rigorous **PyTorch model laboratory** for Tensormorph rather than merely another GPT2 inference script.