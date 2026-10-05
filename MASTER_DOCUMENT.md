# GPT2PytorchContruct

## 1. Project Goal

`GPT2PytorchContruct` is a small, deterministic, pure-PyTorch reference project for studying and debugging the complete execution lifecycle of a GPT-2-family model.

The project reconstructs **DistilGPT2** directly from an existing Hugging Face checkpoint without using Hugging Face `transformers` for the model implementation.

Primary checkpoint:

```text
/Users/thanh/Tensormorph/GPT2PytorchContruct/pytorch_model.bin
```

Model source:

```text
distilbert/distilgpt2
```

The project is intended primarily as a controlled reference model for **Tensormorph** development.

---

# 2. Primary Targets

The project must provide a simple environment for inspecting:

```text
Checkpoint
→ Parameters
→ nn.Module hierarchy
→ Dataset
→ DataLoader
→ Forward pass
→ Intermediate tensors
→ Attention
→ Logits
→ Loss
→ Autograd
→ Gradients
→ Execution trace
```

It should also serve as a reference implementation for future Tensormorph support for:

- PyTorch model inspection
- tensor visualization
- parameter inspection
- Safetensors inspection
- TensorGraph construction
- Tensor IR
- execution tracing
- activation tracing
- model debugging
- checkpoint analysis

---

# 3. Design Principles

The project follows several strict principles.

## Pure PyTorch

The model architecture must use ordinary PyTorch components:

```python
torch
torch.nn.Module
torch.nn.Linear
torch.nn.Embedding
torch.nn.LayerNorm
torch.nn.Dropout
torch.nn.ModuleList
```

The core implementation must not depend on:

```text
transformers
accelerate
datasets
lightning
deepspeed
```

Hugging Face may only be used by an optional reference validation script.

---

## Explicit Tensor Flow

Tensor transformations should be visible in source code.

Prefer:

```text
[B, T, C]
→ QKV
→ Q / K / V
→ [B, H, T, D]
→ Attention Scores
→ Attention Probabilities
→ Context
→ [B, T, C]
```

over hiding computation behind high-level abstractions.

The reference attention path must not use:

```text
nn.MultiheadAttention
FlashAttention
scaled_dot_product_attention
```

as the primary implementation.

---

## Deterministic

The primary examples use fixed token tensors.

No network, tokenizer, dataset download, or random data should be required.

Example:

```python
input_ids = torch.tensor(
    [
        [15496, 11, 616, 1438, 318, 1332, 13, 198],
        [1212, 318, 257, 1332, 6827, 284, 1332, 13],
    ],
    dtype=torch.long,
)
```

Default execution should use:

```python
model.eval()
torch.no_grad()
```

for reproducible inference.

---

# 4. Model Target

The project reconstructs the DistilGPT2 architecture represented by the downloaded checkpoint.

Expected configuration:

```text
Vocabulary:       50,257
Hidden size:      768
Transformer:      6 blocks
Attention heads:  12
Head dimension:   64
MLP dimension:    3,072
Max positions:    1,024
```

These values must be verified from the checkpoint rather than assumed blindly.

---

# 5. Architecture

The model hierarchy should approximately be:

```text
DistilGPT2LMHeadModel
│
├── GPT2Model
│   │
│   ├── TokenEmbedding
│   ├── PositionEmbedding
│   ├── Dropout
│   │
│   ├── TransformerBlock × 6
│   │   ├── LayerNorm
│   │   ├── CausalSelfAttention
│   │   │   ├── QKV Linear
│   │   │   └── Output Linear
│   │   │
│   │   ├── LayerNorm
│   │   └── GPT2MLP
│   │       ├── Linear 768 → 3072
│   │       ├── GELU
│   │       └── Linear 3072 → 768
│   │
│   └── Final LayerNorm
│
└── LM Head
```

Forward path:

```text
input_ids
    ↓
Token Embedding
    +
Position Embedding
    ↓
Transformer Block × 6
    ↓
Final LayerNorm
    ↓
LM Head
    ↓
logits [B, T, 50257]
```

---

# 6. Attention Reference Flow

Attention must remain inspectable.

```text
hidden_states
[B, T, 768]

        ↓ QKV projection

[B, T, 2304]

        ↓ split

Q [B, T, 768]
K [B, T, 768]
V [B, T, 768]

        ↓ reshape

Q [B, 12, T, 64]
K [B, 12, T, 64]
V [B, 12, T, 64]

        ↓

Q × Kᵀ / sqrt(64)

        ↓

attention_scores
[B, 12, T, T]

        ↓ causal mask
        ↓ softmax

attention_probs
[B, 12, T, T]

        ↓ × V

context
[B, 12, T, 64]

        ↓ merge heads

[B, T, 768]

        ↓ output projection

[B, T, 768]
```

Important intermediate tensors should be available to tracing/debugging systems.

---

# 7. Checkpoint Handling

The existing checkpoint is authoritative:

```text
pytorch_model.bin
```

The original file must never be modified.

The project must inspect:

- state-dict keys
- shapes
- dtype
- tensor count
- parameter count
- storage size
- module hierarchy

A major compatibility concern is Hugging Face GPT2's historical `Conv1D` representation.

Example checkpoint weight:

```text
transformer.h.0.attn.c_attn.weight
[768, 2304]
```

A PyTorch equivalent:

```python
nn.Linear(768, 2304)
```

stores:

```text
[2304, 768]
```

Therefore affected weights must be transposed when loading.

Typical affected modules:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
mlp.c_proj
```

Checkpoint conversion logic must remain centralized and separate from model code.

---

# 8. Safetensors

The project must support both:

```text
pytorch_model.bin
```

and:

```text
model.safetensors
```

Required operations:

```text
inspect .bin
convert .bin → safetensors
inspect safetensors
load safetensors
compare outputs
```

The following must hold:

```text
pytorch_model.bin
        ↓
      model
        ↓
    logits A
```

and:

```text
pytorch_model.bin
        ↓
   safetensors
        ↓
      model
        ↓
    logits B
```

with:

```python
torch.testing.assert_close(logits_a, logits_b)
```

---

# 9. Weight Tying

GPT2 normally shares the token embedding weights with the language-model output projection.

The implementation should preserve true sharing where confirmed by the checkpoint/model architecture:

```python
model.lm_head.weight = model.transformer.token_embedding.weight
```

The project should distinguish:

```text
logical parameter references
```

from:

```text
unique underlying parameter storage
```

This is important for Tensormorph tensor/storage inspection.

---

# 10. Data Pipeline

The project includes a small deterministic PyTorch data pipeline:

```text
FixedTokenDataset
        ↓
DataLoader
        ↓
input_ids
        ↓
DistilGPT2
```

The purpose is not dataset training.

The purpose is to provide a minimal real PyTorch workflow containing:

```text
Dataset
DataLoader
Batch
Model
Forward
Loss
Backward
Gradient
```

---

# 11. Training Reference

Provide one small debugging training step:

```text
Fixed batch
    ↓
Forward
    ↓
Causal LM loss
    ↓
Backward
    ↓
Gradient inspection
```

This is not intended to fine-tune DistilGPT2.

Its purpose is to expose:

```text
Parameter
→ Activation
→ Loss
→ Autograd
→ Gradient
```

for Tensormorph inspection.

---

# 12. Tracing

The project should expose three levels of tracing.

## Module Trace

Use PyTorch hooks:

```python
register_forward_pre_hook()
register_forward_hook()
```

Capture:

```text
module
module type
input shape
output shape
dtype
device
execution order
```

---

## Internal Tensor Trace

Attention contains important tensors that normal hooks cannot capture.

Expose optional debug tensors such as:

```text
query
key
value
attention_scores
causal_mask
attention_probs
context
```

---

## Graph Trace

Experiment with:

```python
torch.fx.symbolic_trace
```

and:

```python
torch.export.export
```

These provide reference material for future TensorGraph and Tensor IR integration.

---

# 13. Tensormorph Trace Format

Generate a simple versioned trace artifact.

Example:

```json
{
  "format": "tensormorph-pytorch-trace",
  "version": 1,
  "model": {
    "name": "distilgpt2"
  },
  "parameters": [],
  "inputs": [],
  "events": [],
  "outputs": []
}
```

Use stable logical tensor names such as:

```text
input.input_ids

embedding.token.output
embedding.position.output

block.0.attention.query
block.0.attention.key
block.0.attention.value
block.0.attention.scores
block.0.attention.probs
block.0.attention.context

block.0.mlp.output

model.final_hidden
model.logits
```

Do not use Python object addresses as primary identifiers.

---

# 14. Tensor Inventory

The project should generate an inventory for model parameters and optionally activations.

Required fields:

```text
name
role
module
shape
rank
dtype
device
numel
bytes
stride
requires_grad
shared/tied
```

Parameters and runtime activations must be represented separately.

---

# 15. Project Structure

Target structure:

```text
GPT2PytorchContruct/
├── README.md
├── MASTER.md
├── pyproject.toml
├── pytorch_model.bin
│
├── src/
│   └── gpt2_pytorch/
│       ├── config.py
│       ├── outputs.py
│       │
│       ├── model/
│       │   ├── embeddings.py
│       │   ├── attention.py
│       │   ├── mlp.py
│       │   ├── block.py
│       │   ├── backbone.py
│       │   └── causal_lm.py
│       │
│       ├── checkpoint/
│       │   ├── inspect.py
│       │   ├── infer_config.py
│       │   ├── mapping.py
│       │   └── loader.py
│       │
│       ├── data/
│       │   ├── fixed_samples.py
│       │   └── dataset.py
│       │
│       ├── tracing/
│       │   ├── activation_recorder.py
│       │   ├── execution_trace.py
│       │   └── tensor_stats.py
│       │
│       └── utils/
│           ├── model_tree.py
│           └── tensor_inventory.py
│
├── examples/
│   ├── forward_fixed.py
│   ├── dataloader_forward.py
│   ├── inspect_attention.py
│   ├── execution_trace.py
│   ├── fx_trace.py
│   ├── export_model.py
│   └── single_train_step.py
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
│
└── artifacts/
```

---

# 16. Runtime Dependencies

Primary dependencies:

```text
torch
safetensors
```

Development:

```text
pytest
```

Optional reference validation:

```text
transformers
```

`transformers` must not be imported anywhere under the core `src/` implementation.

---

# 17. Required Validation

The project is considered correct only when the following pipeline passes:

```text
Inspect checkpoint
        ↓
Infer architecture
        ↓
Construct pure PyTorch model
        ↓
Map checkpoint tensors
        ↓
Load all required weights
        ↓
Fixed deterministic forward
        ↓
Inspect intermediate tensors
        ↓
Convert to Safetensors
        ↓
Reload Safetensors
        ↓
Verify output parity
        ↓
Run tracing/export
        ↓
Run tests
```

Optional Hugging Face validation should compare final logits against the original implementation.

---

# 18. Main Commands

The repository should support commands similar to:

```bash
python -m tools.inspect_checkpoint
python -m tools.print_model
python -m tools.tensor_inventory

python -m examples.forward_fixed
python -m examples.dataloader_forward
python -m examples.inspect_attention
python -m examples.execution_trace
python -m examples.fx_trace
python -m examples.export_model
python -m examples.single_train_step

python -m tools.convert_to_safetensors
python -m tools.inspect_safetensors

pytest
```

---

# 19. Non-Goals

This project is not intended to become:

- a replacement for Hugging Face Transformers,
- a general transformer framework,
- a production inference engine,
- a distributed training framework,
- a tokenizer library,
- a model-serving platform,
- a high-performance FlashAttention implementation.

Its value is simplicity and observability.

---

# 20. Relationship to Tensormorph

`GPT2PytorchContruct` acts as a controlled reference workload for Tensormorph.

Conceptually:

```text
GPT2PytorchContruct
        │
        ├── PyTorch Modules
        ├── Parameters
        ├── Activations
        ├── Safetensors
        ├── Execution Trace
        ├── FX Graph
        └── ExportedProgram
                │
                ▼
           Tensormorph
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     ITensor  IModel  TensorGraph
        │                │
        ▼                ▼
    TensorView        Tensor IR
```

The project itself should remain independent of Tensormorph libraries so it can serve as an external reference and compatibility test.

---

# 21. Definition of Done

The first project milestone is complete when:

- DistilGPT2 runs entirely through custom `nn.Module` code.
- `pytorch_model.bin` loads without relying on Transformers.
- Conv1D-style checkpoint weights are mapped correctly.
- fixed DataLoader input produces deterministic logits.
- Q/K/V and attention tensors can be inspected.
- parameter and activation inventories can be generated.
- a single backward pass exposes gradients.
- checkpoint conversion to Safetensors works.
- `.bin` and Safetensors outputs match.
- execution traces can be exported.
- FX and `torch.export` behavior is documented.
- tests validate the critical model components.

The final result should be a small, transparent **PyTorch model laboratory** that provides Tensormorph with a reliable reference for understanding how real PyTorch models move from checkpoint storage to modules, tensors, execution graphs, activations, and gradients.