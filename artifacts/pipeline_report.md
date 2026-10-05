# DistilGPT2 pipeline report

Generated 2026-10-05 18:12:07 on macOS-26.6.2-arm64-arm-64bit-Mach-O | Python 3.13.15 | torch 2.14.1 | device cpu

**17/17 steps passed.**

## Summary

| # | Step | Command | Result | Time | Produces |
|--:|---|---|---|--:|---|
| 1 | Inspect checkpoint | `python -m tools.inspect_checkpoint` | PASS | 0.8s | 83 tensors, tying, inferred architecture, tree |
| 2 | Print model | `python -m tools.print_model` | PASS | 1.1s | module tree, parameter counts, logical vs unique accounting |
| 3 | Tensor inventory | `python -m tools.tensor_inventory` | PASS | 1.9s | every parameter/buffer -> artifacts/tensor_inventory.{json,csv} |
| 4 | Fixed forward | `python -m examples.forward_fixed` | PASS | 1.7s | deterministic eval forward, intermediate shapes, logits stats |
| 5 | DataLoader forward | `python -m examples.dataloader_forward` | PASS | 1.8s | Dataset -> DataLoader -> batch -> logits |
| 6 | Inspect attention | `python -m examples.inspect_attention` | PASS | 1.8s | Q/K/V, mask, probabilities, causality |
| 7 | Forward hooks | `python -m examples.forward_hooks` | PASS | 1.8s | pre/post hook execution order |
| 8 | Execution trace | `python -m examples.execution_trace` | PASS | 1.7s | artifacts/execution_trace.json |
| 9 | Causal LM loss | `python -m examples.causal_lm_loss` | PASS | 1.8s | shift_logits, shift_labels, loss |
| 10 | Single train step | `python -m examples.single_train_step` | PASS | 1.8s | backward, gradients, one SGD step |
| 11 | Autograd inspect | `python -m examples.autograd_inspect` | PASS | 1.8s | grad_fn, is_leaf, backward graph |
| 12 | Convert to Safetensors | `python -m tools.convert_to_safetensors` | PASS | 1.6s | weights/pytorch_model.bin -> artifacts/model.safetensors |
| 13 | Inspect Safetensors | `python -m tools.inspect_safetensors` | PASS | 0.6s | header-only inspection |
| 14 | Hugging Face parity | `python -m reference.compare_huggingface` | PASS | 5.1s | optional; needs transformers |
| 15 | FX trace | `python -m examples.fx_trace` | PASS | 1.9s | artifacts/fx_graph.{txt,json} |
| 16 | torch.export | `python -m examples.export_model` | PASS | 2.4s | artifacts/export/ |
| 17 | Tests | `pytest` | PASS | 9.6s | full pytest suite |

## Key results

- Architecture: 6 layers, hidden 768, 12 heads (head_dim 64), MLP 3072, vocab 50257, 1024 positions
- Parameters: 81,912,576 unique / 120,509,952 logical (tied lm_head)
- Load report (.bin): loaded 76, transposed 24, tied 1, ignored 6, missing 0, unexpected 0
- Load report (.safetensors): ['lm_head.weight -> transformer.embeddings.token_embedding.weight (reconstructed)']
- Fixed input [2, 8] -> logits [2, 8, 50257]; logits[0,-1,:3] = [-54.2574, -47.8187, -50.4389]
- `.bin` vs Safetensors logits bit-identical: **True**

## Step output

### 1. Inspect checkpoint

`python -m tools.inspect_checkpoint` | exit 0 | 0.8s

```text
Checkpoint:
  pytorch_model.bin
Format:
  PyTorch state_dict (pickle, weights_only)
Tensor count:
  83
Scalar count:
  126,801,408 logical, 88,204,032 unique storage
Size:
  336.47 MiB unique (352,816,128 bytes); on-disk 352,833,716 bytes
Dtypes:
  ['torch.float32']
Tied (shared storage):
  {'transformer.wte.weight': ['lm_head.weight'], 'lm_head.weight': ['transformer.wte.weight']}
Architecture inference:
  vocab_size: 50257
  hidden_size: 768
  positions: 1024
  layers: 6
  heads: 12 (head_dim 64; heads are assumed, not in any shape)
  mlp_size: 3072
Tensors:
transformer.wte.weight
  shape = [50257, 768]
  dtype = torch.float32
transformer.wpe.weight
  shape = [1024, 768]
  dtype = torch.float32
transformer.h.0.ln_1.weight
  shape = [768]
  dtype = torch.float32
transformer.h.0.ln_1.bias
  shape = [768]
  dtype = torch.float32
transformer.h.0.attn.bias
  shape = [1, 1, 1024, 1024]
  dtype = torch.float32
transformer.h.0.attn.c_attn.weight
  shape = [768, 2304]
  dtype = torch.float32
transformer.h.0.attn.c_attn.bias
  shape = [2304]
  dtype = torch.float32
transformer.h.0.attn.c_proj.weight
  shape = [768, 768]
... [55 lines omitted, full log in artifacts/logs/] ...
│   │   │       ├── c_fc
│   │   │       │   ├── weight [768,3072]
│   │   │       │   └── bias [3072]
│   │   │       └── c_proj
│   │   │           ├── weight [3072,768]
│   │   │           └── bias [768]
│   │   └── 1..5: same structure as 0
│   └── ln_f
│       ├── weight [768]
│       └── bias [768]
└── lm_head
    └── weight [50257,768]
```

### 2. Print model

`python -m tools.print_model` | exit 0 | 1.1s

```text
DistilGPT2LMHeadModel  [81,912,576 params]
├── transformer: GPT2Model  [81,912,576 params]
│   ├── embeddings: GPT2Embeddings  [39,383,808 params]
│   │   ├── token_embedding: TokenEmbedding  [38,597,376 params]  (tied)
│   │   ├── position_embedding: PositionEmbedding  [786,432 params]
│   │   └── dropout: Dropout
│   ├── blocks: ModuleList  [42,527,232 params]
│   │   ├── 0: TransformerBlock  [7,087,872 params]
│   │   │   ├── ln_1: LayerNorm  [1,536 params]
│   │   │   ├── attention: CausalSelfAttention  [2,362,368 params]
│   │   │   │   ├── qkv_projection: Linear  [1,771,776 params]
│   │   │   │   ├── output_projection: Linear  [590,592 params]
│   │   │   │   ├── attention_dropout: Dropout
│   │   │   │   └── residual_dropout: Dropout
│   │   │   ├── ln_2: LayerNorm  [1,536 params]
│   │   │   └── mlp: GPT2MLP  [4,722,432 params]
│   │   │       ├── fc: Linear  [2,362,368 params]
│   │   │       ├── activation: GPT2GELU
│   │   │       ├── projection: Linear  [2,360,064 params]
│   │   │       └── dropout: Dropout
│   │   └── 1..5: same structure as 0
│   └── final_layer_norm: LayerNorm  [1,536 params]
└── lm_head: Linear  [38,597,376 params]  (tied)

category              logical        unique
token embeddings    38,597,376    38,597,376
position embeddings     786,432       786,432
layernorm               19,968        19,968
attention           14,174,208    14,174,208
mlp                 28,334,592    28,334,592
lm head             38,597,376             0
total              120,509,952    81,912,576
```

### 3. Tensor inventory

`python -m tools.tensor_inventory` | exit 0 | 1.9s

```text
name                                                          role      shape           dtype          numel       bytes  shared
transformer.embeddings.token_embedding.weight                 parameter [50257,768]     float32   38,597,376 154,389,504  shared-with-lm_head.weight
transformer.embeddings.position_embedding.weight              parameter [1024,768]      float32      786,432   3,145,728  no
transformer.blocks.0.ln_1.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.0.ln_1.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.0.attention.qkv_projection.weight          parameter [2304,768]      float32    1,769,472   7,077,888  no
transformer.blocks.0.attention.qkv_projection.bias            parameter [2304]          float32        2,304       9,216  no
transformer.blocks.0.attention.output_projection.weight       parameter [768,768]       float32      589,824   2,359,296  no
transformer.blocks.0.attention.output_projection.bias         parameter [768]           float32          768       3,072  no
transformer.blocks.0.ln_2.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.0.ln_2.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.0.mlp.fc.weight                            parameter [3072,768]      float32    2,359,296   9,437,184  no
transformer.blocks.0.mlp.fc.bias                              parameter [3072]          float32        3,072      12,288  no
transformer.blocks.0.mlp.projection.weight                    parameter [768,3072]      float32    2,359,296   9,437,184  no
transformer.blocks.0.mlp.projection.bias                      parameter [768]           float32          768       3,072  no
transformer.blocks.1.ln_1.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.1.ln_1.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.1.attention.qkv_projection.weight          parameter [2304,768]      float32    1,769,472   7,077,888  no
transformer.blocks.1.attention.qkv_projection.bias            parameter [2304]          float32        2,304       9,216  no
transformer.blocks.1.attention.output_projection.weight       parameter [768,768]       float32      589,824   2,359,296  no
transformer.blocks.1.attention.output_projection.bias         parameter [768]           float32          768       3,072  no
transformer.blocks.1.ln_2.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.1.ln_2.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.1.mlp.fc.weight                            parameter [3072,768]      float32    2,359,296   9,437,184  no
transformer.blocks.1.mlp.fc.bias                              parameter [3072]          float32        3,072      12,288  no
transformer.blocks.1.mlp.projection.weight                    parameter [768,3072]      float32    2,359,296   9,437,184  no
transformer.blocks.1.mlp.projection.bias                      parameter [768]           float32          768       3,072  no
transformer.blocks.2.ln_1.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.2.ln_1.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.2.attention.qkv_projection.weight          parameter [2304,768]      float32    1,769,472   7,077,888  no
transformer.blocks.2.attention.qkv_projection.bias            parameter [2304]          float32        2,304       9,216  no
transformer.blocks.2.attention.output_projection.weight       parameter [768,768]       float32      589,824   2,359,296  no
transformer.blocks.2.attention.output_projection.bias         parameter [768]           float32          768       3,072  no
transformer.blocks.2.ln_2.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.2.ln_2.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.2.mlp.fc.weight                            parameter [3072,768]      float32    2,359,296   9,437,184  no
transformer.blocks.2.mlp.fc.bias                              parameter [3072]          float32        3,072      12,288  no
transformer.blocks.2.mlp.projection.weight                    parameter [768,3072]      float32    2,359,296   9,437,184  no
transformer.blocks.2.mlp.projection.bias                      parameter [768]           float32          768       3,072  no
transformer.blocks.3.ln_1.weight                              parameter [768]           float32          768       3,072  no
transformer.blocks.3.ln_1.bias                                parameter [768]           float32          768       3,072  no
transformer.blocks.3.attention.qkv_projection.weight          parameter [2304,768]      float32    1,769,472   7,077,888  no
transformer.blocks.3.attention.qkv_projection.bias            parameter [2304]          float32        2,304       9,216  no
transformer.blocks.3.attention.output_projection.weight       parameter [768,768]       float32      589,824   2,359,296  no
transformer.blocks.3.attention.output_projection.bias         parameter [768]           float32          768       3,072  no
... [30 lines omitted, full log in artifacts/logs/] ...
transformer.final_layer_norm.weight                           parameter [768]           float32          768       3,072  no
transformer.final_layer_norm.bias                             parameter [768]           float32          768       3,072  no
lm_head.weight                                                parameter [50257,768]     float32   38,597,376 154,389,504  shared-with-transformer.embeddings.token_embedding.weight
transformer.embeddings.position_ids                           buffer    [1024]          int64          1,024       8,192  no
transformer.blocks.0.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no
transformer.blocks.1.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no
transformer.blocks.2.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no
transformer.blocks.3.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no
transformer.blocks.4.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no
transformer.blocks.5.attention.causal_mask                    buffer    [1024,1024]     bool       1,048,576   1,048,576  no

wrote artifacts/tensor_inventory.json and .csv (84 rows)
```

### 4. Fixed forward

`python -m examples.forward_fixed` | exit 0 | 1.7s

```text
loaded: 76  transposed: 24  tied: 1  ignored: 6

input.input_ids                   [2, 8]
embedding.token.output            [2, 8, 768]
embedding.position.output         [1, 8, 768]
embedding.output                  [2, 8, 768]
block.0.input                     [2, 8, 768]
block.0.ln_1.output               [2, 8, 768]
block.0.attention.qkv             [2, 8, 2304]
block.0.attention.q               [2, 12, 8, 64]
block.0.attention.k               [2, 12, 8, 64]
block.0.attention.v               [2, 12, 8, 64]
block.0.attention.scores          [2, 12, 8, 8]
block.0.attention.mask            [8, 8]
block.0.attention.probs           [2, 12, 8, 8]
block.0.attention.context         [2, 12, 8, 64]
block.0.attention.output          [2, 8, 768]
block.0.attention_residual.output [2, 8, 768]
block.0.ln_2.output               [2, 8, 768]
block.0.mlp.fc.output             [2, 8, 3072]
block.0.mlp.activation.output     [2, 8, 3072]
block.0.mlp.output                [2, 8, 768]
block.0.output                    [2, 8, 768]
block.5.output                    [2, 8, 768]
model.final_hidden                [2, 8, 768]
model.logits                      [2, 8, 50257]

input_ids:
tensor([[15496,    11,   616,  1438,   318,  1332,    13,   198],
        [ 1212,   318,   257,  1332,  6827,   284,  1332,    13]])

logits[0, -1, :5] = [-54.2574462890625, -47.81869888305664, -50.43892288208008, -51.153560638427734, -53.93793487548828]
logits[1, 0, :5]  = [-31.92154312133789, -29.890575408935547, -31.864479064941406, -31.4849853515625, -32.15761184692383]
argmax next token per position:
tensor([[ 383,  290, 1438,  318,  449,   88,  198,  198],
        [ 383,  257,  845,  286,   13,  766,  262,  198]])
logits stats: min=-91.7142 max=-27.6808 mean=-63.3081 std=11.7353
```

### 5. DataLoader forward

`python -m examples.dataloader_forward` | exit 0 | 1.8s

```text
dataset: sample_batch, 2 rows, 1 batches, shuffle=False
batch 0: input_ids [2, 8] labels [2, 8] -> logits [2, 8, 50257]
  logits[:, -1, :3] = [[-54.2574462890625, -47.81869888305664, -50.43892288208008], [-73.98898315429688, -73.70838165283203, -72.94536590576172]]
```

### 6. Inspect attention

`python -m examples.inspect_attention` | exit 0 | 1.8s

```text
qkv               [1, 8, 2304]      torch.float32
query             [1, 12, 8, 64]    torch.float32
key               [1, 12, 8, 64]    torch.float32
value             [1, 12, 8, 64]    torch.float32
attention_scores  [1, 12, 8, 8]     torch.float32
causal_mask       [8, 8]            torch.bool
attention_probs   [1, 12, 8, 8]     torch.float32
context           [1, 12, 8, 64]    torch.float32

causal_mask [T, T] (True = may attend):
tensor([[1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)

attention_probs[0, head 0] (row i = query token i):
tensor([[1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.827, 0.173, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.488, 0.096, 0.415, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.323, 0.131, 0.446, 0.100, 0.000, 0.000, 0.000, 0.000],
        [0.222, 0.085, 0.045, 0.518, 0.131, 0.000, 0.000, 0.000],
        [0.253, 0.113, 0.166, 0.301, 0.135, 0.031, 0.000, 0.000],
        [0.290, 0.128, 0.203, 0.158, 0.130, 0.051, 0.040, 0.000],
        [0.213, 0.104, 0.088, 0.153, 0.083, 0.086, 0.103, 0.170]])

row sums: min=1.000000 max=1.000000
max probability on future positions (j > i): 0.0
most attended key per query (head 0): [0, 0, 0, 2, 3, 3, 0, 0]
```

### 7. Forward hooks

`python -m examples.forward_hooks` | exit 0 | 1.8s

```text
entry order (pre-hook):
000 DistilGPT2LMHeadModel <root>                                                    [2, 8] -> [2, 8, 50257]
001   GPT2Model             transformer                                             [2, 8] -> [2, 8, 768]
002     GPT2Embeddings        transformer.embeddings                                [2, 8] -> [2, 8, 768]
003       TokenEmbedding        transformer.embeddings.token_embedding              [2, 8] -> [2, 8, 768]
004       PositionEmbedding     transformer.embeddings.position_embedding           [1, 8] -> [1, 8, 768]
005       Dropout               transformer.embeddings.dropout                      [2, 8, 768] -> [2, 8, 768]
006     TransformerBlock      transformer.blocks.0                                  [2, 8, 768] -> [2, 8, 768]
007       LayerNorm             transformer.blocks.0.ln_1                           [2, 8, 768] -> [2, 8, 768]
008       CausalSelfAttention   transformer.blocks.0.attention                      [2, 8, 768] -> [2, 8, 768]
009         Linear                transformer.blocks.0.attention.qkv_projection     [2, 8, 768] -> [2, 8, 2304]
010         Dropout               transformer.blocks.0.attention.attention_dropout  [2, 12, 8, 8] -> [2, 12, 8, 8]
011         Linear                transformer.blocks.0.attention.output_projection  [2, 8, 768] -> [2, 8, 768]
012         Dropout               transformer.blocks.0.attention.residual_dropout   [2, 8, 768] -> [2, 8, 768]
013       LayerNorm             transformer.blocks.0.ln_2                           [2, 8, 768] -> [2, 8, 768]
014       GPT2MLP               transformer.blocks.0.mlp                            [2, 8, 768] -> [2, 8, 768]
015         Linear                transformer.blocks.0.mlp.fc                       [2, 8, 768] -> [2, 8, 3072]
016         GPT2GELU              transformer.blocks.0.mlp.activation               [2, 8, 3072] -> [2, 8, 3072]
017         Linear                transformer.blocks.0.mlp.projection               [2, 8, 3072] -> [2, 8, 768]
018         Dropout               transformer.blocks.0.mlp.dropout                  [2, 8, 768] -> [2, 8, 768]
019     TransformerBlock      transformer.blocks.1                                  [2, 8, 768] -> [2, 8, 768]
020       LayerNorm             transformer.blocks.1.ln_1                           [2, 8, 768] -> [2, 8, 768]
021       CausalSelfAttention   transformer.blocks.1.attention                      [2, 8, 768] -> [2, 8, 768]
022         Linear                transformer.blocks.1.attention.qkv_projection     [2, 8, 768] -> [2, 8, 2304]
023         Dropout               transformer.blocks.1.attention.attention_dropout  [2, 12, 8, 8] -> [2, 12, 8, 8]
024         Linear                transformer.blocks.1.attention.output_projection  [2, 8, 768] -> [2, 8, 768]
025         Dropout               transformer.blocks.1.attention.residual_dropout   [2, 8, 768] -> [2, 8, 768]
026       LayerNorm             transformer.blocks.1.ln_2                           [2, 8, 768] -> [2, 8, 768]
027       GPT2MLP               transformer.blocks.1.mlp                            [2, 8, 768] -> [2, 8, 768]
028         Linear                transformer.blocks.1.mlp.fc                       [2, 8, 768] -> [2, 8, 3072]
029         GPT2GELU              transformer.blocks.1.mlp.activation               [2, 8, 3072] -> [2, 8, 3072]
030         Linear                transformer.blocks.1.mlp.projection               [2, 8, 3072] -> [2, 8, 768]
031         Dropout               transformer.blocks.1.mlp.dropout                  [2, 8, 768] -> [2, 8, 768]
032     TransformerBlock      transformer.blocks.2                                  [2, 8, 768] -> [2, 8, 768]
033       LayerNorm             transformer.blocks.2.ln_1                           [2, 8, 768] -> [2, 8, 768]
034       CausalSelfAttention   transformer.blocks.2.attention                      [2, 8, 768] -> [2, 8, 768]
035         Linear                transformer.blocks.2.attention.qkv_projection     [2, 8, 768] -> [2, 8, 2304]
036         Dropout               transformer.blocks.2.attention.attention_dropout  [2, 12, 8, 8] -> [2, 12, 8, 8]
037         Linear                transformer.blocks.2.attention.output_projection  [2, 8, 768] -> [2, 8, 768]
038         Dropout               transformer.blocks.2.attention.residual_dropout   [2, 8, 768] -> [2, 8, 768]
039       LayerNorm             transformer.blocks.2.ln_2                           [2, 8, 768] -> [2, 8, 768]
040       GPT2MLP               transformer.blocks.2.mlp                            [2, 8, 768] -> [2, 8, 768]
041         Linear                transformer.blocks.2.mlp.fc                       [2, 8, 768] -> [2, 8, 3072]
042         GPT2GELU              transformer.blocks.2.mlp.activation               [2, 8, 3072] -> [2, 8, 3072]
043         Linear                transformer.blocks.2.mlp.projection               [2, 8, 3072] -> [2, 8, 768]
... [48 lines omitted, full log in artifacts/logs/] ...
004 LayerNorm
005 Linear
006 Dropout
007 Linear
008 Dropout
009 CausalSelfAttention
010 LayerNorm
011 Linear
012 GPT2GELU
013 Linear

86 module calls; parameters/buffers untouched, hooks removed.
```

### 8. Execution trace

`python -m examples.execution_trace` | exit 0 | 1.7s

```text
wrote artifacts/execution_trace.json (112 KiB): format=tensormorph-pytorch-trace v1
77 parameters, 7 buffers, 86 events
  0 model                                                         DistilGPT2LMHeadModel[[2, 8]] -> [[2, 8, 50257]]
  1 transformer                                                   GPT2Model           [[2, 8]] -> [[2, 8, 768]]
  2 transformer.embeddings                                        GPT2Embeddings      [[2, 8]] -> [[2, 8, 768]]
  3 transformer.embeddings.token_embedding                        TokenEmbedding      [[2, 8]] -> [[2, 8, 768]]
  4 transformer.embeddings.position_embedding                     PositionEmbedding   [[1, 8]] -> [[1, 8, 768]]
  5 transformer.embeddings.dropout                                Dropout             [[2, 8, 768]] -> [[2, 8, 768]]
  6 transformer.blocks.0                                          TransformerBlock    [[2, 8, 768]] -> [[2, 8, 768]]
  7 transformer.blocks.0.ln_1                                     LayerNorm           [[2, 8, 768]] -> [[2, 8, 768]]
  8 transformer.blocks.0.attention                                CausalSelfAttention [[2, 8, 768]] -> [[2, 8, 768]]
  9 transformer.blocks.0.attention.qkv_projection                 Linear              [[2, 8, 768]] -> [[2, 8, 2304]]
 10 transformer.blocks.0.attention.attention_dropout              Dropout             [[2, 12, 8, 8]] -> [[2, 12, 8, 8]]
 11 transformer.blocks.0.attention.output_projection              Linear              [[2, 8, 768]] -> [[2, 8, 768]]
...
attention internals: [('block.0.attention.qkv', [2, 8, 2304]), ('block.0.attention.q', [2, 12, 8, 64]), ('block.0.attention.k', [2, 12, 8, 64]), ('block.0.attention.v', [2, 12, 8, 64]), ('block.0.attention.scores', [2, 12, 8, 8]), ('block.0.attention.mask', [8, 8]), ('block.0.attention.probs', [2, 12, 8, 8]), ('block.0.attention.context', [2, 12, 8, 64])]
outputs: [('model.final_hidden', [2, 8, 768]), ('model.logits', [2, 8, 50257])]
```

### 9. Causal LM loss

`python -m examples.causal_lm_loss` | exit 0 | 1.8s

```text
logits       [2, 8, 50257]
shift_logits [2, 7, 50257]  (logits[:, :-1])
shift_labels [2, 7]  (labels[:, 1:])
tensor([[  11,  616, 1438,  318, 1332,   13,  198],
        [ 318,  257, 1332, 6827,  284, 1332,   13]])
per-token loss: [4.029, 5.513, 2.314, 0.343, 11.667, 3.442, 2.264, 5.526, 1.748, 6.791, 9.638, 3.723, 3.03, 4.042]
loss = 4.576363  (mean of per-token = 4.576363)
```

### 10. Single train step

`python -m examples.single_train_step` | exit 0 | 1.8s

```text
loss before: 4.576363  grad_fn=NllLossBackward0
embedding (tied with lm_head)   grad shape=[50257, 768] mean=+6.744e-11 std=7.508e-03 max|g|=2.713e+01
attention qkv weight [out, in]  grad shape=[2304, 768] mean=+5.373e-07 std=1.113e-03 max|g|=6.312e-02
mlp fc weight [out, in]         grad shape=[3072, 768] mean=-1.124e-06 std=9.274e-04 max|g|=5.879e-02
ln_1 weight (block 0)           grad shape=[768] mean=-3.074e-03 std=3.267e-02 max|g|=3.689e-01
final layernorm weight          grad shape=[768] mean=-1.512e-02 std=1.713e-01 max|g|=2.219e+00
embedding rows with nonzero grad: 50257 of 50257 (lm_head softmax touches all rows)
loss after one SGD step (lr=0.001): 2.711443
```

### 11. Autograd inspect

`python -m examples.autograd_inspect` | exit 0 | 1.8s

```text
parameter: requires_grad=True is_leaf=True grad_fn=None grad=None
activation block.0.mlp.fc.output: requires_grad=True is_leaf=False grad_fn=ViewBackward0
input_ids: requires_grad=False (integer tensors never do)
loss: grad_fn=NllLossBackward0

backward graph from loss (BFS, first 30 nodes):
  NllLossBackward0             -> ['LogSoftmaxBackward0']
  LogSoftmaxBackward0          -> ['UnsafeViewBackward0']
  UnsafeViewBackward0          -> ['CloneBackward0']
  CloneBackward0               -> ['SliceBackward0']
  SliceBackward0               -> ['UnsafeViewBackward0']
  UnsafeViewBackward0          -> ['MmBackward0']
  MmBackward0                  -> ['ViewBackward0', 'TBackward0']
  ViewBackward0                -> ['NativeLayerNormBackward0']
  TBackward0                   -> ['AccumulateGrad']
  NativeLayerNormBackward0     -> ['AddBackward0', 'AccumulateGrad', 'AccumulateGrad']
  AccumulateGrad               -> []
  AddBackward0                 -> ['AddBackward0', 'ViewBackward0']
  AccumulateGrad               -> []
  AccumulateGrad               -> []
  AddBackward0                 -> ['AddBackward0', 'ViewBackward0']
  ViewBackward0                -> ['AddmmBackward0']
  AddBackward0                 -> ['AddBackward0', 'ViewBackward0']
  ViewBackward0                -> ['AddmmBackward0']
  AddmmBackward0               -> ['AccumulateGrad', 'ViewBackward0', 'TBackward0']
  AddBackward0                 -> ['AddBackward0', 'ViewBackward0']
  ViewBackward0                -> ['AddmmBackward0']
  AddmmBackward0               -> ['AccumulateGrad', 'ViewBackward0', 'TBackward0']
  AccumulateGrad               -> []
  ViewBackward0                -> ['MulBackward0']
  TBackward0                   -> ['AccumulateGrad']
  AddBackward0                 -> ['AddBackward0', 'ViewBackward0']
  ViewBackward0                -> ['AddmmBackward0']
  AddmmBackward0               -> ['AccumulateGrad', 'ViewBackward0', 'TBackward0']
  AccumulateGrad               -> []
  ViewBackward0                -> ['ViewBackward0']

after backward: weight.grad shape=[3072, 768], hidden.grad shape=[2, 8, 3072] (kept via retain_grad)
```

### 12. Convert to Safetensors

`python -m tools.convert_to_safetensors` | exit 0 | 1.6s

```text
wrote artifacts/model.safetensors (352,824,504 bytes): 82 tensors, values bit-identical to the .bin
omitted tied tensors (reconstructed on load): {'lm_head.weight': 'transformer.wte.weight'}
```

### 13. Inspect Safetensors

`python -m tools.inspect_safetensors` | exit 0 | 0.6s

```text
file: artifacts/model.safetensors
header bytes: 8376
metadata: {'source': 'pytorch_model.bin', 'format': 'pt', 'tied': '{"lm_head.weight": "transformer.wte.weight"}'}
name                                                    shape             dtype offsets (begin,end)              numel        bytes
transformer.h.0.attn.bias                               [1,1,1024,1024]   F32   0,4194304                    1,048,576    4,194,304
transformer.h.0.attn.c_attn.bias                        [2304]            F32   4194304,4203520                  2,304        9,216
transformer.h.0.attn.c_attn.weight                      [768,2304]        F32   4203520,11281408             1,769,472    7,077,888
transformer.h.0.attn.c_proj.bias                        [768]             F32   11281408,11284480                  768        3,072
transformer.h.0.attn.c_proj.weight                      [768,768]         F32   11284480,13643776              589,824    2,359,296
transformer.h.0.ln_1.bias                               [768]             F32   13643776,13646848                  768        3,072
transformer.h.0.ln_1.weight                             [768]             F32   13646848,13649920                  768        3,072
transformer.h.0.ln_2.bias                               [768]             F32   13649920,13652992                  768        3,072
transformer.h.0.ln_2.weight                             [768]             F32   13652992,13656064                  768        3,072
transformer.h.0.mlp.c_fc.bias                           [3072]            F32   13656064,13668352                3,072       12,288
transformer.h.0.mlp.c_fc.weight                         [768,3072]        F32   13668352,23105536            2,359,296    9,437,184
transformer.h.0.mlp.c_proj.bias                         [768]             F32   23105536,23108608                  768        3,072
transformer.h.0.mlp.c_proj.weight                       [3072,768]        F32   23108608,32545792            2,359,296    9,437,184
transformer.h.1.attn.bias                               [1,1,1024,1024]   F32   32545792,36740096            1,048,576    4,194,304
transformer.h.1.attn.c_attn.bias                        [2304]            F32   36740096,36749312                2,304        9,216
transformer.h.1.attn.c_attn.weight                      [768,2304]        F32   36749312,43827200            1,769,472    7,077,888
transformer.h.1.attn.c_proj.bias                        [768]             F32   43827200,43830272                  768        3,072
transformer.h.1.attn.c_proj.weight                      [768,768]         F32   43830272,46189568              589,824    2,359,296
transformer.h.1.ln_1.bias                               [768]             F32   46189568,46192640                  768        3,072
transformer.h.1.ln_1.weight                             [768]             F32   46192640,46195712                  768        3,072
transformer.h.1.ln_2.bias                               [768]             F32   46195712,46198784                  768        3,072
transformer.h.1.ln_2.weight                             [768]             F32   46198784,46201856                  768        3,072
transformer.h.1.mlp.c_fc.bias                           [3072]            F32   46201856,46214144                3,072       12,288
transformer.h.1.mlp.c_fc.weight                         [768,3072]        F32   46214144,55651328            2,359,296    9,437,184
transformer.h.1.mlp.c_proj.bias                         [768]             F32   55651328,55654400                  768        3,072
transformer.h.1.mlp.c_proj.weight                       [3072,768]        F32   55654400,65091584            2,359,296    9,437,184
transformer.h.2.attn.bias                               [1,1,1024,1024]   F32   65091584,69285888            1,048,576    4,194,304
transformer.h.2.attn.c_attn.bias                        [2304]            F32   69285888,69295104                2,304        9,216
transformer.h.2.attn.c_attn.weight                      [768,2304]        F32   69295104,76372992            1,769,472    7,077,888
transformer.h.2.attn.c_proj.bias                        [768]             F32   76372992,76376064                  768        3,072
transformer.h.2.attn.c_proj.weight                      [768,768]         F32   76376064,78735360              589,824    2,359,296
transformer.h.2.ln_1.bias                               [768]             F32   78735360,78738432                  768        3,072
transformer.h.2.ln_1.weight                             [768]             F32   78738432,78741504                  768        3,072
transformer.h.2.ln_2.bias                               [768]             F32   78741504,78744576                  768        3,072
transformer.h.2.ln_2.weight                             [768]             F32   78744576,78747648                  768        3,072
transformer.h.2.mlp.c_fc.bias                           [3072]            F32   78747648,78759936                3,072       12,288
transformer.h.2.mlp.c_fc.weight                         [768,3072]        F32   78759936,88197120            2,359,296    9,437,184
transformer.h.2.mlp.c_proj.bias                         [768]             F32   88197120,88200192                  768        3,072
transformer.h.2.mlp.c_proj.weight                       [3072,768]        F32   88200192,97637376            2,359,296    9,437,184
transformer.h.3.attn.bias                               [1,1,1024,1024]   F32   97637376,101831680           1,048,576    4,194,304
transformer.h.3.attn.c_attn.bias                        [2304]            F32   101831680,101840896              2,304        9,216
... [31 lines omitted, full log in artifacts/logs/] ...
transformer.h.5.ln_2.bias                               [768]             F32   176378880,176381952                768        3,072
transformer.h.5.ln_2.weight                             [768]             F32   176381952,176385024                768        3,072
transformer.h.5.mlp.c_fc.bias                           [3072]            F32   176385024,176397312              3,072       12,288
transformer.h.5.mlp.c_fc.weight                         [768,3072]        F32   176397312,185834496          2,359,296    9,437,184
transformer.h.5.mlp.c_proj.bias                         [768]             F32   185834496,185837568                768        3,072
transformer.h.5.mlp.c_proj.weight                       [3072,768]        F32   185837568,195274752          2,359,296    9,437,184
transformer.ln_f.bias                                   [768]             F32   195274752,195277824                768        3,072
transformer.ln_f.weight                                 [768]             F32   195277824,195280896                768        3,072
transformer.wpe.weight                                  [1024,768]        F32   195280896,198426624            786,432    3,145,728
transformer.wte.weight                                  [50257,768]       F32   198426624,352816128         38,597,376  154,389,504

82 tensors, 352,816,128 data bytes (336.47 MiB)
```

### 14. Hugging Face parity

`python -m reference.compare_huggingface --check-gelu-variants` | exit 0 | 5.1s

```text
HF load_state_dict: missing=[] unexpected=['transformer.h.0.attn.bias', 'transformer.h.1.attn.bias', 'transformer.h.2.attn.bias', 'transformer.h.3.attn.bias', 'transformer.h.4.attn.bias', 'transformer.h.5.attn.bias']
tanh GELU (ours)       max_abs=7.629e-05 mean_abs=9.686e-06 max_rel=1.067e-06 allclose(rtol=1e-4,atol=1e-3)=True
argmax identical: True
erf GELU (wrong)       max_abs=7.070e-02 mean_abs=1.966e-02 max_rel=1.388e-03 allclose(rtol=1e-4,atol=1e-3)=False
```

### 15. FX trace

`python -m examples.fx_trace` | exit 0 | 1.9s

```text
opcode          name                                              target                                            shape               args
placeholder     input_ids                                         input_ids                                         [2,8]               
placeholder     labels_1                                          labels_1                                          None                None
call_function   _assert_is_none                                   _assert_is_none                                   None                labels_1, 'labels has been specialized to have value None but got another value'
placeholder     position_ids_1                                    position_ids_1                                    None                None
call_function   _assert_is_none_1                                 _assert_is_none                                   None                position_ids_1, 'position_ids has been specialized to have value None but got another value'
placeholder     return_debug_1                                    return_debug_1                                    None                False
call_function   eq                                                eq                                                None                return_debug_1, False
call_function   _assert                                           _assert                                           None                eq, 'return_debug has been specialized to have value False but got another value'
call_method     size                                              size                                              None                input_ids, 1
call_function   le                                                le                                                None                size, 1024
call_function   _assert_1                                         _assert                                           None                le, 'sequence longer than max_position_embeddings'
get_attr        transformer_embeddings_position_ids               transformer.embeddings.position_ids               [1024]              
call_function   narrow                                            narrow                                            [8]                 transformer_embeddings_position_ids, 0, 0, size
call_method     unsqueeze                                         unsqueeze                                         [1,8]               narrow, 0
get_attr        transformer_embeddings_token_embedding_weight     transformer.embeddings.token_embedding.weight     [50257,768]         
call_function   embedding                                         embedding                                         [2,8,768]           input_ids, transformer_embeddings_token_embedding_weight
get_attr        transformer_embeddings_position_embedding_weight  transformer.embeddings.position_embedding.weight  [1024,768]          
call_function   embedding_1                                       embedding                                         [1,8,768]           unsqueeze, transformer_embeddings_position_embedding_weight
call_function   add                                               add                                               [2,8,768]           embedding, embedding_1
call_module     transformer_embeddings_dropout                    transformer.embeddings.dropout                    [2,8,768]           add
call_module     transformer_blocks_0_ln_1                         transformer.blocks.0.ln_1                         [2,8,768]           transformer_embeddings_dropout
call_module     transformer_blocks_0_attention_qkv_projection     transformer.blocks.0.attention.qkv_projection     [2,8,2304]          transformer_blocks_0_ln_1
call_method     chunk                                             chunk                                             None                transformer_blocks_0_attention_qkv_projection, 3
call_function   getitem                                           getitem                                           [2,8,768]           chunk, 0
call_function   getitem_1                                         getitem                                           [2,8,768]           chunk, 1
call_function   getitem_2                                         getitem                                           [2,8,768]           chunk, 2
call_method     size_1                                            size                                              None                getitem, 0
call_method     size_2                                            size                                              None                getitem, 1
call_method     view                                              view                                              [2,8,12,64]         getitem, size_1, size_2, 12, 64
call_method     permute                                           permute                                           [2,12,8,64]         view, 0, 2, 1, 3
call_method     size_3                                            size                                              None                getitem_1, 0
call_method     size_4                                            size                                              None                getitem_1, 1
call_method     view_1                                            view                                              [2,8,12,64]         getitem_1, size_3, size_4, 12, 64
call_method     permute_1                                         permute                                           [2,12,8,64]         view_1, 0, 2, 1, 3
call_method     size_5                                            size                                              None                getitem_2, 0
call_method     size_6                                            size                                              None                getitem_2, 1
call_method     view_2                                            view                                              [2,8,12,64]         getitem_2, size_5, size_6, 12, 64
call_method     permute_2                                         permute                                           [2,12,8,64]         view_2, 0, 2, 1, 3
call_method     transpose                                         transpose                                         [2,12,64,8]         permute_1, -2, -1
call_function   matmul                                            matmul                                            [2,12,8,8]          permute, transpose
call_function   truediv                                           truediv                                           [2,12,8,8]          matmul, 8.0
call_method     size_7                                            size                                              None                truediv, -1
get_attr        transformer_blocks_0_attention_causal_mask        transformer.blocks.0.attention.causal_mask        [1024,1024]         
call_function   narrow_1                                          narrow                                            [8,1024]            transformer_blocks_0_attention_causal_mask, 0, 0, size_7
call_function   narrow_2                                          narrow                                            [8,8]               narrow_1, 1, 0, size_7
... 335 nodes total -> artifacts/fx_graph.txt, artifacts/fx_graph.json
node kinds: {'placeholder': 4, 'call_function': 136, 'call_method': 128, 'get_attr': 9, 'call_module': 57, 'output': 1}
traced module output == eager output: True
```

### 16. torch.export

`python -m examples.export_model` | exit 0 | 2.4s

```text
exported (346 graph nodes) -> artifacts/export/graph.json, artifacts/export/exported_program.txt
graph inputs by kind: {'PARAMETER': 77, 'BUFFER': 7, 'USER_INPUT': 1}
top aten ops: [('aten.mul.Tensor', 36), ('aten.add.Tensor', 25), ('aten.linear.default', 25), ('aten.view.default', 24), ('aten.permute.default', 24), ('aten.dropout.default', 19), ('<built-in function getitem>', 18), ('aten.narrow.default', 13), ('aten.layer_norm.default', 13), ('aten.matmul.default', 12)]
exported module matches eager: True
```

### 17. Tests

`pytest -q` | exit 0 | 9.6s

```text
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 8.31s
```

## Generated artifacts

| file | size |
|---|--:|
| `artifacts/execution_trace.json` | 115,037 B |
| `artifacts/export/exported_program.txt` | 77,897 B |
| `artifacts/export/graph.json` | 90,996 B |
| `artifacts/fx_graph.json` | 68,763 B |
| `artifacts/fx_graph.txt` | 53,533 B |
| `artifacts/model.safetensors` | 352,824,504 B |
| `artifacts/tensor_inventory.csv` | 11,185 B |
| `artifacts/tensor_inventory.json` | 31,542 B |
