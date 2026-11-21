# 07e DESIGN

## Blocks and templates

- `encoder` template (BERT family): word + position (+ token type) embeddings → LayerNorm (`lower.py`'s layernorm
  composite) → post-norm layers (bidirectional MHA with biases, residual + LayerNorm, exact GELU via `erf`, residual +
  LayerNorm); the last hidden state is `L`. Task heads (pooler, MLM, classifier, QA) are components declared without a
  graph, so these rows are `partial`.
- `seq2seq` template: T5 / mT5 (pre-norm RMS, relative position buckets — HF's bucketing as an int64 constant [q, k] —
  gathered from block 0's [Nb, H] table, no score scaling, ReLU or gated GELU, tied head rescaled by D^-0.5) and BART
  (post-norm LayerNorm, learned positions offset 2, layernorm_embedding, GELU, final_logits_bias). Encoder and decoder
  share T; cross-attention reads the encoder output; L is the decoder's logits.
- `Program` gains the unary functions `erf` and `relu` (TEL builtins).

## Keys (encoder + encoder_decoder catalog rows) and C4 hand counts

| key | rows | descriptor(s) | C4 closed form (per layer, B·T = 8) |
| --- | --: | --- | --- |
| bert | 1237 | bert, bert_tf (gamma/beta names), bert_bare (BertModel) | 8 contractions: q, k, v, o, scores, context, intermediate, output; MACs 4·BT·D·Dq + 2·B·H·T²·Dh + 2·BT·D·F |
| xlm-roberta 497, roberta 373 | 870 | roberta, bert_bare | as bert |
| distilbert | 189 | distilbert | as bert (no token types) |
| t5 234, mt5 90 | 324 | t5 | encoder 6 + ffn (2 ReLU, 3 gated), decoder 12 + ffn, head 1 |
| bart | 117 | bart, bart_bare | encoder 8, decoder 14, head 1 |
| modernbert 553, deberta-v2 304, marian 117 | — | — | not this wave: rotary/local-global encoder, disentangled attention, sinusoidal-position Marian |

No encoder goldens exist, so C4 compares the build with these closed forms (tools/conformance.py).
Predicted rows unlocked: bert family 2,296 + t5/mt5 324 + bart 117 = 2,737.

## Later fixes (2026-11-21, encoder rows from the sample run)

TF-era `LayerNorm.gamma` / `.beta` bind as `.weight` / `.bias` (bind.legacy_key, the rename transformers applies on
load; path B gets it from the class's WeightRenaming list). Encoder GGUFs (arch `bert`) get a scheme generated from
gguf-py's HF -> GGUF tables (namemaps.tabled_scheme) and Hq from `bert.attention.head_count`. Path B ignores every
stored `*.inv_freq` (ESM's rotary_embeddings). Misses kept: GLiNER/GLiClass and other custom libraries (no config
class), jina's XLM-RoBERTa-flash (`emb_ln`, remote code), ALBERT (no descriptor; path B), ModernBERT task wrappers.

## Observed (07e.C, 2026-11-21 catalog run)

Catalog rows by `model_type`, ok + partial (heads as components) of the non-deferred rows the prediction counts:
bert 1,170 / 1,237, roberta + xlm-roberta 823 / 870, distilbert 186 / 189, t5 + mt5 272 / 324, bart 96 / 117:
2,547 = 2,737 − the 190 misses below. Path B adds modernbert 488, deberta-v2 280, electra 58, mpnet 33, tapas 25,
albert 15, esm 10 (encoder) and mbart 38, m2m_100 18, pegasus 9, marian 5 (encoder_decoder).

Misses (190): t5/mt5 GGUF with no generated scheme (32: llama.cpp's `enc.` / `dec.` names; bound after this record by tabled_scheme + GGUF hyper sources + gate presence), BERT / XLM-R GGUF
embedders whose binding fails (26: bge-m3, snowflake-arctic, jina qkv-fused), remote code (35), gated (9), BART/T5
classification heads and T5 encoder-only cross-encoders (22), custom heads and variants (ColBERT `linear`,
`distance_embedding`, `scorer`, geneformer, gena-lm: ~20), no candidates (~17). After this record: BART without
`model.shared` / `final_logits_bias` binds the tied copy (bind) and omits the zero bias (seq2seq), BART
`classification_head` is a component and fairseq `*.version` buffers are ignored; `bart` C1 uses
facebook/bart-large-cnn. Those rows move in the final 07f pass.
