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
