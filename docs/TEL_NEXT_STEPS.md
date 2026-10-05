# DistilGPT2 × TEL: analysis and next steps

Date: 2026-10-05. Inputs: the TEL documentation set and the `tensorparser` (`packages/model-import`) code in
`tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage` (below: `TEL/`), and the artifacts of this repo
(`artifacts/pipeline_report.md`: 17/17 pipeline steps pass, 98 tests pass).

Evidence tags: **[V]** I ran or read the code myself. **[M]** measured by a research agent running a command (not re-run by me).
**[D]** read from TEL docs by a research agent, with the file named. Anything untagged is my inference.

## 1. Summary

1. **TEL can already ingest this model's structure.** The `tensorparser` CLI turns `artifacts/model.safetensors` into a compiled TEL program:
   82 tensors, 591 primitive ops, 8,376 header bytes read, **0 payload bytes read** [V]. Our `.safetensors` is accepted as-is because it matches HF's
   key set (82 keys, Conv1D `[in,out]`, F32 `attn.bias` masks kept, `lm_head` omitted).
2. **No numbers flow yet.** TEL has no payload reader and no full-width numeric golden anywhere in its repos [D `TEL/packages/tel/CODE_NAVIGATION.md`; M].
   Its GPT-2 tests are structural (counts, byte offsets) plus a synthetic seeded one-block run. This repo is the missing numerical oracle.
3. **The correspondence is exact.** Every TEL intermediate symbol in the emitted `model.tel` (`X_0`, `N1_l`, `Hq_l`, `Q_l` … `L`) has a 1:1 tensor in our `return_debug` output (section 4).
   The golden contract is a mapping, not a design problem. 87 intermediates, 2 inputs, 6 masks.
4. **DistilGPT2 stresses TEL in specific, listable places**: masking (`-inf`), the loss shift, six-fold layer repetition, a square linear with bias (DR-41), opaque `gelu`/`attention`, and tied weights (section 6).
5. **Recommended next step: a golden export from this repo** (section 7, Phase 1, about a day), then TEL-side payload loading and teacher-forced per-block parity (Phase 2). Phase 1 unblocks everything else and needs no TEL changes.

## 2. TEL in brief

TEL is an executable dialect of paper math: one semantics, three representations (Quick TEL for typing, ThreeTeX/LaTeX for display, semantic IR for the compiler) over a Typed Tensor IR that feeds TensorGraph, visualization and a runtime [D `TEL/docs/language/00-overview.md`].
Quick TEL (ASCII) is the canonical saved source (ADR-0025). Shapes and **named axes** are the type system (`X : bf16[B,T,D]`, `axis q : query = T`, `axis k : key = T`), `@` is matmul, `softmax(S, axis=k)` takes an axis by name, and `relabel(T -> q)` makes two equal-length axes distinct.
Inference happens only when the result is unique; errors are phrased as mathematics; source maps and canonical semantics (for semantic diff, e.g. GELU → SiLU) are language features.
TEL is "a mathematical tensor interface to computation", not a replacement for Python or CUDA.

Pipeline and status [D `TEL/docs/architecture/*`, `packages/tel/CODE_NAVIGATION.md`; M]:

| stage | state today |
|---|---|
| lexer → parser → normalize → infer → lower to TTIR | implemented in TypeScript, tested (about 80 corpus fixtures) |
| TTIR → graph/`TensorProgram` JSON 0.2.0 | implemented; throws on opaque ops |
| executor (`exec.run` → itensor `execute`) | "oracle, never a runtime": `Float64Array` with per-op fp32 rounding, 44 of 48 primitives |
| storage binding (`tensor://`, safetensors header, `bindStorage`) | implemented, header only. The payload hook `readPayload` is unimplemented |
| `tensorparser` GPT-2 importer | implemented, 12 test files; the only GPT-2 path |
| semantic diff, `←` bind statement, FX/export importer | not implemented (spec or roadmap only) |
| milestone | M2 (bind a real checkpoint) done; **M3 (execution) partial and current** |

Opaque composites (`gelu`, `attention`, `relu`) lower to nodes the executor refuses; only `softmax`, `norm`, `layernorm`, `grad` have expansions.

## 3. DistilGPT2 through TEL's lens

Architecture (verified from the checkpoint): 6 pre-LN blocks, C=768, 12 heads × 64, MLP 3072, vocab 50257, 1024 learned positions, tied head, fused QKV `[768,2304]`, biases everywhere, `gelu_new`.

Program emitted by TEL's importer for our checkpoint (`model.tel`, block 0 shown) [V]:

```text
N1_0 = layernorm(X_0, axis=D, eps=0.00001) * g1_0 + b1_0
Hq_0 = N1_0 @ W_qkv_0 + b_qkv_0
Q_0 = relabel(permute(reshape(Hq_0[:, :, 0:768], [B,T,H,D_h]), [0,2,1,3]), T -> q)
K_0 = relabel(permute(reshape(Hq_0[:, :, 768:1536], [B,T,H,D_h]), [0,2,1,3]), T -> k)
V_0 = relabel(permute(reshape(Hq_0[:, :, 1536:2304], [B,T,H,D_h]), [0,2,1,3]), T -> k)
S_0 = Q_0 @ K_0.T / sqrt(D_h)
A_0 = softmax(S_0 + log(Mb_0), axis=k)
C_0 = A_0 @ V_0
O_0 = reshape(permute(relabel(C_0, q -> T), [0,2,1,3]), [B,T,Dc]) @ W_o_0 + b_o_0
Xm_0 = X_0 + O_0
N2_0 = layernorm(Xm_0, axis=D, eps=0.00001) * g2_0 + b2_0
G_0 = N2_0 @ W_fc_0 + b_fc_0
M_0 = (0.5 * G_0 * (1 + tanh(0.7978845608 * (G_0 + 0.044715 * G_0 ^ 3)))) @ W_pr_0 + b_pr_0
X_1 = Xm_0 + M_0
```

with `X_0 = E[ids] + Wp[pos]`, and at the end `Xf = layernorm(X_6, axis=D, eps=0.00001) * gf + bf` and `L = Xf @ E.T` (tied head: the same symbol `E`).

| component | our PyTorch | TEL (as emitted / documented) | status |
|---|---|---|---|
| token + position embedding | `Embedding`, add | `E[ids] + Wp[pos]` (gather, named broadcast) | works; `Wp[0:T]` impossible (symbolic bound), so a `pos` input is used |
| linear, HF `[in,out]` | `Linear` `[out,in]` (transposed on load) | `X @ W + b`; `X @ W.T` for `[out,in]` stores | works; layout is an adapter convention, not checked by binding |
| square linear + bias (`c_proj`) | `Linear(768,768)` | needs a second axis name `Dc` | **DR-41**: inexpressible without the workaround |
| fused QKV split | `chunk(3, -1)` | static slices `Hq[:, :, 0:768]` | works; `chunk`/`split` are M in the catalog, `tuple_pattern` undefined in the grammar |
| head split/merge | `view` + `permute` | `reshape` + `permute` + `relabel` | works |
| scores `QKᵀ/√D` | `matmul` / `sqrt` | `Q @ K.T / sqrt(D_h)` | works |
| causal mask | `masked_fill(~mask, -inf)` | `softmax(S + log(Mb))`, `Mb` an F32 0/1 input | **workaround**: no `-inf` literal, no usable `causal_mask(q,k)` |
| softmax on key axis | `softmax(dim=-1)` | `softmax(…, axis=k)` | works (needs the `q`/`k` relabel) |
| GELU tanh | written out in `GPT2GELU` | written out; `gelu(approximate="tanh")` is opaque | **workaround**: 40-character polynomial instead of `gelu(G_0)` |
| LayerNorm eps, γ, β | `LayerNorm(eps=1e-5)` | `layernorm(…, eps=…) * g + b` | works (biased variance matches) |
| residuals | `x + f(x)` | `X_1 = Xm_0 + M_0` | works |
| six blocks, distinct weights | `ModuleList` | unrolled, decorated names (`S_3`) | works, no loop construct exists |
| tied head | same `Parameter` | reuse of `E` | works; adapter infers tying from `lm_head` being absent and ignores our `tied` metadata |
| dropout (eval) | `Dropout` (19 nodes in the export) | omitted | fine; needs a stated eval rule |
| loss with shift | `logits[:, :-1]`, `labels[:, 1:]`, `cross_entropy` | not expressible | **gap** (symbolic slice bounds, batched gather, class axis fixed at 1) |
| backward | `loss.backward()` | `grad` (P0 in the spec) | blocked by the loss gap |

## 4. The golden contract: TEL symbol ↔ our tensor [V]

Names come from `TEL/packages/model-import/src/program/emit.ts`; ours from `model(ids, return_debug=True).debug`. `l` = 0…5.

| TEL symbol | our debug name | shape (B=2, T=8) |
|---|---|---|
| `ids`, `pos` | `input.input_ids`, `position_ids[:T]` | `[2,8]`, `[8]` |
| `Mb_l` | `block.l.attention.mask` (as F32 0/1) | `[8,8]` |
| `X_0` | `embedding.output` | `[2,8,768]` |
| `N1_l` | `block.l.ln_1.output` | `[2,8,768]` |
| `Hq_l` | `block.l.attention.qkv` | `[2,8,2304]` |
| `Q_l`, `K_l`, `V_l` | `block.l.attention.q`, `.k`, `.v` | `[2,12,8,64]` |
| `S_l` | `block.l.attention.scores` (scaled, unmasked) | `[2,12,8,8]` |
| `A_l` | `block.l.attention.probs` | `[2,12,8,8]` |
| `C_l` | `block.l.attention.context` | `[2,12,8,64]` |
| `O_l` | `block.l.attention.output` | `[2,8,768]` |
| `Xm_l` | `block.l.attention_residual.output` | `[2,8,768]` |
| `N2_l` | `block.l.ln_2.output` | `[2,8,768]` |
| `G_l` | `block.l.mlp.fc.output` | `[2,8,3072]` |
| `M_l` | `block.l.mlp.output` | `[2,8,768]` |
| `X_{l+1}` | `block.l.output` | `[2,8,768]` |
| `Xf` | `model.final_hidden` | `[2,8,768]` |
| `L` | `model.logits` | `[2,8,50257]` |

Count: 14 symbols per block (`N1, Hq, Q, K, V, S, A, C, O, Xm, N2, G, M, X_{l+1}`) × 6 = 84, plus `X_0`, `Xf`, `L` = **87 intermediates** (95 tensors per sample with `ids`, `pos` and the six masks). Our `mlp.activation.output` has no TEL counterpart (TEL inlines the GELU), so it stays a PyTorch-only tensor.
We already have five fixed samples (`[1,8]`, `[2,8]`, `[2,4]`, repeated tokens, attention debug), so the same TEL program with symbolic `B`, `T` can be checked at several bindings.

## 5. What the architecture and measurements tell us

**Compute is split almost evenly between the blocks and the head [V].** Per token, the six blocks hold 6 × 7,077,888 = 42.5M matmul MACs; `lm_head` holds 38.6M (**47.6%**).
Generation needs logits only at the last position, and pushing a slice through `L = Xf @ E.T` cuts that term by T×. TEL's demand-driven evaluation and pushdown (ADR-0008) is the right mechanism, and this model is a clean demonstration: `L[:, -1]` versus `L`.
TEL's cost model (`ir.cost`, from types) can be checked against these hand numbers.

**The tied embedding is 47% of all parameters (38.6M of 81.9M) and is read two different ways [V].** The lookup touches 8 of 50257 rows; the unembedding needs all of them. Its gradient is the sum of both paths (our `test_weight_tying` checks that all rows get gradient).
That makes it the natural test of storage-region reads (`tensor://…#sel`) and, later, of `grad` through a shared symbol.

**Tolerances must be scale-aware [V].** I ran the same model in float64 and compared (fixed batch, `[2,8]`):

| tensor | max abs error fp32 vs fp64 | max abs value |
|---|--:|--:|
| logits | 1.9e-4 | ~92 |
| `embedding.output` | 1.6e-7 | 4.5 |
| `block.0.attention.scores` | 6.6e-6 | 18.7 |
| `block.0.attention.probs` | 1.6e-6 | 1.0 |
| `block.0.output` | 3.1e-4 | 251 |
| `block.2.output` | 1.3e-3 | **1705** |
| `block.5.output` | 1.6e-3 | 507 |
| `model.final_hidden` | 7.5e-5 | 154 |

The residual stream has large outliers (up to ~1700 at block 2), so a fixed `atol` is wrong. PyTorch fp32's own error against the exact value is about 1e-6 of the tensor's max magnitude; the HF-versus-ours difference (7.6e-5 on logits) is smaller than fp32's error against fp64.
Proposal for DR-38 (tolerance policy): `|a − b| ≤ 1e-5 · max|reference|` per tensor, checked against both the fp32 and the fp64 golden. TEL's executor computes in fp64 with per-op fp32 rounding, so it should land inside the same band as fp32 PyTorch. The fp64 golden separates "TEL is wrong" from "fp32 rounding differs".

**Explicit attention is O(T²) memory [V].** At T=1024, B=1, the scores of one layer are 12 × 1024² × 4 B = 48 MiB (six layers, plus probabilities). TEL's decomposition keeps `attention` expandable (P14); recognising `softmax(QKᵀ/√d + mask)V` and fusing it is the kernel-lowering question (`TEL/docs/operators/62-kernel-lowering.md`) [D].

**Ingestion fit of our artifacts [M unless noted]:**

| artifact | fit with TEL | main mismatch |
|---|---|---|
| `artifacts/model.safetensors` | **works** [V]: compiled, 82 tensors, 591 ops, payloadReads 0 | URIs carry no revision; `weights/model.safetensors` and ours differ in header size (8,277 vs 8,376 B), hence in absolute offsets |
| `tensor_inventory.json` | rejected as-is (bare list); wrapped it imports as `inventory-only`; with `model_type: gpt2` it throws | our names (`blocks.N.attention.qkv_projection`) vs HF keys; weights `[out,in]` vs on-disk `[in,out]`; bool `[1024,1024]` mask vs F32 `[1,1,1024,1024]`; extra `position_ids`; duplicate `lm_head.weight` reads as "untied" |
| execution trace JSON | no ingestion point; usable as a shape oracle for `exec` named values | module paths vs TEL block spans; bool `[T,T]` mask vs F32 `Mb` |
| FX graph (335 nodes) | none (a PyTorch importer is an open roadmap row) | `call_module` leaves, method calls: less normalised than TTIR |
| `torch.export` graph (346 aten nodes) | none yet; closest to TTIR | 37 contractions in both (25 `linear` + 12 `matmul` = 37 TEL `Contraction`); TEL has 591 ops because 144 are explicit `Broadcast`; `masked_fill(-inf)` vs `+ log(Mb)`; 19 `dropout(train=False)` nodes |

The adapter reads HF config keys (`n_head`…), while our trace config says `num_heads`; head count is "assumed 12 of 64", correct by luck. A config file with HF key names removes that.

## 6. Where DistilGPT2 forces decisions in TEL

Ordered by how much of the model each item blocks. References are to the TEL docs [D] unless marked.

| # | gap | evidence | suggested resolution |
|---|---|---|---|
| 1 | **Masking**: no `-inf` literal; `causal_mask(q,k)` is a catalog row with integer arguments returning bool; `S + causal_mask(q,k)` in example 03 adds a bool to a float and is self-referential | `language/` has no `-inf`; `operators/23-attention.md`, `60-decomposition.md` recipe-attention uses `where(causal_mask(q,k), s, -inf)` | define `-inf` and a mask-by-axis form; fix mask polarity once (`masked_fill` fills where true, attention `mask` keeps where true); retire the `log(Mb)` trick |
| 2 | **Opaque composites**: `gelu`, `attention`, `layernorm` with γ/β are not expandable; `gelu(approximate)` has no attribute in itensor | `CODE_NAVIGATION.md`; `operators/21-activation.md`, `22-normalization.md` Gaps | add expansions so the emitter can write `gelu(G_0, approximate="tanh")`, `attention(Q_0,K_0,V_0, causal=true)`. Keep the polynomial form as the regression oracle: expansion must equal it |
| 3 | **Loss**: symbolic slice bounds (`0:T-1`), negative indices, batched gather, `cross_entropy` class axis fixed at position 1 | TM2104 (emitter comment); `08-indexing.md`; ADR-0023 is V2; `operators/25-losses.md` | pick a named-axis `cross_entropy(logits, targets, axis=V)` and a shift form; this unlocks training goldens |
| 4 | **Layer repetition and tying**: no loop or layer-family construct in `language/`; functions lower to opaque nodes | `12-functions.md`; ADR-0015 families `W^{(l)}` apply only at the bind site | keep unrolling as the readable form, decide whether a function call must expand; document tying as symbol reuse |
| 5 | **Square linear + bias (DR-41)**: `c_proj` `[768,768]`; the bias axis is ambiguous without a slot | `DECISIONS-REQUIRED.md` DR-41; emitter comment on TM2205 | slot syntax or `out=` annotation (the `Dc` axis is the current workaround) |
| 6 | **Grammar**: `chunk`/`split` tuple results, `tuple_pattern`, keyword arguments (`eps=`, `axis=`) are referenced but not defined | `language/04-grammar.md`; DR-43 (PyTorch spellings vs catalog) | define them; the emitter then uses `chunk(Hq, 3, axis=QKV)` instead of static slices |
| 7 | **Bind**: `←` and morph-at-bind are ADR-only; `BindRequest` has no `role`; `bindStorage` cannot catch a transposed square weight | ADR-0015, DR-40 | add a bind-time value check (a golden-hash or a probe row), since shape checks pass for transposed `[768,768]` |
| 8 | **Eval semantics**: `dropout` requires a key; exported nodes carry `train=False` | `operators/30-random.md` | state "identity in eval" as a rule so imported graphs normalise without a `Key` |
| 9 | **Doc drift** (for the TEL maintainers) | `docs/README.md` still says the ADRs contradict `language/` on eleven points, but ADR-0025 closed them; executor primitive counts (32 vs 44 of 48); corpus counts (67 vs 81) | refresh banners and counts |

## 7. Recommended roadmap

Effort: S under a day, M a few days, L a week or more. "Here" is this repo; "TEL" is `tensormorph-source`.

**Phase 1: golden contract (here, S).** Highest value, no TEL changes.
- `tools/export_golden.py`: for each of the five samples write `artifacts/golden/<sample>.f32.safetensors` and `.f64.safetensors` holding `ids`, `pos`, `Mb_l`, and the 87 intermediates under **TEL symbol names**, plus `manifest.json` (symbol, our name, shape, dtype, `max_abs`, tolerance, source `sha256` of `model.safetensors`).
- `artifacts/config.json` with HF key names (`n_head`, `n_layer`, `n_embd`, `n_positions`, `activation_function: gelu_new`, `layer_norm_epsilon`).
- Add `checkpoint_key`, `layout` (`in-out` on disk vs `out-in` in the pure model) and `tel_symbol` to the inventory and trace, so they join with TEL bindings (`W_qkv_0` ↔ `transformer.h.0.attn.c_attn.weight`, transposed).
- Optional `--hash` in the trace (tensor-byte sha256) to match the recorder format TEL's TensorCore evidence uses [D `TEL/.plan/tensorcore`], which records HF's own model; ours would be a second, independent reference.
- Acceptance: the manifest covers 95 tensors per sample; fp32-vs-fp64 error stays inside the proposed band; a test checks shapes against the emitted `model.tel` declarations.

**Phase 2: numeric bring-up (TEL, M).**
- Implement the `readPayload` hook (fp32 → `Float64Array` lazily, per tensor, demand-driven).
- Teacher-forced per-block parity: feed golden `X_l`, compare every symbol of block `l`; this stops one error from hiding behind the next. Then end-to-end `L` and `L[:, -1]`.
- Measure time and memory (about 706 MB of `Float64Array` for all stored weights; the token table alone is about 309 MB) and set the tolerance numbers for DR-38.
- Acceptance: all 87 tensors within `1e-5 · max|ref|` on fp32 and fp64 goldens, at every sample's `(B,T)` without recompiling.

**Phase 3: language gaps (TEL spec, M).** Items 1, 2, 5 and 6 from section 6 first; after each, regenerate the program and re-run Phase 2 (the golden is the regression test). Then 3, 4, 7, 8.

**Phase 4: loss and gradients (both, M).** Extend the golden with loss, `shift_logits`, per-token loss, and gradients from `examples/single_train_step.py` (embedding with both paths, `W_qkv_0`, `W_fc_0`, `ln_1`). Verify TEL `grad`. Needs gap 3.

**Phase 5: graph interop (both, L).** `torch.export` → TTIR importer (the roadmap row that is not built) with a structural equivalence check against the emitted TTIR (37 contractions, 13 layernorms, 6 softmaxes, 2 gathers), after normalising `linear`, `masked_fill`, `dropout`. This is the first realistic test of semantic diff (V2): `gelu_new` → `silu` should show up as one change.

**Phase 6: generalise.** `gpt2` (124M) and `gpt2-medium` through the same adapter; bf16 goldens for the dtype rules (`operators/03-dtype-and-promotion.md`); a second pure-PyTorch architecture in this repo with RoPE/RMSNorm/SwiGLU, because TEL's own transformer-block example is that style and has no goldens.

## 8. Risks and decisions for you

- **Where the golden lives.** Here (`artifacts/golden/`, ignored by git, regenerated) or committed as fixtures in TEL (about 3 MB fp32 per sample, dominated by logits)? I suggest generating here and copying a trimmed set (blocks plus `L[:, -1]`) into TEL.
- **Who changes TEL.** Phases 2 and 3 touch the TEL repo, which I have only read so far. Should I propose patches or stay on this side?
- **Executor performance.** A JS `Float64Array` executor over 82M weights may be too slow for full-width runs; per-block teacher forcing and last-position-only `L` keep the first runs small. TensorCore (Rust/CUDA/Metal) is a separate stack with its own claims of a DistilGPT2 per-operator comparison [D `TEL/.plan/tensorcore/STATUS.md`; status docs conflict with `TEL/docs/ARCHITECTURE.md`], which I did not run.
- **Stale revisions.** TEL URIs carry no revision, so our file and HF's file give the same URI with different offsets. Pin by `sha256` in the golden manifest.
- **Counts differ across TEL docs by date**, so treat any number tagged [D] as approximate until re-run.

## 9. How this was produced

Pipeline: `python -m tools.run_pipeline` → `artifacts/pipeline_report.md` (17 steps, all PASS; logs in `artifacts/logs/`).
Reading: three parallel read-only agents covered (a) language and examples, (b) architecture, status and artifact fit, (c) the operator catalog. I read `language/00-overview.md`, `examples/03-transformer-block.md`, `packages/model-import/README.md` and `src/program/emit.ts` myself, ran the `tensorparser` CLI on our safetensors (output to a scratch directory, nothing written in the TEL repo), and ran the fp32/fp64 and MAC computations in section 5.
