# 06c DESIGN

## Current state

`csv/categories/<category>/metadata_spec.json` gives per dim a config string and a GGUF string in prose
(`"hidden_size | n_embd (GPT-2)"`, `"<arch>.expert_feed_forward_length (else <arch>.feed_forward_length)"`).
`tools/model_graph.py` binds qwen3_5 only: FP8 `<key>_scale_inv` named in the principal's view, `model.visual`/`mtp`
counted as unbound. distilgpt2 stores exactly its 82 keys; its attention-mask buffers bind through a slice view
`[0,0,:T,:T]` (stored [1,1,1024,1024]), so a `[`-view skips the shape check.

## Ladder

`candidates(manifest, row) -> (list[Candidate], status)`; `Candidate(descriptor, rung, evidence, score, library_class)`.

1. R1: GGUF `general.architecture` ∈ `match.gguf_architecture` (score 1.0). Mixtral GGUF says `llama`: both llama and
   mixtral are proposed; binding decides.
2. R2: `architectures` (0.9) before `model_type` (0.8), at the root and in text_config/llm_config/language_config/
   thinker_config; diffusers `_class_name` (model_index.json, component configs), timm `architecture` + `pretrained_cfg`,
   catalog `backbone_class`/`timm_module` give `library_class` candidates (path B, 06f).
3. Remote code (config `auto_map`, catalog flag `trust_remote_code`): only architectures-matched candidates stay;
   none → `skipped:remote_code`. `trust_remote_code` is never set anywhere.
4. Only when no path-A candidate exists: R3 fingerprint (share of the descriptor's scheme names, layer index → N, present
   in the header; ≥ 0.8) and R4 name hint (a descriptor model_type inside the repo id, 0.3). R3/R4 winners → `ok_assumed`.

Example: Kimi-K2 (`model_type: kimi_k2`, `architectures: [DeepseekV3ForCausalLM]`, `auto_map`) → R2 architectures →
`deepseek_v3`; the remote-code rule keeps it because it matched by architectures.

`resolve_and_bind(manifest, row) -> Resolution(status, candidate, descriptor, hyper, graph, expected, binding,
near_misses, candidates)` tries path-A candidates in rank order (scheme `gguf` for GGUF manifests, else `hf`); the first
that binds wins, every other tried candidate leaves a near-miss.

## Hyper

`resolve_hyper(d, manifest, row) -> {name: (value, source)}`. Sources per name, in order: the descriptor's explicit
sources, then the spec's parsed alternatives for the same symbol (`K` ↔ spec `topk`), then `catalog:<column>`, then the
descriptor's `=expr` defaults. Source forms: `<config key>` (root, then nested configs) → `config:<key>`;
`gguf:<key>` (`<arch>` = general.architecture; `len(x)` = array count) → `gguf:<key>`; `shape:<tensor>[i]`;
`catalog:<col>`; `missing:<format>:<tensor>` (true when that tensor is absent, e.g. tied heads); `=expr` →
`default:<expr>`. The spec parser keeps `a | b` alternatives and `(else X)` fallbacks that are dotted names or
`len(name)`; anything else (prose, `Hq * Dh`) is ignored — descriptors cover it. Never eval.

Example: Qwen2.5-7B: D = (3584, config:hidden_size), Hkv = (4, config:num_key_value_heads), Dh = (128, default:D // Hq).
`dim_sources(d, hyper, graph)` gives every graph dim a source (hyper source, `example`, or `derived:<expr>`).

## Binding

`verify(expected, manifest, d, aliases) -> Binding(ok, partition, missing, misshaped, unclaimed, views)`. Classes:
`bound` (checkpoint_key with equal shape; a `[`-slice view skips the shape check; `stack of A.{0…N}.B` views claim each
member with the per-member shape), `tied` (a stored alias key, e.g. lm_head.weight), `ignored` (descriptor `ignored`
globs), `aux` (F8 principal's `<key>_scale_inv`, named in the principal's view as model_graph.py does; 06e generalizes),
`component` (descriptor `components` prefixes, e.g. qwen3_5 `model.visual.`, `mtp.`). Anything else is `unclaimed`.
Near-miss: missing (param, key), misshaped (param, key, want, got), unclaimed keys (first 20 plus a count).

Example: distilgpt2 fixture with `transformer.h.0.attn.c_attn.weight` removed → `missing: [Wqkv_0 …]`, ok False.
