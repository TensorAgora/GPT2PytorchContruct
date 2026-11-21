# DECISIONS

Superseded and open items are kept, never deleted.

## Decisions

| id | decision | reason |
|---|---|---|
| D-01 | Master IDs R-00…R-16 kept verbatim; R-17…R-52 derived from must-statements in §2, §3, §5, §7 wave exits, §8, §9 and §10, one owner each | the master's IDs are compound and later waves would otherwise own nothing |
| D-02 | Sequential mode (no `execution` block) | the user did not ask for parallel or autonomous execution; add an `execution` block to switch |
| D-03 | Master Wave 0 → ten modules in plan wave 0; Wave T → 09a (wave 0) and 09b (wave 1) | keeps the master's wave numbers and gates; same-wave dependencies are real data flow |
| D-04 | Layout: fetch code in `src/gpt2_pytorch/hub/`, registry in `src/gpt2_pytorch/registry/` (D1) with one `blocks.py`, `descriptors/*.json`, `namemaps/`; CLI `tools/hf_graph.py`; schemas in `src/gpt2_pytorch/schemas/` | the master allows simplifying its layout; fewer files |
| D-05 | Golden extras live in `csv/goldens.csv` | the master names no folder; outside `csv/categories/` so `split_model_catalog.py --check` is unaffected |
| D-06 | `tools/graph_check.py` (schema, §8.2 invariants, §8.1 comparator) is built first (00a) | every later verify uses it |
| D-07 | 06a/06b prove the ports on spelled configs (R-29…R-31); 08a proves R-11 on pinned real headers | separates porting errors from real-config deltas |
| D-08 | Safetensors: own 2-range reader on `hf_hub_url`, cross-checked once against `get_safetensors_metadata` (final call in 05b.A) | the hub helpers exist in 2.4.0 but expose no bytes-read count and no component subfolders |
| D-09 | `jsonschema` added to the `hub` extra | a hand-rolled validator is worse |
| D-10 | C4 without a golden uses hand counts stored in descriptor `provenance.c4` | encoders and new keys have no golden json |
| D-11 | Master defaults D1–D9 and T-1–T-3 adopted as written | §10: defaults hold unless the owner overrides |
| D-12 | Provisional GGUF header budget 64 MiB until 05d.C.P1.T1 sets D3 | the master measures before fixing it |
| D-13 | Blocks set `axes[].symbol` for B and T (R-52); the §8.1 comparator ignores symbols | arch.mjs goldens have null symbols; distilgpt2 already has them |
| D-15 | Fixtures store header JSON (safetensors header dict; GGUF KV with arrays summarized plus tensor infos), not raw header bytes; the GGUF byte reader is tested on a synthetic header written to tmp_path | owner storage rule: keep only config, metadata and JSON |
| D-16 | The §8.1 comparator takes `--ignore view` for header-derived views (qwen3.8 FP8) when building from a spelled config; 08a checks full equality from headers | spelled configs cannot name `*_scale_inv` tensors |
| D-17 | distilgpt2 byte identity is checked on the builder path (06a); the pipeline path is structural like the other goldens | the pipeline adds §5.3 fields |
| D-18 | **D3 header byte budget: 64 MiB per GGUF file, 256 MiB per model** (supersedes D-12). Measured on the 100-row smoke (05d.C, 2026-11-21): header bytes per model p50 137,592, p95 14,770,403, max 47,100,000 (a 281-shard safetensors repo); max single GGUF header 15.8 MB. Throughput: 9.5 rows/min at 4 workers with serial shard reads and cold discovery; 111 rows/min at 8 workers (16 requests in flight) with shard headers read in parallel and discovery cached. Full catalog estimate: about 3.5-5 h at 8 workers | 4-5x headroom over the measured maxima; budget exits are `failed:over_budget`, never truncation |
| D-19 | Storage rule applied to outputs: manifests and shared graphs are written as deterministic `.json.gz` (mtime 0), cache entries over 32 KiB are gzipped, and the shard index is not copied into `manifest.config` (§5.2 lists only config.json, model_index.json and component configs; `tensors[].file` already holds the shard map). Projection from the 100-row smoke: 170 MB raw for 54 fetched rows, i.e. about 70 GB for the catalog, down to about 3.5 GB (150 KB per fetched row) | the owner asked to keep only small metadata and JSON; readers accept either form |
| D-20 | A registry version (sha256 of descriptors, registry code and the lowerings, 12 hex) is recorded in report.json, ledger lines and shared-graph provenance. Resume re-runs transient failures (`failed:http_0/429/5xx`, `failed:exception_*`) and re-runs `no_graph`, `ok_assumed` and `skipped:remote_code` rows when the registry changed; a repo whose manifest exists at the discovered sha is re-resolved offline from it | §3 determinism is per (repo, sha, registry version); coverage must rise when descriptors are added, without re-reading headers |
| D-21 | Every `transformers` import of the pipeline lives in `src/gpt2_pytorch/registry/trace.py` (path B), the one file `tests/test_model.py::test_src_never_imports_transformers` exempts; the reference model and path A stay free of it | the repo keeps its pure-PyTorch invariant; path B needs the upstream classes by design (master §3, §6) |
| D-22 | Stored tensors that are not learned weights of the drawn model are `ignored`, never dropped: GGUF `rope_freqs` / `rope_factors_*` (RoPE scaling tables), `cls.output.weight` (the score head of reranker conversions; the graph is the backbone) and GPT-2's old `attn.masked_bias`. Modern quant forms (compressed-tensors FP8/INT8/NVFP4/pack-quantized, ModelOpt FP8/NVFP4, GPTQ zero biases) are sidecars of their logical principal | binding stays total (every stored tensor has a class) and the report names what is not drawn |
| D-23 | report.json `sidecars` holds counts (`count`, `by_scheme`, `file`); the per-tensor records (R-07: every binding) go to `sidecars.json.gz` beside it | one record per sidecar tensor made a quantized MoE report 33 MB (Kimi-K2) and reports 585 MB of 2.2 GB; the records stay, gzipped, in the repo folder |
| D-24 | A GGUF file's `general.architecture` (R1) candidates survive a remote-code flag; only config candidates are dropped | llama.cpp runs the file as that architecture, never the repo's Python; the catalog flag describes the original HF repo (GGUF repos ship no config) |
| D-14 | Stage A/B verifies are offline; network checks are marked `[network]` and sit in stage C (or capture fixtures) | sessions can work without the Hub |

## Open questions

| id | question | blocks | default taken |
|---|---|---|---|
| Q-01 | Which repo is the mixtral golden? | 05a.B.P1.T1, 08a | RESOLVED 2026-11-21 (B-02): `mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF` |
| Q-02 | TEL store name for an HF repo and how a sharded repo maps to one store (ADR-0027, bind.ts) | 09a.A | settled in 09a.A from the ADR; nothing else waits on it |
| Q-03 | Where is the local Qwen3.8-27B-FP8 sample the master mentions? | 06b, 06e | not found at planning; fetch config.json and the header fixture from the Hub |
| Q-04 | D8: v1 + annotations or v2 route/scan primitives? (owner) | 07b, 07c | v1 + `execution` annotations |
| Q-05 | T-2: printer in tel.js or a Python port? (owner) | 09b | Python port |
| Q-06 | What is the Wave T "coverage analyzer"? | 09a | TEL counted-exclusion analyzer (R-49), not R-15 |
| Q-07 | Gemma-2 needs an accepted HF_TOKEN for its fixture | 08a | skip with a message |
| Q-08 | Re-running whole categories in 07a/07b/07c/07d/07e.C may take hours | those C tasks | size by D3 throughput; `--limit` with a stratified sample if a run exceeds ~2 h, recorded in DECISIONS |

## Risks

| id | risk | mitigation (module task) |
|---|---|---|
| K-01 | Hub rate limits make the full run slow or fail | 05a.B.P2.T1 backoff; 05d.C.P1.T1 measures; 07f resumes via the ledger |
| K-02 | Real configs differ from spelled configs, goldens not reproducible | 08a.B.P1.T2 lists every delta with its cause |
| K-03 | metadata_spec.json prose is not machine-readable | 06c.B.P1.T2 parses alternation only; descriptors carry the rest |
| K-04 | Path B traces fail on data-dependent MoE | 06f keeps MoE on path A; failed:trace_<code> |
| K-05 | Graph storage grows to GBs | content-addressed store, 05c.B.P1.T2 (D9) |
| K-06 | Upstream TensormorphDesignUI/TEL changes break tel_roundtrip | 09a records the commits it ran against in docs/GRAPH_TEL.md |

## During implementation

- B-02 (2026-11-21, 05b.C, plan v2026-11-21.3): the mixtral golden repo is `mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF`. Its Q4_K_M file stores stacked `blk.N.ffn_{gate,up,down}_exps.weight` (323 tensors, the golden's 323 keyed parameters). TheBloke's and MaziyarPanahi's/second-state's Dec-2023 files store per-expert `blk.N.ffn_gate.E.weight` (995 tensors); TheBloke's fixture is kept as the per-expert storage form for 07b.

- B-01 (2026-11-21, plan v2026-11-21.2): R-08 adds diffusers/timm/catalog R2 evidence; R-19 and R-25 follow the storage rule; tasks 00a.B.P1.T1, 00a.B.P2.T3, 05b.B.P1.T2, 05b.B.P2.T1, 06b.B.P1.T3, 06c.B.P1.T1, 08a.B.P1.T3, 08a.C.P1.T1 revised before any state was recorded.

Append B-01, B-02, ... with module, date and the spec change caused. Failures and dead ends go to
HARNESS.md, not here.
