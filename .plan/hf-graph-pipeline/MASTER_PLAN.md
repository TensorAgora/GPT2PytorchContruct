# MASTER_PLAN

Derived from `prompts/hf-weights-to-graph-json-pipeline.md` on 2026-11-20.

## 1. Objective

An offline Python pipeline that reads only the headers and small config files of every Hugging Face repo in the catalog, writes an Observed-only tensor manifest, and builds a verified viz-graph v1 JSON through a registered architecture library; plus the graph⇔TEL design with one round-trip proof.

When DONE: `python -m tools.hf_graph <huggingface_url>` writes manifest.json, report.json and a verified graph for one repo; `python -m tools.hf_graph --all` gives every catalog job a terminal status with coverage.csv, coverage_model_type.csv, near_misses.csv and category READMEs that add up to the job count; the 8 goldens rebuild from pinned headers; docs/GRAPH_TEL.md and the TEL round trip hold on every stored graph.

## 2. Principles

1. Header only: 0 payload bytes per model, counted and asserted (§3, §5.1).
2. Nothing from a repo executes (§3, D4).
3. Revisions are pinned: every manifest records its `sha` (§3).
4. The header proposes, the binding verifier decides: every stored tensor in exactly one class (§4 §7.5).
5. The manifest holds Observed facts only; architecture-dependent facts go to report.json and provenance (§5.2).
6. graph.json is viz-graph v1 plus only the §5.3 additive fields (§5.3).
7. Same (repo, sha, registry version) → byte-identical output (§3).
8. Reuse the existing Program/viz_graph code; adapt the design doc, don't copy it (§2, §4).
9. No IDE, Rust, TIP or TIS work; no TEL converter beyond the §9 proof (§3, §11).

## 3. Facts

- Catalog: `csv/categories/` 29,779 rows, 21 categories × 5 formats; 23,372 in scope (19,580 safetensors, 3,792 GGUF); `tools/split_model_catalog.py --check` passes (§2).
- Goldens: `json/*.json`, 8 files (§8.1). Graph code: `src/gpt2_pytorch/tensorprog/{program,viz,lower,qwen3_5}.py`; header parser `checkpoint/safetensors_loader.py`.
- One-model CLI `tools/model_graph.py` (qwen3_5 only, local folder); `tools/viz_to_tel.py` + `tools/tel_compile.mjs` (PASS on distilgpt2); `tools/tel_roundtrip.cjs` (untracked; mistral diff_total 0).
- JS architecture library `TensormorphDesignUI/.../viewport/graphs/arch.mjs`, configs in `.../activity_bar/shared/mock.js`.
- Gaps: no Hub fetching, no GGUF reader, no manifest, no registry/descriptors/binding, no batch, no graph schema; `gguf`, `jsonschema`, `timm`, `diffusers` not installed; `csv/goldens.csv` missing; `output/` not git-ignored.

## 4. Architecture

```text
catalog row ─05a─► discover (sha, files, gated) ─05a─► small files (cache) ─05b─► headers ─05b─► manifest.json
   manifest ─06c─► candidates (R1–R4) ─06c─► hyper (+sources) ─06a/06b─► build (descriptor × blocks on Program)
            ─06c/06e─► bind (bound|tied|ignored|aux|component) ─05c─► graph.json/graph.ref + report.json
   run_one ─05d─► ledger.jsonl, coverage*.csv, backlog ─07f─► full run, READMEs
   fallback: trace (06f, path B)        TEL: 09a design + round trip, 09b Python T♭
```

| component | location | owner |
|---|---|---|
| graph checker / comparator | `tools/graph_check.py`, `src/gpt2_pytorch/schemas/` | 00a |
| hub: catalog, http, discover | `src/gpt2_pytorch/hub/` | 05a |
| headers, manifest, fixtures | `src/gpt2_pytorch/hub/` | 05b |
| pipeline, CLI | `src/gpt2_pytorch/hub/pipeline.py`, `tools/hf_graph.py` | 05c, 05d |
| registry: descriptor, blocks, build | `src/gpt2_pytorch/registry/` | 06a, 06b |
| resolve, hyper, bind, sidecars, namemaps, trace | `src/gpt2_pytorch/registry/` | 06c, 06e, 06d, 06f |
| conformance | `tools/conformance.py` | 07a |
| TEL | `docs/GRAPH_TEL.md`, `tools/tel_roundtrip.cjs`, `tools/tel_coverage.py`, `tensorprog/tel_flat.py` | 09a, 09b |

Interfaces: manifest v1 schema (05b), viz-graph v1 schema + §5.3 fields (00a, 05c), descriptor JSON (06a), report.json (05c), ledger.jsonl (05d).

## 5. Order of execution

Master Wave 0 is split into ten modules inside plan wave 0 (dependencies inside the wave are real data flow); master Wave T is 09a (wave 0) and 09b (wave 1).

| wave | content | gate |
|---|---|---|
| 0 | 00a, 05a, 05b, 06a, 06b, 06c, 05c, 05d, 08a; 09a | goldens equal from pinned headers; D3 set; §9.5 reproduced |
| 1 | 06d, 06e, 07a; 09b | C1/C4/C6 per dense descriptor; Qwen3.8 FP8 0 unbound; dense coverage up |
| 2 | 07b | mixtral, deepseek-v3 equal; active_flops = hand count |
| 3 | 07c | qwen3.8 equal with scan annotations |
| 4 | 07d | no tensor outside a class on VLM/draft rows |
| 5 | 07e | C1, C4 per encoder descriptor |
| 6 | 06f | tiny timm/UNet bind; C2 passes |
| 7 | 07f | every job terminal; reports add up; no-op re-run |

Order follows master §7 (registry, dense, MoE, hybrid, assemblies, sparse) because each wave reuses the previous one's blocks and conformance runner.

## 6. Acceptance

In scope: R-00…R-52 (REQUIREMENTS.md). Out of scope (§11): downloading weights, dequantizing, forward passes on real weights; IDE/TIP/Rust/TensorAgora; executing repo code; the TEL converter beyond §9; graphing classic_nlp; deferred formats (D7 later wave).

| id | criterion | verified by |
|---|---|---|
| AC-1 | One URL (Qwen2.5-7B-Instruct) → three outputs passing §8.2 | 05c.C.P1.T1 |
| AC-2 | 8 goldens rebuild from pinned headers structurally equal (gemma-2 may skip) | 08a.C.P1.T1 |
| AC-3 | 100-row smoke measured, D3 set | 05d.C.P1.T1 |
| AC-4 | §9.5 reproduced (6/8, then 8/8 D3) and distilgpt2 TEL baseline PASS | 09a.C.P1.T1, 09a.C.P1.T2 |
| AC-5 | Qwen3.8 FP8: 0 unbound sidecars | 06e.C.P1.T1 |
| AC-6 | active_flops equals the K-of-E hand count | 07b.C.P1.T1 |
| AC-7 | C2 passes for every path-A descriptor with an upstream class | 06f.B.P1.T4 |
| AC-8 | Every job terminal, reports add up to the R-01 count, re-run is a no-op | 07f.C.P1.T1 |

The plan is complete when every module is DONE (or BLOCKED on something outside the plan's control)
and every criterion above holds.
