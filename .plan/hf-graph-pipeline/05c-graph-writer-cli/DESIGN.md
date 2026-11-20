# 05c DESIGN

## Stages → functions

| stage | function | side exit → status |
|---|---|---|
| S1 discover | `hub.discover.discover` | deferred format / no weights / gated → `skipped:deferred_format|no_weights|gated|gated_not_accepted`; HTTP → `failed:http_<n>` |
| S2 small files | `hub.discover.small_files` | — |
| S3 headers | `hub.headers.read_header` | `failed:truncated|corrupt|over_budget|unsupported_version` |
| S4 manifest | `hub.manifest.build_manifest` (S1–S4: `read_manifest`) | payload bytes → assertion (never written) |
| S5 sidecars | `registry.bind.verify` aux class (06e: `sidecars.py`) | — |
| S6 ladder | `registry.resolve.candidates` | remote code → `skipped:remote_code`; none → `no_graph` + inventory/prefix tree |
| S7 hyper | `registry.hyper.resolve_hyper` | missing source → near-miss, next candidate |
| S8 binding | `registry.bind.verify` | fails → near-miss, next candidate; none left → `no_graph` |
| S9 build | `registry.build.build` (path A; 06f path B) | — |
| S10 write | `hub.pipeline.run_one` | components declared without graph → `partial`; R3/R4 winner → `ok_assumed` |
| S11 validate | `tools.graph_check` (schema, invariants, `--manifest`, `--against`) | — |
| S12 ledger | 05d | — |

## report.json

`{format: tensormorph-hf-graph-report, version 1, repo, category, weights_format, sha, status, message?, bytes
{header_bytes_read, payload_bytes_read 0, requests}, candidates[], near_misses[], inventory? (no_graph), winner
{descriptor, rung, evidence, path}, hyper {name: {value, source}}, dims {dim: {value, source}}, binding {class: count,
unclaimed}, components[], sidecars[], content_key, graph, totals}`. No timestamps or durations (ledger holds those).

## Graph storage (D9)

The shared graph is the built graph with per-repo facts removed (model name and root group label = descriptor id;
provenance {certainty, evidence, origin, descriptor, path}). **Content key = sha256 of its canonical bytes** (first 24
hex): identical graphs share one file by construction, which also covers config-driven layer selectors that a
(descriptor, hyper, scheme) key would miss. Every repo writes `graph.ref` = `{content_key, model, overrides {tensor:
{view}}, checkpoint, components, provenance {certainty, evidence, rung, hyper_sources}}`; `load_graph(folder)` overlays it.
A full per-repo `graph.json` is read when present but not written (no override needs more than a ref).

## CLI

`python -m tools.hf_graph <url> [--revision R] [--out output/categories]`, `--fixture DIR` (offline; job from the catalog
when the repo is in it), `--materialize DIR [--to FILE]` (full graph for graph_check and viewers).
