# 05a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

`csv/categories/` holds 29,779 rows in 21 categories × 5 formats (`index.csv`, `all_models.csv`). huggingface_hub 2.4.0 is installed (HfApi.model_info, hf_hub_url); its HTTP stack imports as `httpx2`, there is no `requests` or `httpx`. No hub code exists in `src/`.

## Interfaces

- `Job(repo_id, category, format, row: dict)`; `load_jobs(root='csv/categories', goldens='csv/goldens.csv') -> list[Job]`.
- `Client(concurrency=4)`: `get_range(url, start, end) -> bytes` (counts bytes), `get_small(repo, sha, path) -> bytes` (cache `~/.cache/tensormorph/hf/<repo>/<sha>/<path>`; raises on non-allowlisted path).
- `discover(job, client, revision=None) -> Discovery(sha, files, gated, weights, status)`.

## Files to change

| path | change | requirement |
|---|---|---|
| `src/gpt2_pytorch/hub/__init__.py` | new | R-01 |
| `src/gpt2_pytorch/hub/catalog.py` | new | R-01 |
| `src/gpt2_pytorch/hub/http.py` | new | R-03, R-22, R-23, R-24 |
| `src/gpt2_pytorch/hub/discover.py` | new | R-02, R-23 |
| `csv/goldens.csv` | new | R-01 |
| `tests/test_hub_http.py` | new | R-03, R-22, R-24 |
| `tests/test_discover.py` | new | R-02, R-23 |
| `tests/fixtures/model_info/` | recorded model_info JSON | R-02 |

## Approach

Catalog → jobs (repo id = path after `huggingface.co/`). Discovery calls model_info(files_metadata=True), then picks weights (§5.1): index shards; else root `*.safetensors`; diffusers: each component folder from model_index.json; GGUF: Q4_K_M else smallest, multi-part `-0000N-of-0000M` grouped (D2). pytorch/onnx/other → `skipped:deferred_format` (D7). Gated (§3): no token → `skipped:gated`; 401/403 with token → `skipped:gated_not_accepted`.
Rejected: adding `requests` (huggingface_hub's client already exists).

## Likely blockers

- Smoke-run rate limits: concurrency 4, backoff; 05d sizes the full run.
