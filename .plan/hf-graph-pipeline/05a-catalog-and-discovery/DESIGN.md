# 05a DESIGN

## Current state

`csv/categories/<category>/<format>.csv` (format = file stem: safetensors, gguf, pytorch, onnx, other) holds 29,779 rows;
`all_models.csv` is the merge. huggingface_hub 2.3.0 (lock): `HfApi.model_info`, `hf_hub_url`, `get_session()`
(an `httpx2.Client`), `utils.build_hf_headers`. Of the 8 goldens only `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` is not
in the catalog.

## Job

`Job(repo_id, category, format, row: dict, source: "catalog"|"golden")`, frozen dataclass.
`load_jobs(root="csv/categories", goldens="csv/goldens.csv") -> list[Job]`: every per-format CSV row (repo id = the
`huggingface_model_id` column, else the URL path), sorted by (category, format, repo_id); then goldens.csv rows whose repo
is not already present. `csv/goldens.csv` columns: `repo,golden,category,format,revision`.

## Statuses

Terminal statuses (§5.4): `ok`, `ok_assumed`, `partial`, `no_graph`, `skipped:gated`, `skipped:gated_not_accepted`,
`skipped:no_weights`, `skipped:deferred_format`, `skipped:remote_code`, `failed:<code>`. 05a produces the `skipped:*` ones
except remote_code (06c) and `failed:http_<status>` / `failed:network` after retries are exhausted.

## HTTP client

`hub/http.py`, one `Client` per process, thread-safe.

- `Client(concurrency=4, cache_dir=None, transport=None, max_retries=6)`. `cache_dir` defaults to `$TM_HF_CACHE` or
  `~/.cache/tensormorph/hf`. `transport(method, url, headers) -> Response(status, headers, body)` defaults to
  huggingface_hub's `get_session()` with redirects followed; tests pass a fake.
- `get(url, headers=None) -> Response`: holds a semaphore slot; retries 429 and 5xx with exponential backoff
  (1 s doubling, cap 60 s), sleeping `Retry-After` (seconds or HTTP date) when present; counts `requests`.
- `get_range(url, start, end) -> bytes`: `Range: bytes=start-end`; accepts 206 (or 200 only if the body is no longer than
  asked); adds `len(body)` to `bytes_read`; raises `HttpError(status)` on 401/403/404.
- `get_small(repo, sha, path) -> bytes`: allowlist first (`ALLOWED`: `config.json`, `*.safetensors.index.json`,
  `model_index.json`, `<component>/config.json`, `params.json`; anything else raises `NotAllowed`), then the on-disk
  cache `<cache_dir>/<repo>/<sha>/<path>`, else GET `hf_hub_url(repo, path, revision=sha)`.
- `model_info(repo, revision=None) -> dict`: `HfApi.model_info(files_metadata=True)` through the same retry wrapper,
  cached as JSON under `<cache_dir>/<repo>/_info/<revision or main>.json` (`refresh=True` re-queries).
- Token: `HF_TOKEN` from the environment only, sent as the `Authorization` header via `build_hf_headers`; a log filter
  and `HttpError.__str__` replace it with `***`. Nothing else about the user is sent (no custom user agent fields).

## File selection

`discover(job, client, revision=None) -> Discovery(sha, files, gated, weights, variants, status)`:

1. `job.format` in pytorch/onnx/other, or category `classic_nlp`: `skipped:deferred_format`, no request (D7).
2. `model_info` → `sha`, `files` = siblings `[{path, size}]`, `gated` (False/"auto"/"manual").
3. Gated and no `HF_TOKEN` → `skipped:gated`. Gated with a token: probe `get_range(<first weight>, 0, 7)`; 401/403 →
   `skipped:gated_not_accepted`.
4. Weights (§5.1): root `*.safetensors.index.json` → its `weight_map` shards; else root `*.safetensors`; diffusers
   (`model_index.json`): per component folder its index shards or its `*.safetensors` (plain name preferred over a
   `.fp16`/variant one). GGUF: group `-0000N-of-0000M.gguf` parts into one logical file, skip `mmproj*` (listed as
   components), take the one whose name has `Q4_K_M` (case-insensitive), else the smallest; the rest go to `variants` (D2).
5. No weight file → `skipped:no_weights`. Otherwise `status = None` (continue to headers).

Smoke: `python -m gpt2_pytorch.hub.discover --smoke N` takes N jobs round-robin over (category, format) in sorted
order, discovers each and fetches its small files, then prints status counts and `network requests: <n>`.
