# 05a SCOPE: Catalog jobs, HTTP client, repo discovery and small-file cache

Kind feature, wave 0, complexity L. Master sections §2, §3, §5.1, D2, D7.
Depends on 00a; blocks 05b, 05c. Progress: [TASKS.md](TASKS.md).

## Goal

Turn catalog rows into jobs and discover each repo (sha, files, gated, weight files) through one rate-limited, token-safe HTTP client with a (repo, sha, path) small-file cache.

## Invariant

No request leaves the client without passing the §3 allowlist, and HF_TOKEN never appears in any output.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-01 | Every row of `csv/categories/<category>/<format>.csv` becomes a job (huggingface_url → repo id), plus golden repos missing from the catalog from `csv/goldens.csv`; job count = 29,779 + golden extras (1 today) | §1, §2, §8.1 |
| R-02 | Discovery records each repo's pinned `sha`, file list with sizes, `gated`, and the weight files to read per §5.1 (index shards, else root safetensors, diffusers component folders, one GGUF per D2 with multi-part as one file), or a terminal `skipped:*` status | §1, §5.1, D2, D7 |
| R-03 | Small files are fetched through `hf_hub_url(repo, path, revision=sha)` and cached by (repo, sha, path); a second run makes 0 network requests for cached rows | §1, §5.1 |
| R-22 | Besides header ranges, only config.json, *.safetensors.index.json, model_index.json, <component>/config.json, the timm config.json and params.json are ever fetched | §3 |
| R-23 | Gated repos get skipped:gated without HF_TOKEN and skipped:gated_not_accepted when the token lacks accepted terms; the token never appears in logs, manifests or reports, and nothing about the user beyond it is sent | §3 |
| R-24 | HTTP access uses bounded concurrency, exponential backoff on 429 and 5xx, and honours Retry-After | §3 |

## Scope and non-goals

- R-01: `load_jobs()` over the catalog plus `csv/goldens.csv` extras.
- R-02: `discover()` with §5.1/D2/D7 file selection and terminal skips.
- R-03: small-file cache keyed (repo, sha, path).
- R-22: allowlist in `get_small`.
- R-23: gated statuses and token redaction.
- R-24: concurrency, backoff, Retry-After.

Non-goals:

- Header parsing (05b)
- Batch ledger (05d)

## Write set

- `src/gpt2_pytorch/hub/__init__.py`
- `src/gpt2_pytorch/hub/catalog.py`
- `src/gpt2_pytorch/hub/http.py`
- `src/gpt2_pytorch/hub/discover.py`
- `csv/goldens.csv`
- `tests/test_hub_http.py`
- `tests/test_discover.py`
- `tests/fixtures/model_info/`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -c "from gpt2_pytorch.hub.catalog import load_jobs; print(len(load_jobs()))"
.venv/bin/python -m gpt2_pytorch.hub.discover --smoke 100      # [network]
```

## Deliverables and acceptance

- src/gpt2_pytorch/hub/catalog.py
- src/gpt2_pytorch/hub/http.py
- src/gpt2_pytorch/hub/discover.py
- csv/goldens.csv

| criterion | verified by |
|---|---|
| Job count is 29,780 (29,779 + 1 golden extra) | 05a.B.P1.T2 |
| On 100 rows every row has a sha or a terminal skipped:*; the cached re-run makes 0 requests | 05a.C.P1.T1 |
| Backoff, Retry-After and token redaction hold on the fake transport | 05a.B.P2.T1 |
| A file outside the §3 allowlist cannot be fetched | 05a.B.P2.T2 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- The Hub keeps answering 429 after backoff: stop the smoke run, record throughput seen in HARNESS.md.
