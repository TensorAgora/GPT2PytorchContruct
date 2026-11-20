# HARNESS: what failed, what does not work, what not to do

Read this before every task. Each row is something that already failed, errored, did not work or is
not appropriate in this repository, and what to do instead. Do not retry a row's approach unless the
row says it is resolved.

Add a row the moment you hit one (rows from `verify-fail`, `block` and failed releases are added
automatically):

```sh
python3 .plan/hf-graph-pipeline/tools/plan_tool.py harness <task-or-module-id> --what '<what failed and the error, one line>' --instead '<what works instead>'
```

Never delete a row. When a problem is fixed, add a new row for the same id saying `resolved: <how>`.

## Known at planning time

- `TensormorphDesignUI/tensormorph/ui/activity_bar/viewport/shared/mock.js` does not exist: the file is `.../activity_bar/shared/mock.js`.
- `csv/goldens.csv` does not exist yet: 05a.B.P1.T1 creates it.
- Not in `.venv`: `gguf`, `jsonschema`, `timm`, `diffusers`, `requests`, `httpx` (huggingface_hub 2.4.0's client imports as `httpx2`). Until 00a adds the `hub` extra, gguf works only via `uv run --with gguf` (0.19.0).
- `uv sync` is exact: it removes transformers (in the `reference` extra), which `tests/test_qwen3_5_lowering.py` needs. Use `uv sync --inexact` or name every extra. Never conda.
- Storage rule (owner, 2026-11-21): no weight or large file is kept in the repo. All *.safetensors and *.bin weights were deleted (about 1 GB); only their header JSON remains (`weights/distillgpt2/model.safetensors.header.json`, `artifacts/model.safetensors.header.json`, `artifacts/trace/sample_batch.safetensors.header.json`). Fixtures are small JSON only; never write weight bytes, never download a payload to test.
- No local Qwen3.8-27B-FP8 weights exist and no distilgpt2 config.json is local. Fetch config.json and headers from the Hub.
- `json/distilgpt2.json` is compact JS `JSON.stringify` output (JS number formatting, e.g. `0.00001` not `1e-05`): Python `json.dumps` of the same data is 38 bytes off. Byte identity (06a) needs a JS-compatible float formatter or `node` re-serialization.
- `artifacts/models/qwen3.8-27b-fp8/viz_graph.json` (9.4 MB) is not the golden; the golden is `json/qwen3.8-27b-fp8.json` (9.2 MB).
- `tools/tel_roundtrip.cjs` and the master document are untracked: commit them with the first change that touches them.
- `metadata_spec.json` dims are prose (`"hidden_size | n_embd (GPT-2)"`): parse, never `eval`.
- Forbidden by the master: `GGUFReader` (maps whole files), `trust_remote_code`, payload reads, editing `json/*.json`, editing `nodes/tel.js`.
- `google/gemma-2-2b-it` is gated: its fixture test skips with a message, never blocks Wave 0.
- Run commands from the repository root: `python -m tools.X` and pytest (`pythonpath = src, .`) depend on it.
- Baselines at planning (2026-11-21, after the weight deletion): `pytest -q` 91 passed, 21 skipped; viz_to_tel → tel_compile on distilgpt2 PASS; `uv run --with gguf python tools/split_model_catalog.py --check` ok (29,779); `node tools/tel_roundtrip.cjs json/mistral-7b-instruct-v0.3.json --const-as-buffer` diff_total 0, D3 pass.

## Log

| date | id | what failed / not to do | instead |
|---|---|---|---|
| 2026-11-20 | 00a | master §2 path TensormorphDesignUI/.../activity_bar/viewport/shared/mock.js does not exist | use .../activity_bar/shared/mock.js |
| 2026-11-20 | 05a.C.P1.T1 | repos with component folders (vae/config.json + safetensors) but no model_index.json and no root safetensors were skipped:no_weights | discover falls back to one-level component folders that have config.json (fixed, test_component_folders_without_model_index) |
| 2026-11-20 | 05d.C.P1.T1 | first smoke: 9.5 rows/min, 497 s for one 281-shard repo; shard headers were read serially and per-row request counts were inflated by a shared client counter | read shard headers in parallel (read_manifest pool of 8) and give each job Client.scoped() counters: 111 rows/min |
| 2026-11-20 | 05d.C.P1.T1 | GGUF pick took an mtp draft file as smallest quant; unknown ggml types reported as corrupt | mtp*/-mtp files are components like mmproj; unknown ggml type -> failed:unsupported_type |
