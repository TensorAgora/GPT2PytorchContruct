# 06c SCOPE: Resolution ladder, hyperparameter sources, binding verifier

Kind foundation, wave 0, complexity L. Master sections §3, §4, §6, D4.
Depends on 05b, 06b; blocks 05c, 06e. Progress: [TASKS.md](TASKS.md).

## Goal

Turn a manifest into ranked descriptor candidates, resolved dims with sources, and a binding partition that accepts or rejects each candidate with a near-miss report.

## Invariant

A candidate is accepted only when every stored tensor falls in exactly one binding class.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-08 | The ladder R1 (GGUF general.architecture), R2 (config architectures, then model_type, also nested in text_config/llm_config/language_config/thinker_config), R3 (tensor-name fingerprint), R4 (name hint) proposes ranked candidates; Kimi-K2 resolves to the deepseek_v3 descriptor | §1, §4 (§6) |
| R-09 | Hyperparameters resolve from `metadata_spec.json` expressions, each recorded with its source (config key, GGUF key, shape, catalog, default); every dim in a graph has a recorded source in report.json | §1, §2 |
| R-10 | The binding verifier puts every stored tensor in exactly one class (bound, tied, ignored, aux, component) or fails the candidate with a near-miss report naming the tensor | §1, §4 (§7.5) |
| R-26 | No repository code executes: trust_remote_code is never set, and remote-code repos bind only through a descriptor matched by `architectures`, else skipped:remote_code | §3, D4 |

## Scope and non-goals

- R-08: ladder R1–R4 with ranked candidates and fall-through.
- R-09: hyper resolution with a source per dim.
- R-10: binding verifier with near-miss reports.
- R-26: remote-code repos bind only via `architectures`; never trust_remote_code.

Non-goals:

- Sidecar rules (06e)
- Path B fallback (06f)

## Write set

- `src/gpt2_pytorch/registry/resolve.py`
- `src/gpt2_pytorch/registry/hyper.py`
- `src/gpt2_pytorch/registry/bind.py`
- `tests/test_resolve.py`
- `tests/test_hyper.py`
- `tests/test_bind.py`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m gpt2_pytorch.registry.resolve tests/fixtures/headers/<dir>
```

## Deliverables and acceptance

- src/gpt2_pytorch/registry/resolve.py
- src/gpt2_pytorch/registry/hyper.py
- src/gpt2_pytorch/registry/bind.py

| criterion | verified by |
|---|---|
| Kimi-K2 resolves to deepseek_v3; remote-code-only repos are skipped:remote_code | 06c.B.P1.T1 |
| Every dim of the fixture graphs has a recorded source | 06c.C.P1.T1 |
| A removed or misshaped tensor fails binding and is named | 06c.B.P1.T3 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
- A metadata_spec.json string seems to need `eval`: never; move that dim into the descriptor's hyperparameters.
