# 07e SCOPE: BERT family and T5/BART descriptors

Kind feature, wave 5, complexity M. Master sections §7 (Wave 5).
Depends on 07a; blocks 06f. Progress: [TASKS.md](TASKS.md).

## Goal

Cover encoders and encoder-decoders with post-norm, bidirectional and cross-attention blocks.

## Invariant

Decoder goldens stay equal while encoder blocks are added.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-44 | BERT-family (bert, modernbert, xlm-roberta, roberta, deberta-v2, distilbert) and T5/BART (t5, mt5, bart, marian) descriptors pass C1 and C4 | §7 (Wave 5) |

## Scope and non-goals

- R-44: BERT-family and T5/BART descriptors passing C1 and C4.

Non-goals:

- classic_nlp rows (reported only, §11)

## Write set

- `src/gpt2_pytorch/registry/blocks.py`
- `src/gpt2_pytorch/registry/descriptors/`

## Commands

Run from the repository root.

```sh
.venv/bin/python -m pytest -q                      # full suite; baseline 91 passed, 21 skipped
.venv/bin/python -m tools.conformance <descriptor>...
.venv/bin/python -m tools.hf_graph <url> | --fixture DIR | --all [--limit N] [--category C] | --coverage
```

## Deliverables and acceptance

- encoder blocks
- encoder and encoder-decoder descriptors

| criterion | verified by |
|---|---|
| C1 and C4 pass for every encoder descriptor | 07e.C.P1.T1 |

## Stop conditions

- A check needs a payload byte, a non-allowlisted file or `trust_remote_code`: stop, `block`, add a HARNESS row.
- A golden in `json/` seems wrong: never edit it; record the difference and its cause in DESIGN.md.
