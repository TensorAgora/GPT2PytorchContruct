# 09a SCOPE: Graph⇔TEL design doc, round trip over goldens, TEL coverage analyzer

Kind feature, wave 0, complexity M. Master sections §9, §7 (Wave T).
Depends on none; blocks 09b, 07f. Progress: [TASKS.md](TASKS.md).

## Goal

Write the §9 design, make tel_roundtrip.cjs run over many graphs, add the counted-exclusion analyzer, and reproduce §9.5 on the goldens.

## Invariant

TensormorphDesignUI `nodes/tel.js` is never modified; fixes live in tools/tel_roundtrip.cjs flags.

## Requirements

| id | requirement | master source |
|---|---|---|
| R-16 | docs/GRAPH_TEL.md records the §9 graph⇔TEL design (mapping, bindings, counted exclusions), and the per-graph round trip reproduces §9.5 on the 8 goldens: --const-as-buffer gives 0 differences and 6/8 pass D3; adding --fix-contract gives 8/8 with 0 differences, compile clean and D3 pass | §1, §9 |
| R-49 | A TEL coverage analyzer counts per graph the T♭-spellable ops and each §9.4 counted exclusion | §7 (Wave T), §9.4 |
| R-50 | `tools/viz_to_tel.py` → `tools/tel_compile.mjs` on distilgpt2 still prints PASS | §9.6 |

## Scope and non-goals

- R-16: docs/GRAPH_TEL.md and the §9.5 reproduction on the 8 goldens.
- R-49: tools/tel_coverage.py counted-exclusion analyzer.
- R-50: viz_to_tel → tel_compile PASS on distilgpt2.

Non-goals:

- The batch TEL converter (next stage)
- Editing TensormorphDesignUI/nodes/tel.js

## Write set

- `docs/GRAPH_TEL.md`
- `tools/tel_roundtrip.cjs`
- `tools/tel_coverage.py`

## Commands

Run from the repository root.

```sh
node tools/tel_roundtrip.cjs json/*.json --const-as-buffer [--fix-contract]
.venv/bin/python -m tools.tel_coverage json/*.json
.venv/bin/python -m tools.viz_to_tel --graph json/distilgpt2.json --out $TMPDIR/d.tel && node tools/tel_compile.mjs $TMPDIR/d.tel artifacts/tel/model.tel
```

## Deliverables and acceptance

- docs/GRAPH_TEL.md
- tools/tel_roundtrip.cjs (multi-graph)
- tools/tel_coverage.py

| criterion | verified by |
|---|---|
| §9.5 reproduced: 6/8 then 8/8 D3 with 0 differences | 09a.C.P1.T1 |
| distilgpt2 viz_to_tel baseline prints PASS | 09a.C.P1.T2 |
| Exclusion counts agree with tel_roundtrip | 09a.B.P1.T2 |

## Stop conditions

- The TEL dist (`tensormorph-source/.../packages/tel/dist`) or TensormorphDesignUI nodes are missing or changed: record the commit seen in HARNESS.md and stop.
- §9.5 numbers do not reproduce: record the observed numbers; never adjust the flags to fit.
