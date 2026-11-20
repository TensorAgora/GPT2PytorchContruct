# 07a DESIGN

Drafted at plan time; completed in Stage A.

## Current state

After Wave 0: descriptors gpt2, llama, qwen2, mistral, phi3, gemma2, mixtral, deepseek_v3, qwen3_5; backlog in output/categories/descriptor_backlog.csv.

## Interfaces

- `python -m tools.conformance ID... [--c2]` prints `ID C1 PASS C3 PASS …`; exit 1 on any FAIL.
- Descriptor `provenance.c4` holds hand counts where no golden exists.

## Files to change

| path | change | requirement |
|---|---|---|
| `tools/conformance.py` | new | R-38 |
| `src/gpt2_pytorch/registry/descriptors/` | qwen3, hf_olmo, gemma3_text, … | R-39 |
| `tests/fixtures/headers/` | one fixture per descriptor | R-39 |

## Approach

Conformance uses fixtures only (offline). New descriptors extend existing ones (match + small recipe flags) rather than copy them.

## Likely blockers

none
