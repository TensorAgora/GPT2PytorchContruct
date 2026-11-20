# HF weights → tensor manifest → graph JSON pipeline

This plan builds an offline Python pipeline that takes every Hugging Face URL in the model catalog, reads only safetensors/GGUF *headers* plus small config files into an Observed-only tensor manifest, and turns it into a `tensormorph-viz-graph` v1 computation graph through a registered architecture library whose binding verifier decides every candidate. It also writes the graph⇔TEL design and proves the round trip on the 8 goldens, so TensorAgora can later open any HF URL and draw its graph from header metadata.

Plan version 2026-11-21.1, from `prompts/hf-weights-to-graph-json-pipeline.md`. 20 modules, 53
requirements, 109 tasks, 8 waves. Stages per module: A Design, B Build, C Integrate.

## Files

| file | kind | read when |
|---|---|---|
| [CONTINUE.md](CONTINUE.md) | authored | start of every session |
| [HARNESS.md](HARNESS.md) | append-only | before every task |
| [STATUS.md](STATUS.md) | generated | start of every session |
| [MASTER_PLAN.md](MASTER_PLAN.md) | authored | first session, or when scope is unclear |
| [DECISIONS.md](DECISIONS.md) | authored | before changing scope or design |
| [REQUIREMENTS.md](REQUIREMENTS.md), [EXECUTION_GRAPH.md](EXECUTION_GRAPH.md) | generated | planning reviews only |
| `<id>-<slug>/SCOPE.md`, `DESIGN.md` | authored | working on that module |
| `<id>-<slug>/TASKS.md` | generated | working on that module |

Asynchronous plans (spec has an `execution` block) also have `ORCHESTRATION.md`, `WORKER_PROMPT.md`
and `QUEUE.md`.

## Tool

```sh
python3 .plan/hf-graph-pipeline/tools/plan_tool.py next                      # what to do now
python3 .plan/hf-graph-pipeline/tools/plan_tool.py complete-task <id> --reason '<one-line status>'
python3 .plan/hf-graph-pipeline/tools/plan_tool.py complete-stage <module> <stage> --reason '<one-line status>'
python3 .plan/hf-graph-pipeline/tools/plan_tool.py block <module> --reason '<cause>' --unblock '<condition>'
python3 .plan/hf-graph-pipeline/tools/plan_tool.py harness <id> --what '<what failed>' --instead '<what works>'
python3 .plan/hf-graph-pipeline/tools/plan_tool.py render | validate
```

`tools/plan-spec.json` is the structure (edit it, then `render`); `tools/plan-state.json` is progress
(change it only through the tool). Generated files are never edited by hand.
