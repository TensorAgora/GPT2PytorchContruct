# Continue: HF weights → tensor manifest → graph JSON pipeline

Restart-safe prompt. Every run continues from recorded state; never start over.

## Start of session

1. `pwd` (repository root), `git status`, `git log --oneline -10`: keep existing work.
2. Read `.plan/hf-graph-pipeline/HARNESS.md`. Do not repeat anything it lists.
3. Read `.plan/hf-graph-pipeline/STATUS.md` and run `python3 .plan/hf-graph-pipeline/tools/plan_tool.py next`.
4. Open only the active module's `SCOPE.md`, `DESIGN.md` and `TASKS.md`. Read MASTER_PLAN.md or
   DECISIONS.md only when the task needs them.

If the spec has an `execution` block, do not implement here: follow "Asynchronous" below.

## Project invariants

- Header only: never read a payload byte; besides header ranges fetch only the §3 small files (allowlist in `hub/http.py`).
- Never execute repository code: no `trust_remote_code`, no `modeling_*.py`; path B uses library classes on the `meta` device only.
- Reuse `Program`, `viz_graph()`, `qwen3_5.py`, `lower.py`, `read_safetensors_header`; never re-implement them.
- `json/*.json` are oracles: never edit them; `json/distilgpt2.json` must stay byte-identical.
- Never modify TensormorphDesignUI or tensormorph-source (`nodes/tel.js` included).
- `HF_TOKEN` is never written to logs, manifests, reports, ledger or fixtures.
- Python via `.venv/bin/python` or `uv` (never conda); `uv sync --inexact`; new deps only in the `hub`/`trace` extras.
- Offline tests run from `tests/fixtures/`; only tasks marked `[network]` touch the Hub.
- All outputs are canonical JSON; `fetched_at` is the only volatile field.
- `output/` is never committed; fixtures are. Commit subjects are one short line.

## Loop

One task at a time, in the order `next` gives:

1. Do the task inside the module's write set, following DESIGN.md.
2. Run the task's `verify` check. Fix until it passes.
3. Record a one-line status (no report, no log file):
   `python3 .plan/hf-graph-pipeline/tools/plan_tool.py complete-task <task-id> --reason '<what changed; check result>'`
4. Commit by explicit path (never `git add -A`) with a short subject.

When every task of a stage is closed: `complete-stage <module> <stage> --reason '<one line>'`.

Something failed, errored, did not work, or turned out to be the wrong approach? Add it to HARNESS.md
before trying something else:
`python3 .plan/hf-graph-pipeline/tools/plan_tool.py harness <id> --what '<what and the error>' --instead '<what works>'`.
Truly blocked: `block <module> --reason '<cause>' --unblock '<condition>'`, then continue with other
legal work.

**One module per session.** When the module is DONE or BLOCKED, stop. The next session starts from
recorded state, not from memory.

## Rules

- The master document wins over the plan; record a disagreement in DECISIONS.md and fix the spec.
- A task is done because its check passed, not because code exists.
- Never weaken an acceptance criterion to pass it; change it only with a DECISIONS entry.
- Spec edits: append tasks, never reorder or delete recorded ones; then `render` and `validate`.

## Asynchronous

When the spec has an `execution` block, this session is the orchestrator:

1. Read `.plan/hf-graph-pipeline/ORCHESTRATION.md`; run `tmux list-sessions`.
2. `python3 .plan/hf-graph-pipeline/tools/plan_tool.py validate`, then `python3 .plan/hf-graph-pipeline/tools/orchestrate.py up`
   (reuse an existing supervisor; never start a second one).
3. Check now and then with `orchestrate.py status` and `plan_tool.py queue`. For a BLOCKED module, read
   HARNESS.md and the newest logs in `tools/_runs/`, fix the cause or ask the user, then `unblock`.
4. Commit by explicit path at wave gates. Stop with `orchestrate.py down`. Never touch tmux sessions
   this plan did not create.
