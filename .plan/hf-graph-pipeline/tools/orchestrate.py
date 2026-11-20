#!/usr/bin/env python3
"""Run a plan asynchronously with a pool of tmux worker sessions.

Standard library only. Copied into every plan as <plan>/tools/orchestrate.py next
to plan_tool.py; it needs the spec's "execution" block.

Model:

  supervisor  one tmux session (<prefix>-sup) that watches the work queue and
              spawns worker sessions on demand, up to execution.max_workers
  worker      one tmux session (<prefix>-wNN) running a loop: claim the next
              item (verification first, then stage closing, rework, new
              implementation), run ONE fresh headless agent for it, check that
              the agent recorded the result, claim again. An implementer that
              submits a task moves straight on to the next task; another worker
              verifies the submitted one.

Every agent run starts with no conversation context: the prompt (rendered from
WORKER_PROMPT.md) carries the task, its verify check, the write set and any
rework findings. State changes go only through plan_tool.py, which locks.

Commands:

  up        [--workers N]     preflight, then start the supervisor session
  supervise                   the supervisor loop (runs inside tmux)
  spawn     [--count N]       start worker sessions now
  work      --worker NAME     the worker loop (runs inside tmux)
  status                      sessions, queue and the tail of each pane
  logs      NAME [--lines N]  capture a session's pane
  prompt    --worker NAME     print the prompt for the item NAME holds (no agent run)
  down      [--keep-leases]   kill only the sessions this plan created; release their leases

Sessions are named <execution.session_prefix>-sup and -wNN and recorded in
tools/workers.json. Sessions this tool did not create are never touched.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

HERE = Path(__file__).resolve().parent


class OrchError(Exception):
    pass


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


_LOG_FILE: Path | None = None


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_to(path: Path) -> None:
    global _LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = path


# ---------------------------------------------------------------------------
# plan access

class Ctx:
    def __init__(self, plan_dir: Path):
        self.dir = plan_dir.resolve()
        self.tools = self.dir / "tools"
        self.tool = self.tools / "plan_tool.py"
        if not (self.tools / "plan-spec.json").exists():
            raise OrchError(f"no plan at {self.dir} (tools/plan-spec.json missing)")
        self.spec = json.loads((self.tools / "plan-spec.json").read_text(encoding="utf-8"))
        if self.spec.get("execution") is None:
            raise OrchError("this plan has no \"execution\" block in tools/plan-spec.json; asynchronous execution is off")
        sys.path.insert(0, str(self.tools))
        import plan_tool  # the copy that lives beside this file
        self.pt = plan_tool
        self.ex = {**plan_tool.DEFAULT_EXECUTION, **self.spec["execution"]}
        self.ex.setdefault("session_prefix", "cc-" + self.spec["plan"]["slug"][:24])
        self.prefix = self.ex["session_prefix"]
        self.root = plan_tool.repo_root(self.dir)
        self.registry = self.tools / "workers.json"
        self.logs = self.tools / "_runs"  # agent prompts and output per run; gitignored

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def run_tool(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run([sys.executable, str(self.tool), "--plan-dir", str(self.dir), *args],
                           capture_output=True, text=True, cwd=self.root)
        if check and r.returncode != 0:
            raise OrchError(f"plan_tool {' '.join(args[:2])} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r

    def queue(self) -> dict:
        return json.loads(self.run_tool("queue", "--json").stdout)

    # -- registry of sessions this plan created ------------------------------
    @contextlib.contextmanager
    def reg(self):
        lock = self.tools / ".workers.lock"
        with lock.open("a") as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = json.loads(self.registry.read_text(encoding="utf-8")) if self.registry.exists() else {}
                data.setdefault("sessions", {})
                data.setdefault("next_index", 1)
                yield data
                tmp = self.registry.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                tmp.replace(self.registry)
            finally:
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# tmux

def tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise OrchError(f"tmux {' '.join(args[:3])}: {r.stderr.strip()}")
    return r


def tmux_sessions() -> set[str]:
    r = tmux("list-sessions", "-F", "#{session_name}", check=False)
    return set(r.stdout.split()) if r.returncode == 0 else set()


def require_tmux() -> None:
    if not shutil.which("tmux"):
        raise OrchError("tmux is not installed or not on PATH")


def live_sessions(ctx: Ctx, role: str | None = None) -> list[str]:
    alive = tmux_sessions()
    with ctx.reg() as data:
        for name in [n for n in data["sessions"] if n not in alive]:
            data["sessions"].pop(name)
        return sorted(n for n, s in data["sessions"].items() if role is None or s["role"] == role)


def start_session(ctx: Ctx, name: str, role: str, argv: list[str]) -> None:
    if name in tmux_sessions():
        raise OrchError(f"tmux session {name} already exists")
    ctx.logs.mkdir(parents=True, exist_ok=True)
    tmux("new-session", "-d", "-s", name, "-c", str(ctx.root), shlex.join(argv))
    with ctx.reg() as data:
        data["sessions"][name] = {"role": role, "created": now(), "command": shlex.join(argv)}
    log(f"started tmux session {name} ({role})")


def spawn_workers(ctx: Ctx, count: int) -> list[str]:
    names = []
    for _ in range(count):
        with ctx.reg() as data:  # numbers are never reused, so a new worker never inherits an old identity
            n = data["next_index"]
            data["next_index"] = n + 1
        name = f"{ctx.prefix}-w{n:02d}"
        start_session(ctx, name, "worker", [sys.executable, str(Path(__file__).resolve()), "--plan-dir", str(ctx.dir),
                                            "work", "--worker", name])
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# prompts

ROLE_RE = re.compile(r"<!--\s*ROLE\s+([a-z-]+)\s*-->(.*?)<!--\s*END ROLE\s*-->", re.S)


def build_prompt(ctx: Ctx, item: dict, worker: str) -> str:
    tpl_path = ctx.dir / "WORKER_PROMPT.md"
    if not tpl_path.exists():
        raise OrchError(f"{tpl_path} is missing")
    tpl = tpl_path.read_text(encoding="utf-8")
    sections = {m.group(1): m.group(2).strip() for m in ROLE_RE.finditer(tpl)}
    role = item["role"]
    section = sections.get("implement" if role == "rework" else role, "")
    if role == "rework":
        section = sections.get("rework", "") + "\n\n" + section
    body = ROLE_RE.sub("", tpl).replace("{{ITEM_ROLE_INSTRUCTIONS}}", section)
    tool = f"python3 {ctx.rel(ctx.tool)}"
    task = item.get("task") or ""
    if role in ("implement", "rework") and item.get("verify_mode") == "inline":
        finish = (f"{tool} complete-task {task} --worker {worker} --reason '<one-line status>'\n"
                  "(inline task: you run its verify check yourself; nobody else will)")
    elif role in ("implement", "rework"):
        finish = (f"{tool} submit {task} --worker {worker} --reason '<one line: what changed, check run, result>'\n"
                  "(another worker verifies it independently; do not wait for that)")
    elif role == "verify":
        finish = (f"{tool} verify-pass {task} --worker {worker} --reason '<one-line status>'\n"
                  f"or\n{tool} verify-fail {task} --worker {worker} --reason '<concrete findings to fix>'")
    else:
        finish = (f"{tool} complete-stage {item['module']} {item['stage']} --worker {worker} "
                  "--reason '<one line: how the gate holds>'")
    rework = item.get("rework") or []
    sub = item.get("submitted") or {}
    values = {
        "WORKER": worker, "TOOL": tool, "ITEM_ROLE": role, "ITEM_KEY": item["key"], "ITEM_TASK": task or "-",
        "ITEM_MODULE": item["module"], "ITEM_STAGE": item["stage"], "ITEM_TEXT": item["text"],
        "ITEM_VERIFY": item["verify"], "ITEM_VERIFY_MODE": item.get("verify_mode", "-"),
        "ITEM_REQUIREMENTS": ", ".join(item.get("requirements", [])) or "-",
        "ITEM_MODULE_DIR": item["module_dir"],
        "ITEM_WRITE_SET": ("(parallel phase: other workers are editing the module's other files right now; touch only these)\n"
                           + "\n".join(f"- `{w}`" for w in item["task_write_set"])) if item.get("task_write_set") and
                          item.get("phase_mode") == "parallel" else
                          "\n".join(f"- `{w}`" for w in item.get("write_set", [])) or "- (none declared: stay minimal)",
        "ITEM_SUBMITTED": (f"Submitted by {sub.get('submitted_by')}: {sub.get('reason')}"
                           if sub else "-"),
        "ITEM_REWORK": "\n".join(f"- {r.get('at')} {r.get('by')}: {r.get('findings')}" for r in rework) or "- none",
        "ITEM_ATTEMPT": str(item.get("attempt", 1)), "ITEM_FINISH": finish,
        "ITEM_RESUMED": "yes: an earlier run of this worker held this item; inspect and keep its partial work"
                        if item.get("resumed") else "no",
    }
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda mo: values.get(mo.group(1), mo.group(0)), body)


def agent_argv(ctx: Ctx, prompt: str, prompt_file: Path, item: dict, worker: str) -> list[str]:
    subs = {"{prompt}": prompt, "{prompt_file}": str(prompt_file), "{worker}": worker,
            "{task}": item.get("task") or item["key"], "{role}": item["role"]}
    argv = []
    for a in ctx.ex["agent_command"]:
        for k, v in subs.items():
            a = a.replace(k, v)
        argv.append(a)
    if Path(argv[0]).name == "claude":
        if ctx.ex.get("model"):
            argv += ["--model", ctx.ex["model"]]
        argv += ["--allowedTools", *base_tools(ctx)]
    return argv


def base_tools(ctx: Ctx) -> list[str]:
    """What every worker needs to record its result, plus the project's own list."""
    base = ["Read", "Glob", "Grep", "Edit", "Write",
            f"Bash(python3 {ctx.rel(ctx.tool)}:*)", "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)"]
    return list(dict.fromkeys(base + list(ctx.ex.get("allowed_tools") or [])))


# ---------------------------------------------------------------------------
# worker loop

def run_agent(ctx: Ctx, worker: str, item: dict) -> int:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item["key"])
    wdir = ctx.logs / worker
    wdir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prompt = build_prompt(ctx, item, worker)
    pfile = wdir / f"{stamp}-{item['role']}-{safe}.prompt.md"
    pfile.write_text(prompt, encoding="utf-8")
    ifile = wdir / f"{stamp}-{item['role']}-{safe}.item.json"
    ifile.write_text(json.dumps(item, indent=2), encoding="utf-8")
    lfile = wdir / f"{stamp}-{item['role']}-{safe}.log"
    argv = agent_argv(ctx, prompt, pfile, item, worker)
    env = {**os.environ, "PLAN_DIR": str(ctx.dir), "PLAN_TOOL": str(ctx.tool), "PLAN_WORKER": worker,
           "PLAN_ROLE": item["role"], "PLAN_ITEM": item["key"], "PLAN_TASK": item.get("task") or "",
           "PLAN_MODULE": item["module"], "PLAN_STAGE": item["stage"], "PLAN_ITEM_JSON": str(ifile),
           "PLAN_PROMPT_FILE": str(pfile)}
    log(f"{item['role']} {item['key']}: {item['text']}")
    log(f"agent: {Path(argv[0]).name} (prompt {ctx.rel(pfile)}, log {ctx.rel(lfile)})")

    stop = threading.Event()

    def beat():
        every = max(30.0, float(ctx.ex["lease_minutes"]) * 60 / 3)
        while not stop.wait(every):
            ctx.run_tool("heartbeat", "--worker", worker, check=False)

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()
    timeout = float(ctx.ex["agent_timeout_minutes"]) * 60
    try:
        with lfile.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(argv, cwd=ctx.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, bufsize=1)
            timer = threading.Timer(timeout, proc.kill)
            timer.start()
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    lf.write(line)
                    lf.flush()
                code = proc.wait()
            finally:
                timer.cancel()
    except FileNotFoundError:
        code = 127
        log(f"agent command not found: {argv[0]}")
    finally:
        stop.set()
    log(f"agent exited with code {code}")
    return code


def cmd_work(ctx: Ctx, args) -> int:
    worker = args.worker
    idle = 0
    log_to(ctx.logs / worker / "loop.log")
    log(f"worker {worker} on {ctx.rel(ctx.dir)}")
    while True:
        r = ctx.run_tool("claim", "--worker", worker, "--role", args.role, "--json", check=False)
        if r.returncode != 0:
            log(f"claim failed: {r.stderr.strip()}")
            time.sleep(float(ctx.ex["poll_seconds"]))
            continue
        item = json.loads(r.stdout)
        if item["role"] == "idle":
            if item.get("plan_done"):
                log("plan complete; worker exits")
                return 0
            idle += 1
            if idle >= int(ctx.ex["idle_exit_polls"]) or args.once:
                log(f"idle {idle} poll(s) ({item['reason']}); worker exits")
                return 0
            time.sleep(float(ctx.ex["poll_seconds"]))
            continue
        idle = 0
        code = run_agent(ctx, worker, item)
        q = ctx.queue()
        still = [le for le in q["leases"] if le["worker"] == worker and le["key"] == item["key"]]
        if still:
            reason = f"agent exited with code {code} without recording {item['key']}"
            log(reason + "; releasing as a failed attempt")
            ctx.run_tool("release", item["key"], "--worker", worker, "--reason", reason, "--failed", check=False)
        if args.once:
            return 0


# ---------------------------------------------------------------------------
# supervisor

def cmd_supervise(ctx: Ctx, args) -> int:
    require_tmux()
    cap = int(args.workers or ctx.ex["max_workers"])
    log_to(ctx.logs / f"{ctx.prefix}-sup.log")
    log(f"supervisor for {ctx.rel(ctx.dir)}; up to {cap} workers")
    quiet = 0
    while True:
        q = ctx.queue()
        live = live_sessions(ctx, "worker")
        busy = {le["worker"] for le in q["leases"]}
        pending = len(q["awaiting_verification"]) + len(set(q["ready"]) | set(q["rework"]))
        if q["plan_done"] and not live:
            log("plan complete; supervisor exits")
            return 0
        want = min(cap, len(busy & set(live)) + pending)
        idle_live = [w for w in live if w not in busy]
        # a submitted task needs a worker other than its implementer
        if want <= len(live) and q["awaiting_verification"] and not q["ready"] and len(live) < cap and \
                all(a["submitted_by"] in idle_live for a in q["awaiting_verification"]) and \
                len(idle_live) <= len({a["submitted_by"] for a in q["awaiting_verification"]}):
            want = len(live) + 1
        if want > len(live):
            spawn_workers(ctx, want - len(live))
        if not live and not q["leases"] and pending == 0:
            quiet += 1
            if quiet >= 2:
                blocked = ", ".join(q["blocked"]) or "none"
                log(f"nothing runnable and no workers (blocked: {blocked}); a human must act. Supervisor exits")
                return 0
        else:
            quiet = 0
        time.sleep(float(ctx.ex["poll_seconds"]))


# ---------------------------------------------------------------------------
# operator commands

def cmd_up(ctx: Ctx, args) -> int:
    require_tmux()
    agent = ctx.ex["agent_command"][0]
    if not shutil.which(agent) and not Path(agent).exists():
        raise OrchError(f"agent command {agent!r} is not on PATH")
    if Path(agent).name == "claude" and not ctx.ex.get("allowed_tools"):
        log("warning: execution.allowed_tools is empty; workers may only use the plan tool, read-only git and file "
            "edits, so any build or test command in a verify check will be denied")
    if not (ctx.root / ".git").exists():
        log("warning: no git repository; workers share one working tree and you will have no diff to review")
    print("existing tmux sessions:", ", ".join(sorted(tmux_sessions())) or "none")
    ctx.run_tool("validate")
    sup = f"{ctx.prefix}-sup"
    if sup in tmux_sessions():
        print(f"supervisor {sup} is already running; attach with: tmux attach -t {sup}")
        return 0
    argv = [sys.executable, str(Path(__file__).resolve()), "--plan-dir", str(ctx.dir), "supervise"]
    if args.workers:
        argv += ["--workers", str(args.workers)]
    start_session(ctx, sup, "supervisor", argv)
    print(f"watch:  tmux attach -t {sup}        status: python3 {ctx.rel(Path(__file__).resolve())} status")
    return 0


def cmd_spawn(ctx: Ctx, args) -> int:
    require_tmux()
    for n in spawn_workers(ctx, args.count):
        print(n)
    return 0


def cmd_status(ctx: Ctx, args) -> int:
    alive = tmux_sessions()
    with ctx.reg() as data:
        sessions = dict(data["sessions"])
    q = ctx.queue()
    lease_by = {le["worker"]: le for le in q["leases"]}
    print(f"plan {ctx.rel(ctx.dir)}  done={q['plan_done']}  blocked={','.join(q['blocked']) or '-'}")
    print(f"awaiting verification: {', '.join(a['task'] for a in q['awaiting_verification']) or '-'}")
    print(f"rework: {', '.join(q['rework']) or '-'}   ready: {', '.join(q['ready']) or '-'}")
    for name, s in sorted(sessions.items()):
        le = lease_by.get(name)
        state = "alive" if name in alive else "gone"
        print(f"\n== {name} [{s['role']}, {state}] " + (f"{le['role']} {le['key']}" if le else ""))
        if name in alive:
            pane = tmux("capture-pane", "-p", "-t", name, check=False).stdout.rstrip().splitlines()
            print("\n".join(pane[-args.lines:]))
    if q.get("events"):
        print("\nrecent events:\n" + "\n".join(q["events"][-8:]))
    return 0


def cmd_logs(ctx: Ctx, args) -> int:
    with ctx.reg() as data:
        if args.name not in data["sessions"]:
            raise OrchError(f"{args.name} is not a session this plan created")
    print(tmux("capture-pane", "-p", "-S", f"-{args.lines}", "-t", args.name).stdout)
    return 0


def cmd_prompt(ctx: Ctx, args) -> int:
    q = ctx.queue()
    le = next((x for x in q["leases"] if x["worker"] == args.worker), None)
    if not le:
        raise OrchError(f"{args.worker} holds no lease; run `plan_tool.py claim --worker {args.worker}` first")
    item = json.loads(ctx.run_tool("claim", "--worker", args.worker, "--json").stdout)  # idempotent for a holder
    print(build_prompt(ctx, item, args.worker))
    return 0


def cmd_down(ctx: Ctx, args) -> int:
    alive = tmux_sessions()
    with ctx.reg() as data:
        names = sorted(data["sessions"], key=lambda n: data["sessions"][n]["role"] != "supervisor")
    for name in names:
        if name in alive:
            tmux("kill-session", "-t", f"={name}", check=False)
            log(f"killed {name}")
    if not args.keep_leases:
        for le in ctx.queue()["leases"]:
            if le["worker"] in names:
                ctx.run_tool("release", le["key"], "--worker", le["worker"], "--reason", "orchestrator down", check=False)
                log(f"released {le['key']} held by {le['worker']}")
    with ctx.reg() as data:
        data["sessions"] = {}
    return 0


def default_plan_dir() -> Path | None:
    if HERE.name == "tools" and (HERE / "plan-spec.json").exists():
        return HERE.parent
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-dir", help="plan directory (default: the plan this tool lives in)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("up")
    p.add_argument("--workers", type=int)
    p = sub.add_parser("supervise")
    p.add_argument("--workers", type=int)
    p = sub.add_parser("spawn")
    p.add_argument("--count", type=int, default=1)
    p = sub.add_parser("work")
    p.add_argument("--worker", required=True)
    p.add_argument("--role", choices=["any", "implement", "verify"], default="any")
    p.add_argument("--once", action="store_true", help="handle one item (or one idle poll) and exit")
    p = sub.add_parser("status")
    p.add_argument("--lines", type=int, default=8)
    p = sub.add_parser("logs")
    p.add_argument("name")
    p.add_argument("--lines", type=int, default=300)
    p = sub.add_parser("prompt")
    p.add_argument("--worker", required=True)
    p = sub.add_parser("down")
    p.add_argument("--keep-leases", action="store_true")
    args = ap.parse_args(argv)
    try:
        plan_dir = Path(args.plan_dir) if args.plan_dir else default_plan_dir()
        if not plan_dir:
            raise OrchError("pass --plan-dir, or run the copy of this tool inside a plan's tools/ directory")
        ctx = Ctx(plan_dir)
        return {"up": cmd_up, "supervise": cmd_supervise, "spawn": cmd_spawn, "work": cmd_work,
                "status": cmd_status, "logs": cmd_logs, "prompt": cmd_prompt, "down": cmd_down}[args.cmd](ctx, args)
    except OrchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
