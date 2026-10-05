"""python -m tools.run_pipeline [--out artifacts/pipeline_report.md]

Runs the whole validation pipeline (inspect -> forward -> trace -> Safetensors -> parity -> FX/export -> pytest),
captures every step's output, and writes one Markdown report. Full logs go to artifacts/logs/.
"""
import argparse
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

STEPS = [
    ("Inspect checkpoint", "tools.inspect_checkpoint", "83 tensors, tying, inferred architecture, tree"),
    ("Print model", "tools.print_model", "module tree, parameter counts, logical vs unique accounting"),
    ("Tensor inventory", "tools.tensor_inventory", "every parameter/buffer -> artifacts/tensor_inventory.{json,csv}"),
    ("Fixed forward", "examples.forward_fixed", "deterministic eval forward, intermediate shapes, logits stats"),
    ("DataLoader forward", "examples.dataloader_forward", "Dataset -> DataLoader -> batch -> logits"),
    ("Inspect attention", "examples.inspect_attention", "Q/K/V, mask, probabilities, causality"),
    ("Forward hooks", "examples.forward_hooks", "pre/post hook execution order"),
    ("Execution trace", "examples.execution_trace", "artifacts/execution_trace.json"),
    ("Causal LM loss", "examples.causal_lm_loss", "shift_logits, shift_labels, loss"),
    ("Single train step", "examples.single_train_step", "backward, gradients, one SGD step"),
    ("Autograd inspect", "examples.autograd_inspect", "grad_fn, is_leaf, backward graph"),
    ("Convert to Safetensors", "tools.convert_to_safetensors", "weights/pytorch_model.bin -> artifacts/model.safetensors"),
    ("Inspect Safetensors", "tools.inspect_safetensors", "header-only inspection"),
    ("Hugging Face parity", "reference.compare_huggingface --check-gelu-variants", "optional; needs transformers"),
    ("FX trace", "examples.fx_trace", "artifacts/fx_graph.{txt,json}"),
    ("torch.export", "examples.export_model", "artifacts/export/"),
    ("Tests", "pytest", "full pytest suite"),
]
HEAD, TAIL = 45, 12


def run(module: str):
    argv = [sys.executable, "-m", *module.split()]
    start = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode, time.perf_counter() - start, (proc.stdout + proc.stderr).rstrip()


def clip(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= HEAD + TAIL + 5:
        return text
    return "\n".join([*lines[:HEAD], f"... [{len(lines) - HEAD - TAIL} lines omitted, full log in artifacts/logs/] ...", *lines[-TAIL:]])


def key_results() -> list[str]:
    from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_pretrained
    from gpt2_pytorch.data import FIXED_INPUT_IDS
    from gpt2_pytorch.utils import parameter_breakdown

    model, config, report = load_pretrained(DEFAULT_CHECKPOINT)
    model_st, _, report_st = load_pretrained("artifacts/model.safetensors")
    with torch.no_grad():
        logits, logits_st = model(FIXED_INPUT_IDS), model_st(FIXED_INPUT_IDS)
    counts = parameter_breakdown(model)
    return [
        f"- Architecture: {config.num_layers} layers, hidden {config.hidden_size}, {config.num_heads} heads (head_dim {config.head_dim}), "
        f"MLP {config.intermediate_size}, vocab {config.vocab_size}, {config.max_position_embeddings} positions",
        f"- Parameters: {counts['unique_total']:,} unique / {counts['logical_total']:,} logical (tied lm_head)",
        f"- Load report (.bin): loaded {len(report.loaded)}, transposed {len(report.transposed)}, tied {len(report.tied)}, "
        f"ignored {len(report.ignored)}, missing {len(report.missing)}, unexpected {len(report.unexpected)}",
        f"- Load report (.safetensors): {report_st.tied}",
        f"- Fixed input {list(FIXED_INPUT_IDS.shape)} -> logits {list(logits.shape)}; "
        f"logits[0,-1,:3] = {[round(v, 4) for v in logits[0, -1, :3].tolist()]}",
        f"- `.bin` vs Safetensors logits bit-identical: **{torch.equal(logits, logits_st)}**",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/pipeline_report.md")
    args = parser.parse_args()
    log_dir = Path("artifacts/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (title, module, purpose) in enumerate(STEPS, 1):
        print(f"[{i:02d}/{len(STEPS)}] {title} ...", end=" ", flush=True)
        code, seconds, output = run("pytest -q" if module == "pytest" else module)
        slug = re.sub(r"[^a-z0-9]+", "_", module.split()[0].lower())
        (log_dir / f"{i:02d}_{slug}.txt").write_text(output + "\n")
        print(f"{'ok' if code == 0 else f'FAILED (exit {code})'} {seconds:.1f}s")
        results.append((title, module, purpose, code, seconds, output))

    passed = sum(r[3] == 0 for r in results)
    out = [
        "# DistilGPT2 pipeline report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} on {platform.platform()} | Python {platform.python_version()} | torch {torch.__version__} | device cpu",
        "",
        f"**{passed}/{len(results)} steps passed.**",
        "",
        "## Summary",
        "",
        "| # | Step | Command | Result | Time | Produces |",
        "|--:|---|---|---|--:|---|",
    ]
    for i, (title, module, purpose, code, seconds, _) in enumerate(results, 1):
        command = "pytest" if module == "pytest" else f"python -m {module.split()[0]}"
        out.append(f"| {i} | {title} | `{command}` | {'PASS' if code == 0 else f'FAIL ({code})'} | {seconds:.1f}s | {purpose} |")
    out += ["", "## Key results", "", *key_results(), "", "## Step output", ""]
    for i, (title, module, purpose, code, seconds, output) in enumerate(results, 1):
        out += [f"### {i}. {title}", "", f"`{'pytest -q' if module == 'pytest' else 'python -m ' + module}` | exit {code} | {seconds:.1f}s", "", "```text", clip(output), "```", ""]
    artifacts = sorted(p for p in Path("artifacts").rglob("*") if p.is_file() and "logs" not in p.parts)
    out += ["## Generated artifacts", "", "| file | size |", "|---|--:|"]
    out += [f"| `{p}` | {p.stat().st_size:,} B |" for p in artifacts]
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"\nwrote {args.out}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
