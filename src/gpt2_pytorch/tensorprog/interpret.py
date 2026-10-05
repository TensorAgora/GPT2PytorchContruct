"""Execute a tensor program with torch, binding parameters straight from an HF-layout checkpoint (no transposition)."""
import torch

from .program import Op, Program

ELEMENTWISE = {
    "add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
    "exp": torch.exp, "tanh": torch.tanh, "rsqrt": torch.rsqrt, "log": torch.log,
}
REDUCE = {"mean": torch.mean, "sum": torch.sum, "max": torch.amax}
DTYPES = {"float32": torch.float32, "int64": torch.int64}


def _align(t: torch.Tensor, axes, out_axes) -> torch.Tensor:
    """Broadcast by axis name: order t's axes like out_axes, insert size-1 axes for the missing ones."""
    present = [a for a in out_axes if a in axes]
    t = t.permute([axes.index(a) for a in present]) if present else t
    return t.reshape([t.shape[present.index(a)] if a in axes else 1 for a in out_axes])


def _run(p: Program, op: Op, env: dict, sizes: dict) -> torch.Tensor:
    x = [env[name] for name in op.inputs]
    axes = [p.axes(name) for name in op.inputs]
    out_axes = p.axes(op.out)
    if op.primitive == "Contraction":
        letters = {a: chr(ord("a") + i) for i, a in enumerate(dict.fromkeys(axes[0] + axes[1]))}
        spec = ",".join("".join(letters[a] for a in ax) for ax in axes) + "->" + "".join(letters[a] for a in out_axes)
        return torch.einsum(spec, *x)
    if op.primitive == "Elementwise":
        return ELEMENTWISE[op.fn](*[_align(t, ax, out_axes) for t, ax in zip(x, axes)])
    if op.primitive == "Reduce":
        return REDUCE[op.fn](x[0], dim=axes[0].index(op.attrs["axis"]))
    if op.primitive == "Gather":
        assert op.attrs["axis"] == p.axes(op.inputs[0])[0], "gather is along the table's first axis"
        return x[0][x[1]]
    if op.primitive == "Slice":
        at = op.attrs
        return x[0].narrow(axes[0].index(at["axis"]), at["start"], at["stop"] - at["start"])
    if op.primitive == "Permute":
        return x[0].permute(op.attrs["perm"])
    if op.primitive == "Reshape":
        return x[0].reshape([sizes[a] for a in out_axes])
    if op.primitive == "Rename":
        return x[0]
    raise NotImplementedError(op.primitive)


def run_program(p: Program, state: dict[str, torch.Tensor], input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run every op in order. `state` is a checkpoint state_dict (HF key names, HF layout). Returns all tensors by name."""
    batch, seq = input_ids.shape
    sizes = dict(p.dims, B=batch, T=seq, q=seq, k=seq)
    env: dict[str, torch.Tensor] = {"ids": input_ids, "pos": torch.arange(seq, device=input_ids.device)}
    for name, decl in p.tensors.items():
        if decl.role in ("parameter", "buffer"):
            t = state[decl.checkpoint_key]
            if decl.view == "[0,0,:T,:T]":
                t = t[0, 0, :seq, :seq]
            assert tuple(t.shape) == tuple(sizes[a] for a in decl.axes), f"{name}: checkpoint {tuple(t.shape)} vs axes {decl.axes}"
            env[name] = t
        elif decl.role == "constant":
            env[name] = torch.tensor(decl.value, dtype=DTYPES[decl.dtype], device=input_ids.device)
    with torch.no_grad():
        for op in p.ops:
            env[op.out] = _run(p, op, env, sizes)
    return env


REL_TOL = 1e-5  # per tensor, relative to max|reference| (fp32 error scales with the tensor, see docs/TEL_NEXT_STEPS.md)


def verify(p: Program, model, state: dict[str, torch.Tensor], input_ids: torch.Tensor) -> dict:
    """Run the program on `input_ids` and compare every tensor that has a counterpart in the model's debug output."""
    env = run_program(p, state, input_ids)
    with torch.no_grad():
        debug = model(input_ids, return_debug=True).debug
    worst, worst_name, compared = 0.0, "", 0
    for name, decl in p.tensors.items():
        if decl.pytorch_name is None:
            continue
        got, ref = env[name], debug[decl.pytorch_name]
        assert got.shape == ref.shape, f"{name}: {tuple(got.shape)} vs {decl.pytorch_name} {tuple(ref.shape)}"
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
        compared += 1
        if rel > worst:
            worst, worst_name = rel, name
    logits_err = (env["L"] - debug["model.logits"]).abs().max().item()
    return {
        "shape": list(input_ids.shape), "compared": compared, "worst_relative_error": worst, "worst_tensor": worst_name,
        "logits_max_abs_error": logits_err, "argmax_identical": bool(torch.equal(env["L"].argmax(-1), debug["model.logits"].argmax(-1))),
        "ok": worst <= REL_TOL,
    }
