"""Any nn.Module + its state_dict -> the graph of its matrix multiplications, in execution order.

torch.export captures the model, run_decompositions() lowers linear / matmul / einsum / SDPA / conv to the core ATen
ops mm, bmm, addmm, baddbmm, convolution, so one small op set covers every architecture. Those ops are the graph's
nodes; every other op (add, softmax, layernorm, views, ...) is a pass-through that carries dependencies. An edge
u -> v means v reads a tensor computed from u without another matrix multiplication in between (residual paths
therefore give edges that skip blocks). Operands are traced back through view-like ops to the state_dict key they
read, so `x @ W.t()` reports W by its checkpoint name with via=["permute"].
"""
import math

import torch
from torch.utils._pytree import tree_flatten

aten = torch.ops.aten
# op -> (left operand index, right operand index, bias index or None)
MATMUL_OPS = {
    aten.mm.default: (0, 1, None), aten.bmm.default: (0, 1, None),
    aten.addmm.default: (1, 2, 0), aten.baddbmm.default: (1, 2, 0), aten.convolution.default: (0, 1, 2),
}
VIEW_OPS = {
    aten.view.default, aten._unsafe_view.default, aten.reshape.default, aten.permute.default, aten.t.default,
    aten.transpose.int, aten.expand.default, aten.unsqueeze.default, aten.squeeze.dims, aten.clone.default, aten.alias.default,
}


def load_state_dict_checked(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """model.load_state_dict, but a missing key is allowed only if it is a tied alias of a key that was loaded
    (Safetensors drops tied copies). Shape mismatches and unexpected keys raise inside load_state_dict."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    targets = model.state_dict(keep_vars=True)
    by_storage: dict[int, set[str]] = {}
    for name, t in targets.items():
        by_storage.setdefault(id(t), set()).add(name)
    untied = [k for k in missing if not (by_storage[id(targets[k])] & state.keys())]
    if untied or unexpected:
        raise ValueError(f"state_dict does not match model: missing {untied}, unexpected {unexpected}")
    return model.eval()


def _operand(node, names):
    """(state_dict key / input name / producing node, kind, ops looked through) for one matmul operand."""
    via = []
    while node.op == "call_function" and node.target in VIEW_OPS:
        via.append(node.target._opname)
        node = node.args[0]
    if node.op == "placeholder":
        return names[node.name][0], names[node.name][1], via
    return None, "activation", via


def matmul_graph(model: torch.nn.Module, example_args: tuple, rtol: float = 1e-4) -> dict:
    with torch.no_grad():
        ep = torch.export.export(model.eval(), example_args).run_decompositions()
        expected, _ = tree_flatten(model(*example_args))
        got, _ = tree_flatten(ep.module()(*example_args))
    # relative to each output's max magnitude: decomposed SDPA/conv differ from fused kernels by float rounding
    max_diff = max(((g - e).abs().max() / e.abs().max().clamp_min(1e-12)).item() for g, e in zip(got, expected))
    assert max_diff <= rtol, f"decomposed graph differs from eager by {max_diff:.1e} (relative)"

    sig = ep.graph_signature
    names = {p: (k, "parameter") for p, k in sig.inputs_to_parameters.items()}
    names |= {p: (k, "buffer") for p, k in sig.inputs_to_buffers.items()}
    names |= {p: (k, "constant") for p, k in sig.inputs_to_lifted_tensor_constants.items()}
    names |= {p: (p, "input") for p in sig.user_inputs}

    nodes, reach, order = [], {}, {"input": -1}  # reach[fx node] = matmuls (or "input") reaching it without crossing another matmul
    for n in ep.graph.nodes:
        ups = set().union(*(reach[a] for a in n.all_input_nodes))
        if n.op == "placeholder":
            reach[n] = {"input"} if n.name in sig.user_inputs else set()
        elif n.op == "call_function" and n.target in MATMUL_OPS:
            li, ri, bi = MATMUL_OPS[n.target]
            left, right, out = n.args[li], n.args[ri], n.meta["val"]
            k = math.prod(right.meta["val"].shape[1:]) if n.target is aten.convolution.default else left.meta["val"].shape[-1]
            operands = {}
            for role, arg in (("left", left), ("right", right), ("bias", n.args[bi] if bi is not None else None)):
                if arg is not None:
                    source, kind, via = _operand(arg, names)
                    operands[role] = {"source": source, "kind": kind, "via": via, "shape": list(arg.meta["val"].shape),
                                      "from": sorted(reach[arg], key=order.get)}
            stack = list(n.meta.get("nn_module_stack", {}).values())
            nodes.append({
                "id": n.name, "index": len(nodes), "op": n.target._opname, "module": stack[-1][0] if stack else "",
                "operands": operands, "out_shape": list(out.shape), "flops": 2 * out.numel() * k, "deps": sorted(ups, key=order.get),
            })
            reach[n], order[n.name] = {n.name}, len(nodes) - 1
        else:
            reach[n] = ups
            if n.op == "output":
                outputs = sorted(ups, key=order.get)

    tied = {}
    for name, t in model.named_parameters(remove_duplicate=False):
        tied.setdefault(id(t), []).append(name)
    return {
        "format": "tensormorph-matmul-graph", "version": 1,
        "inputs": [{"name": p, "shape": list(a.shape), "dtype": str(a.dtype)} for p, a in zip(sig.user_inputs, example_args)],
        "nodes": nodes,
        "edges": [[d, v["id"], role] for v in nodes for role, o in v["operands"].items() for d in o["from"]],  # [src, dst, operand]
        "outputs": outputs,
        "tied_parameters": [group for group in tied.values() if len(group) > 1],
        "total_flops": sum(v["flops"] for v in nodes),
        "max_rel_diff_vs_eager": max_diff,
    }


def to_markdown(g: dict) -> str:
    def src(o):
        return f"`{o['source'] or 'activation'}`" + (f" ({'→'.join(o['via'])})" if o["via"] else "")

    lines = [
        "# Matrix-multiplication graph", "",
        f"{len(g['nodes'])} matmuls, {len(g['edges'])} edges, {g['total_flops'] / 1e6:,.1f} MFLOPs; "
        f"decomposed graph vs eager max relative diff {g['max_rel_diff_vs_eager']:.1e}. Generated by `python -m tools.matmul_graph`.", "",
        "| # | id | op | module | left | right | out | MFLOPs | deps |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in g["nodes"]:
        o = v["operands"]
        lines.append(f"| {v['index']} | {v['id']} | {v['op']} | `{v['module']}` | {src(o['left'])} {o['left']['shape']} | "
                     f"{src(o['right'])} {o['right']['shape']} | {v['out_shape']} | {v['flops'] / 1e6:,.2f} | {', '.join(v['deps'])} |")
    lines += ["", "```mermaid", "graph TD", *(f"  {a} -- {r} --> {b}" for a, b, r in g["edges"]), *(f"  {o} --> output" for o in g["outputs"]), "```", ""]
    return "\n".join(lines)
