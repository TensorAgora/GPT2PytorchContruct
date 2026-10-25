"""python -m tools.viz_to_tel [--graph artifacts/viz_graph.json] [--out artifacts/tel/from_viz.tel]

Prints the whole model as Quick TEL, in the style of artifacts/tel/model.tel, from viz_graph.json alone:
axis and tensor declarations from `axes` and `tensors`, then one statement per TEL symbol (a tensor name without a
'.'). Intermediates read once are inlined into their consumer; ops of a composite region print as the composite
(`layernorm(X_0, axis=D, eps=0.00001)`, `softmax(..., axis=k)`). Check the result with tools/tel_compile.mjs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--graph", default="artifacts/viz_graph.json")
parser.add_argument("--out", default="artifacts/tel/from_viz.tel")
args = parser.parse_args()

g = json.loads(Path(args.graph).read_text())
tensors = {t["name"]: t for t in g["tensors"]}
regions = {r["id"]: r for r in g["regions"]}
DTYPE = {"float32": "f32", "int64": "i64"}
ADD, MUL, MATMUL, ATOM = 1, 2, 3, 9
INFIX = {"add": ("+", ADD), "sub": ("-", ADD), "mul": ("*", MUL), "div": ("/", MUL)}
UNARY = {"exp", "log", "tanh"}


def literal(v: float) -> str:
    s = repr(float(v))
    if "e" in s:
        s = f"{v:.20f}".rstrip("0")
    return s.removesuffix(".0")


# Nodes as TEL sees them: every op outside a region, and each region once, as a call on its input.
nodes = []
for op in g["ops"]:
    r = regions.get(op["region"])
    if r is None:
        nodes.append(op)
    elif op["out"] == r["output"]:
        reads = {i for k in r["ops"] for i in g["ops"][k]["inputs"]} - {g["ops"][k]["out"] for k in r["ops"]}
        assert reads - {r["input"]} <= {n for n in reads if tensors[n]["role"] == "constant"}, f"region {r['id']} reads {reads}"
        nodes.append({"primitive": "Region", "region": r, "out": op["out"], "inputs": [r["input"]], "module": op["module"]})
uses = Counter(i for n in nodes for i in dict.fromkeys(n["inputs"]))
named = {n["out"] for n in nodes if "." not in n["out"] or uses[n["out"]] != 1} | set(g["outputs"])
ident = {name: name.replace(".", "_") for name in tensors}
assert len(set(ident.values())) == len(ident), "two tensor names collide once '.' becomes '_'"
expr: dict[str, tuple[str, int]] = {}


def ref(name: str) -> tuple[str, int]:
    t = tensors[name]
    if t["role"] == "constant":
        return literal(t["value"]), ATOM
    return expr[name] if name in expr else (ident[name], ATOM)


def atom(name: str) -> str:
    text, prec = ref(name)
    return text if prec == ATOM else f"({text})"


def render(n: dict) -> tuple[str, int]:
    p, ins = n["primitive"], n["inputs"]
    if p == "Region":
        r = n["region"]
        attrs = ", ".join(f"{k}={literal(v) if isinstance(v, float) else v}" for k, v in r["attrs"].items())
        return f"{r['name']}({ref(r['input'])[0]}, {attrs})", ATOM
    axes = tensors[ins[0]]["axes"]
    at = n["attrs"]
    if p == "Elementwise" and n["fn"] in INFIX:
        sym, prec = INFIX[n["fn"]]
        (lt, lp), (rt, rp) = ref(ins[0]), ref(ins[1])
        lt = lt if lp >= prec else f"({lt})"
        rt = rt if rp > prec and rp != MATMUL else f"({rt})"
        return f"{lt} {sym} {rt}", prec
    if p == "Elementwise" and n["fn"] in UNARY:
        return f"{n['fn']}({ref(ins[0])[0]})", ATOM
    if p == "Contraction":
        over, right = at["over"], tensors[ins[1]]["axes"]
        assert over == [axes[-1]], f"{n['out']}: '@' contracts the left operand's last axis"
        if right[-1] == over[0]:
            return f"{atom(ins[0])} @ {atom(ins[1])}.T", MATMUL
        assert right[-2] == over[0], f"{n['out']}: no '@' spelling for {right} over {over}"
        return f"{atom(ins[0])} @ {atom(ins[1])}", MATMUL
    if p == "Gather":
        return f"{atom(ins[0])}[{ref(ins[1])[0]}]", ATOM
    if p == "Slice":
        parts = [":"] * len(axes)
        parts[axes.index(at["axis"])] = f"{at['start']}:{at['stop']}"
        return f"{atom(ins[0])}[{', '.join(parts)}]", ATOM
    if p == "Reshape":
        return f"reshape({ref(ins[0])[0]}, [{','.join(at['to'])}])", ATOM
    if p == "Permute":
        return f"permute({ref(ins[0])[0]}, [{','.join(map(str, at['perm']))}])", ATOM
    if p == "Rename":
        return f"relabel({ref(ins[0])[0]}, {at['from']} -> {at['to']})", ATOM
    raise NotImplementedError(f"{n['out']}: no TEL spelling for {p} {n.get('fn')}")


lines = [f'# The forward pass of "{g["model"]["name"]}", printed from {args.graph} by tools/viz_to_tel.py.', ""]
for a in g["axes"]:
    lines.append(f"axis {a['name']}" if a["symbol"] == a["name"] else f"axis {a['name']} = {a['symbol'] or a['size']}")
lines.append("")
for t in g["tensors"]:
    if t["role"] in ("input", "parameter", "buffer"):
        lines.append(f"{ident[t['name']]} : {DTYPE[t['dtype']]}[{','.join(t['axes'])}]")
block = None
for n in nodes:
    path = n["module"].split(".")
    here = path[2] if path[:2] == ["transformer", "blocks"] else path[0]
    if here != block:
        block = here
        lines += ["", f"# block {here}" if here.isdigit() else f"# {n['module']}"]
    expr[n["out"]] = render(n)
    if n["out"] in named:
        lines.append(f"{ident[n['out']]} = {expr.pop(n['out'])[0]}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
statements = sum(1 for n in nodes if n["out"] in named)
print(f"{len(g['ops'])} ops -> {len(nodes)} TEL nodes ({len(regions)} composites) -> {statements} statements -> {out}")
