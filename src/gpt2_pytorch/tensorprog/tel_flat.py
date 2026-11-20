"""Graph ⇔ T♭ (master §9.6, T-2): the flat TEL text of a viz-graph — one statement per op, every tensor declared —
and back. A port of TensormorphDesignUI nodes/tel.js `printFlat` / `build` / `decodeExpr` (lines 18-66, 152-200) with
the two §9.6 fixes built in: tensor-valued constants are declared buffers (bound by tensor:// URI, TEL has no array
literal), and a contraction whose right operand has a batch axis the left lacks is spelled `contract(a, b, over=X)`.
Σ carries what TEL never spells (module, module_type, region, equation, keys, views, roles, graph-level fields), joined
by output symbol; `build` re-derives edges, groups, multiplications and totals through viz_graph().

python -m gpt2_pytorch.tensorprog.tel_flat GRAPH.json [--out FILE.tel]
"""
import copy
import json
import re
import sys
from functools import cache
from pathlib import Path

from .viz import viz_graph

DT = {"float32": "f32", "float64": "f64", "float16": "f16", "bfloat16": "bf16", "int64": "i64", "int32": "i32",
      "int8": "i8", "uint8": "u8", "bool": "bool"}
TD = {v: k for k, v in DT.items()}
BIN = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
NIB = {v: k for k, v in BIN.items()}
UNARY = {"exp", "log", "tanh", "rsqrt", "sigmoid", "sqrt", "relu", "gelu", "erf"}
REDUCE = {"mean", "sum", "max"}
GRAPH_FIELDS = ("source", "model", "example", "aliases", "elided_modules", "regions", "verification", "roles", "lanes",
                "checkpoint", "provenance", "components")
TENSOR_FIELDS = ("name", "role", "checkpoint_key", "view", "pytorch_name", "state_dict_key", "value")
OP_FIELDS = ("module", "module_type", "region", "equation", "cost", "multiplication", "execution")


@cache
def vocabulary() -> frozenset:
    return frozenset(json.loads((Path(__file__).with_name("tel_vocabulary.json")).read_text()))


def tel_id(name: str) -> str:
    i = name.replace(".", "_")
    return i + "_" if i in vocabulary() else i


def _js(v) -> str:
    from ..registry.build import _js_number  # JSON.stringify's number spelling, as TEL's printer keeps it
    return str(v) if isinstance(v, int) and not isinstance(v, bool) else _js_number(float(v))


def _num(v, dtype: str) -> str:
    if re.match(r"^(float|bfloat)", dtype) and float(v) == int(v):
        return f"{int(v)}.0"
    return _js(v)


def _rhs(o: dict, idf, T: dict, fix_contract: bool) -> str:
    a, b = (o["inputs"] + [None, None])[:2]
    at = o.get("attrs") or {}
    p = o["primitive"]
    if p == "Elementwise":
        return f"{idf(a)} {BIN[o['fn']]} {idf(b)}" if o["fn"] in BIN else f"{o['fn']}({idf(a)})"
    if p == "Reduce":
        return f"{o['fn']}({idf(a)}, axis={at['axis']})"
    if p == "Contraction":
        A, B, over = T[a]["axes"], T[b]["axes"], at.get("over") or []
        at_form = len(over) == 1 and A[-1] == over[0]
        if at_form and not (fix_contract and any(x not in A[:-2] for x in B[:-2])):
            tr = B[-1] == over[0] and not (len(B) == 2 and B[0] == over[0])
            return f"{idf(a)} @ {idf(b)}" + (".T" if tr else "")
        return f"contract({idf(a)}, {idf(b)}, over={', '.join(over)})"
    if p == "Gather":
        return f"{idf(a)}[{idf(b)}]"
    if p == "Slice":
        return f"{idf(a)}[" + ", ".join(f"{at['start']}:{at['stop']}" if x == at["axis"] else ":" for x in T[a]["axes"]) + "]"
    if p == "Reshape":
        return f"reshape({idf(a)}, [{', '.join(at['to'])}])"
    if p == "Permute":
        return f"permute({idf(a)}, [{', '.join(map(str, at['perm']))}])"
    if p == "Rename":
        return f"relabel({idf(a)}, {at['from']} -> {at['to']})"
    raise ValueError(f"no TEL spelling for {p}")


def print_flat(graph: dict, const_as_buffer: bool = True, fix_contract: bool = True) -> tuple[str, dict]:
    """(T♭ source, Σ). Σ = {"graph": graph-level fields, "tensors": {tel id: fields}, "ops": {tel id of out: fields},
    "dims": {axis: size}}."""
    g = copy.deepcopy(graph)
    if const_as_buffer:
        for t in g["tensors"]:
            if t["role"] == "constant" and isinstance(t["value"], list):
                t["role"], t["value"] = "buffer", None
    T = {t["name"]: t for t in g["tensors"]}
    idf = lambda n: tel_id(n)  # noqa: E731
    lines = []
    for a in g["axes"]:
        lines.append(f"axis {a['name']}" if a["symbol"] == a["name"] else f"axis {a['name']} = {a['symbol'] or a['size']}")
    for t in g["tensors"]:
        lines.append(f"{idf(t['name'])} : {DT.get(t['dtype'], t['dtype'])}" + (f"[{','.join(t['axes'])}]" if t["axes"] else ""))
        if t["role"] == "constant":
            lines.append(f"{idf(t['name'])} := {_num(t['value'], t['dtype'])}")
    for o in g["ops"]:
        lines.append(f"{idf(o['out'])} = {_rhs(o, idf, T, fix_contract)}")
    sigma = {"graph": {k: g[k] for k in GRAPH_FIELDS if k in g}, "dims": g["dims"],
             "tensors": {idf(t["name"]): {k: t[k] for k in TENSOR_FIELDS if k in t} for t in g["tensors"]},
             "ops": {idf(o["out"]): {k: o[k] for k in OP_FIELDS if k in o} for o in g["ops"]}}
    return "\n".join(lines), sigma  # TEL's printer ends without a newline


# ---- T♭ → graph --------------------------------------------------------------------------------------------------
_ID = r"[A-Za-z_]\w*"
_STMT = re.compile(rf"^({_ID}) = (.+)$")


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch in "[(" or -(ch in "])")
        cur += ch
    return out + ([cur.strip()] if cur.strip() else [])


def _decode(e: str, name, axes_of, out: str) -> dict:
    """One T♭ right-hand side → {primitive, fn, inputs, attrs}. `contract` may name several axes."""
    if m := re.fullmatch(rf"({_ID}) @ ({_ID})(\.T)?", e):
        return {"primitive": "Contraction", "fn": None, "inputs": [name(m[1]), name(m[2])], "attrs": {"over": [axes_of(m[1])[-1]]}}
    if m := re.fullmatch(rf"({_ID}) ([-+*/]) ({_ID})", e):
        return {"primitive": "Elementwise", "fn": NIB[m[2]], "inputs": [name(m[1]), name(m[3])], "attrs": {}}
    if m := re.fullmatch(rf"({_ID})\[(.+)\]", e):
        items = _split_args(m[2])
        A = axes_of(m[1])
        if len(items) == 1 and ":" not in items[0]:
            return {"primitive": "Gather", "fn": None, "inputs": [name(m[1]), name(items[0])], "attrs": {"axis": A[0]}}
        k = next(i for i, s in enumerate(items) if s != ":")
        start, stop = items[k].split(":")
        return {"primitive": "Slice", "fn": None, "inputs": [name(m[1])],
                "attrs": {"axis": A[k], "start": int(start or 0), "stop": int(stop), "to_axis": axes_of(out)[k]}}
    if m := re.fullmatch(rf"({_ID})\((.*)\)", e):
        f, args = m[1], _split_args(m[2])
        kw = {a.split("=", 1)[0].strip(): a.split("=", 1)[1].strip() for a in args if re.match(rf"{_ID}=", a)}
        pos = [a for a in args if not re.match(rf"{_ID}=", a)]
        if f in UNARY:
            return {"primitive": "Elementwise", "fn": f, "inputs": [name(pos[0])], "attrs": {}}
        if f in REDUCE:
            return {"primitive": "Reduce", "fn": f, "inputs": [name(pos[0])], "attrs": {"axis": kw["axis"]}}
        if f == "reshape":
            return {"primitive": "Reshape", "fn": None, "inputs": [name(pos[0])], "attrs": {"to": _split_args(pos[1][1:-1])}}
        if f == "permute":
            return {"primitive": "Permute", "fn": None, "inputs": [name(pos[0])], "attrs": {"perm": [int(x) for x in _split_args(pos[1][1:-1])]}}
        if f == "relabel":
            a, b = (x.strip() for x in pos[1].split("->"))
            return {"primitive": "Rename", "fn": None, "inputs": [name(pos[0])], "attrs": {"from": a, "to": b}}
        if f == "contract":  # over=X or over=X, Y: the remaining positional items after the two operands
            over = [kw["over"]] + pos[2:]
            return {"primitive": "Contraction", "fn": None, "inputs": [name(pos[0]), name(pos[1])], "attrs": {"over": over}}
    raise ValueError(f"not a T♭ expression: {e}")


def build(source: str, sigma: dict) -> dict:
    """The viz-graph a T♭ text and its Σ describe (edges, groups, multiplications and totals re-derived)."""
    back = {i: f["name"] for i, f in sigma["tensors"].items()}
    name = lambda i: back.get(i, i)  # noqa: E731
    axes, decls, consts, ops = [], {}, {}, []
    for line in source.splitlines():
        if not line.strip():
            continue
        if m := re.fullmatch(rf"axis ({_ID})(?: = (\S+))?", line):
            axes.append((m[1], m[2]))
        elif m := re.fullmatch(rf"({_ID}) : (\w+)(?:\[([\w,]*)\])?", line):
            decls[name(m[1])] = {"dtype": TD.get(m[2], m[2]), "axes": m[3].split(",") if m[3] else []}
        elif m := re.fullmatch(rf"({_ID}) := (\S+)", line):
            consts[name(m[1])] = m[2]
        elif m := _STMT.fullmatch(line):
            out = name(m[1])
            ops.append({"out": out, **_decode(m[2], name, lambda i: decls[name(i)]["axes"], m[1])})
        else:
            raise ValueError(f"not a T♭ line: {line}")
    dims = sigma["dims"]
    symbols = {a: (s if s and not s.isdigit() else None) for a, s in axes}
    produced = {o["out"]: i for i, o in enumerate(ops)}
    tensors = []
    for n, d in decls.items():
        f = sigma["tensors"].get(tel_id(n), {"name": n})
        role = "activation" if n in produced else f.get("role") or ("constant" if n in consts else "input")
        value = f.get("value")
        if n in consts:
            v = consts[n]
            value = float(v) if re.search(r"[.eE]", v) else int(v)
        tensors.append({"name": n, "role": role, "axes": d["axes"], "dtype": d["dtype"], "shape": [dims[a] for a in d["axes"]],
                        "checkpoint_key": f.get("checkpoint_key"), "view": f.get("view"), "value": value,
                        "produced_by": produced.get(n), "pytorch_name": f.get("pytorch_name"), "state_dict_key": f.get("state_dict_key")})
    graph_fields = sigma["graph"]
    prog_ops = []
    for i, o in enumerate(ops):
        f = sigma["ops"].get(tel_id(o["out"]), {})
        prog_ops.append({"index": i, "primitive": o["primitive"], "fn": o["fn"], "out": o["out"], "inputs": o["inputs"],
                         "attrs": o["attrs"], "module": f.get("module", ""), "module_type": f.get("module_type", ""),
                         "region": f.get("region"), "equation": f.get("equation", ""), "cost": f.get("cost", 0),
                         "multiplication": f.get("multiplication", False), **({"execution": f["execution"]} if "execution" in f else {})})
    prog = {"dims": dims, "axes": [{"name": a, "size": dims[a], "symbol": symbols[a]} for a, _ in axes], "tensors": tensors,
            "ops": prog_ops, **{k: graph_fields.get(k, d) for k, d in (("model", {"name": "model"}), ("example", {}), ("aliases", {}),
                                                                       ("elided_modules", []), ("regions", []), ("verification", []))}}
    if "roles" in graph_fields:
        prog["roles"] = graph_fields["roles"]
    g = _viz(prog, graph_fields.get("source", "tel_flat"))
    for k in ("lanes", "checkpoint", "provenance", "components"):
        if k in graph_fields:
            g[k] = graph_fields[k]
    return g


def _viz(prog: dict, source: str) -> dict:
    """viz_graph() without its GPT-2-specific single-output assertion."""
    outs = [o["out"] for o in prog["ops"] if not any(o["out"] in p["inputs"] for p in prog["ops"])]
    if outs == ["L"]:
        return viz_graph(prog, source)
    raise ValueError(f"graph outputs {outs}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    text, _ = print_flat(json.loads(a.graph.read_text()))
    (a.out.write_text(text) if a.out else sys.stdout.write(text))


if __name__ == "__main__":
    main()
