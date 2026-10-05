"""A model as a flat list of tensor operations. Every operand is a named tensor: parameters, inputs, constants
(rank-0 tensors) and every intermediate. Axes carry names (B, T, D, ...); elementwise operands align by axis name.
The primitive vocabulary follows TEL's TensorIR (Contraction, Elementwise, Reduce, Gather, Slice, Reshape, Permute, Rename).
"""
import math
import re
from dataclasses import asdict, dataclass

FORMAT = "tensormorph-tensor-program"
VERSION = 1
BINARY = {"add": "+", "sub": "-", "mul": "×", "div": "/"}
UNARY = {"exp", "tanh", "rsqrt", "log"}


@dataclass
class TensorDecl:
    name: str
    role: str  # parameter | buffer | input | constant | activation
    axes: tuple[str, ...]
    dtype: str
    shape: tuple[int, ...]  # example shape for the program's (B, T)
    checkpoint_key: str | None = None  # parameter / buffer: key in pytorch_model.bin (original HF layout, no transposition)
    view: str | None = None  # buffer cut applied when binding, e.g. "[0,0,:T,:T]"
    value: float | None = None  # constants
    produced_by: int | None = None  # op index
    pytorch_name: str | None = None  # same tensor in model(ids, return_debug=True).debug


@dataclass
class Op:
    index: int
    primitive: str  # Contraction | Elementwise | Reduce | Gather | Slice | Reshape | Permute | Rename
    fn: str | None  # Elementwise / Reduce function
    out: str
    inputs: list[str]
    attrs: dict
    module: str  # path of the nn.Module whose lowering emitted it
    module_type: str


def pytorch_name(symbol: str) -> str | None:
    """Name of the corresponding tensor in the pure model's debug dict (None if it has no counterpart)."""
    if symbol == "Xf":
        return "model.final_hidden"
    if symbol == "L":
        return "model.logits"
    if m := re.fullmatch(r"X_(\d+)", symbol):
        n = int(m.group(1))
        return "embedding.output" if n == 0 else f"block.{n - 1}.output"
    if m := re.fullmatch(r"(N1|Hq|Q|K|V|S|A|C|O|Xm|N2|G|Ga|M)_(\d+)", symbol):
        kind, layer = m.groups()
        suffix = {
            "N1": "ln_1.output", "Hq": "attention.qkv", "Q": "attention.q", "K": "attention.k", "V": "attention.v",
            "S": "attention.scores", "A": "attention.probs", "C": "attention.context", "O": "attention.output",
            "Xm": "attention_residual.output", "N2": "ln_2.output", "G": "mlp.fc.output",
            "Ga": "mlp.activation.output", "M": "mlp.output",
        }[kind]
        return f"block.{layer}.{suffix}"
    return None


class Program:
    def __init__(self, dims: dict[str, int]):
        self.dims = dims
        self.tensors: dict[str, TensorDecl] = {}
        self.ops: list[Op] = []
        self.aliases: dict[str, str] = {}  # extra checkpoint name -> tensor it shares (tied weights)
        self.elided: list[str] = []  # modules that emit no ops (dropout in eval mode)
        self.module = ""
        self.module_type = ""

    # ---- tensors -------------------------------------------------------------------------------------------
    def declare(self, name, role, axes, dtype="float32", **extra) -> str:
        assert name not in self.tensors, f"tensor {name} declared twice"
        axes = tuple(axes)
        self.tensors[name] = TensorDecl(name, role, axes, dtype, tuple(self.dims[a] for a in axes), **extra)
        return name

    def constant(self, name: str, value: float, dtype="float32") -> str:
        if name in self.tensors:
            assert self.tensors[name].value == value
            return name
        return self.declare(name, "constant", (), dtype, value=value)

    def axes(self, name: str) -> tuple[str, ...]:
        return self.tensors[name].axes

    # ---- operations ----------------------------------------------------------------------------------------
    def _emit(self, primitive, fn, out, inputs, out_axes, attrs=None, dtype=None) -> str:
        assert all(i in self.tensors for i in inputs), f"{out}: unknown input in {inputs}"
        dtype = dtype or next(self.tensors[i].dtype for i in inputs if self.tensors[i].role != "constant")
        index = len(self.ops)
        self.declare(out, "activation", out_axes, dtype, produced_by=index, pytorch_name=pytorch_name(out))
        self.ops.append(Op(index, primitive, fn, out, list(inputs), attrs or {}, self.module, self.module_type))
        return out

    def contract(self, out, a, b, over):
        """out = a · b summed over the named axes `over`; out axes: a's remaining axes, then b's new ones."""
        over = list(over)
        assert all(x in self.axes(a) and x in self.axes(b) for x in over), f"{out}: {over} not shared by {a}, {b}"
        axes = [x for x in self.axes(a) if x not in over] + [x for x in self.axes(b) if x not in self.axes(a) and x not in over]
        return self._emit("Contraction", None, out, [a, b], axes, {"over": over})

    def elementwise(self, fn, out, *inputs):
        assert fn in BINARY or fn in UNARY, fn
        axes: list[str] = []
        for i in inputs:
            axes += [x for x in self.axes(i) if x not in axes]
        return self._emit("Elementwise", fn, out, list(inputs), axes)

    def reduce(self, fn, out, x, axis):
        axes = [a for a in self.axes(x) if a != axis]
        return self._emit("Reduce", fn, out, [x], axes, {"axis": axis})

    def gather(self, out, table, index, axis):
        axes = list(self.axes(index)) + [a for a in self.axes(table) if a != axis]
        return self._emit("Gather", None, out, [table, index], axes, {"axis": axis}, dtype=self.tensors[table].dtype)

    def slice(self, out, x, axis, start, stop, to_axis):
        axes = [to_axis if a == axis else a for a in self.axes(x)]
        return self._emit("Slice", None, out, [x], axes, {"axis": axis, "start": start, "stop": stop, "to_axis": to_axis})

    def reshape(self, out, x, to_axes):
        return self._emit("Reshape", None, out, [x], to_axes, {"to": list(to_axes)})

    def permute(self, out, x, to_axes):
        perm = [self.axes(x).index(a) for a in to_axes]
        return self._emit("Permute", None, out, [x], to_axes, {"perm": perm})

    def rename(self, out, x, old, new):
        axes = [new if a == old else a for a in self.axes(x)]
        return self._emit("Rename", None, out, [x], axes, {"from": old, "to": new})

    # ---- description -----------------------------------------------------------------------------------------
    def ref(self, name: str) -> str:
        axes = self.axes(name)
        return f"{name}[{','.join(axes)}]" if axes else name

    def equation(self, op: Op) -> str:
        r, out = self.ref, self.ref(op.out)
        a = [r(i) for i in op.inputs]
        if op.primitive == "Contraction":
            return f"{out} = {a[0]} · {a[1]}   (Σ {','.join(op.attrs['over'])})"
        if op.primitive == "Elementwise":
            return f"{out} = {a[0]} {BINARY[op.fn]} {a[1]}" if op.fn in BINARY else f"{out} = {op.fn}({a[0]})"
        if op.primitive == "Reduce":
            return f"{out} = {op.fn}_{op.attrs['axis']}({a[0]})"
        if op.primitive == "Gather":
            return f"{out} = gather({a[0]}, {a[1]})   (rows of {op.attrs['axis']})"
        if op.primitive == "Slice":
            at = op.attrs
            return f"{out} = {op.inputs[0]}[{at['axis']} {at['start']}:{at['stop']}]"
        if op.primitive == "Permute":
            return f"{out} = permute({a[0]})"
        if op.primitive == "Rename":
            return f"{out} = rename({a[0]}, {op.attrs['from']}→{op.attrs['to']})"
        return f"{out} = reshape({a[0]})"

    def cost(self, op: Op) -> int:
        """Contraction: multiply-accumulates. Elementwise: output elements. Others: 0."""
        if op.primitive == "Contraction":
            axes = set(self.axes(op.inputs[0])) | set(self.axes(op.inputs[1]))
            return math.prod(self.dims[a] for a in axes)
        if op.primitive == "Elementwise":
            return math.prod(self.tensors[op.out].shape)
        return 0

    def is_multiplication(self, op: Op) -> bool:
        return op.primitive == "Contraction" or (op.primitive == "Elementwise" and op.fn == "mul")

    def to_dict(self, model_name: str, example: dict) -> dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "model": {"name": model_name},
            "example": example,
            "dims": self.dims,
            "aliases": self.aliases,
            "elided_modules": self.elided,
            "tensors": [asdict(t) for t in self.tensors.values()],
            "ops": [
                asdict(op) | {"equation": self.equation(op), "cost": self.cost(op), "multiplication": self.is_multiplication(op)}
                for op in self.ops
            ],
        }
