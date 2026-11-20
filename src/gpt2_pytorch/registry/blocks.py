"""The block library (master §6, design doc §8): Python functions on Program, ported from TensormorphDesignUI
viewport/graphs/arch.mjs (rms 91-98, rope 101-111, softmax 113-121, silu 123, gqa 126-157, swiglu 159-166,
experts 169-178). Each block emits exactly the ops the JS block emits, in the same order and with the same names, so a
descriptor over these blocks rebuilds the arch.mjs goldens. Program and viz_graph are reused, never re-implemented."""
import math
from contextlib import contextmanager

from ..tensorprog.program import Program

EXAMPLE = {"B": 1, "T": 8}
SYMBOLS = {"B": "B", "T": "T", "q": "T", "k": "T"}  # R-52: B and T stay symbolic
ROLES = {"T": "token", "q": "token", "k": "token", "D": "feature", "Dq": "feature", "Dkv": "feature", "Dh": "feature",
         "Dh2": "feature", "F": "feature", "Fe": "feature", "V": "vocab", "B": "lane", "H": "lane", "Hkv": "lane", "E": "lane",
         "QR": "feature", "QH": "feature", "Dqk": "feature", "Dn": "feature", "Dr": "feature", "Dr2": "feature",
         "KA": "feature", "KR": "feature", "KH": "feature", "Dkvh": "feature", "Dv": "feature", "HV": "feature"}


class Builder:
    """Program plus the arch.mjs builder's conveniences: module context, scheme-keyed parameters, shorthands."""

    def __init__(self, dims: dict, root: str, scheme: dict, fmt: dict):
        self.p = Program(dict(dims), symbols={a: s for a, s in SYMBOLS.items() if a in dims})
        self.p.module, self.p.module_type = root, ""
        self.scheme, self.fmt = scheme, fmt
        self.routed = []  # (first op, end op, router op, K) of each experts block (D8 annotation)

    @contextmanager
    def within(self, module: str, module_type: str):
        saved = self.p.module, self.p.module_type
        self.p.module, self.p.module_type = module, module_type
        try:
            yield
        finally:
            self.p.module, self.p.module_type = saved

    def size(self, axis: str) -> int:
        return self.p.dims[axis]

    def key(self, role: str, **kw) -> tuple[str | None, str | None]:
        """(checkpoint_key, view) for a scheme role; a {"view": text} entry is a keyless parameter."""
        t = self.scheme.get(role)
        if t is None:
            return None, None
        kw.setdefault("what", "weight")
        if isinstance(t, dict) and "view" not in t:  # per-field names, e.g. TF-converted LayerNorm gamma / beta
            t = t[kw["what"]]
        if isinstance(t, dict):
            return None, t["view"].format(**self.fmt, **kw)
        return t.format(**self.fmt, **kw), None

    def param(self, name: str, axes, role: str, **kw) -> str:
        if name in self.p.tensors:
            return name
        key, view = self.key(role, **kw)
        return self.p.declare(name, "parameter", axes, checkpoint_key=key, state_dict_key=key, view=view)

    def buffer(self, name: str, axes, view: str, dtype="float32") -> str:
        return name if name in self.p.tensors else self.p.declare(name, "buffer", axes, dtype, view=view)

    def constant(self, name: str, value, axes=()) -> str:
        return name if name in self.p.tensors else self.p.declare(name, "constant", axes, value=value)

    def E(self, fn, out, x, y=None):
        return self.p.elementwise(fn, out, x) if y is None else self.p.elementwise(fn, out, x, y)

    def R(self, fn, out, x, axis):
        return self.p.reduce(fn, out, x, axis)

    def C(self, out, x, y, over):
        return self.p.contract(out, x, y, [over] if isinstance(over, str) else over)

    def permute(self, out, x, perm):
        axes = self.p.axes(x)
        return self.p.permute(out, x, [axes[j] for j in perm])

    def reshape(self, out, x, to):
        return self.p.reshape(out, x, to)

    def rename(self, out, x, old, new):
        return self.p.rename(out, x, old, new)

    def slice(self, out, x, axis, start, stop, to):
        return self.p.slice(out, x, axis, start, stop, to)

    def gather(self, out, table, index, axis):
        return self.p.gather(out, table, index, axis)


# ---- the pieces --------------------------------------------------------------------------------------------------
def rms(b: Builder, out, x, g, axis, plus_one=False):
    """RMSNorm: x · rsqrt(mean_axis(x²) + eps) · g; Gemma scales by (1 + g)"""
    b.E("mul", f"{out}.sq", x, x)
    b.R("mean", f"{out}.ms", f"{out}.sq", axis)
    b.E("add", f"{out}.ms_eps", f"{out}.ms", "c_eps")
    b.E("rsqrt", f"{out}.rstd", f"{out}.ms_eps")
    b.E("mul", f"{out}.normalized", x, f"{out}.rstd")
    if plus_one:
        g = b.E("add", f"{out}.g1", g, b.constant("c_one", 1.0))
    return b.E("mul", out, f"{out}.normalized", g)


def gelu_tanh(b: Builder, out, x):
    """0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x³))): the op order of tensorprog/lower.py lower_gelu"""
    a, c = b.constant("c_gelu_a", 0.044715), b.constant("c_gelu_b", math.sqrt(2.0 / math.pi))
    half, one = b.constant("c_half", 0.5), b.constant("c_one", 1.0)
    b.E("mul", f"{out}.squared", x, x)
    b.E("mul", f"{out}.cubic", f"{out}.squared", x)
    b.E("mul", f"{out}.cubic_a", f"{out}.cubic", a)
    b.E("add", f"{out}.inner_sum", x, f"{out}.cubic_a")
    b.E("mul", f"{out}.inner", f"{out}.inner_sum", c)
    b.E("tanh", f"{out}.tanh", f"{out}.inner")
    b.E("add", f"{out}.one_plus", f"{out}.tanh", one)
    b.E("mul", f"{out}.half_x", x, half)
    return b.E("mul", out, f"{out}.half_x", f"{out}.one_plus")


def layernorm(b: Builder, out, x, g, beta, eps_axis="D"):
    """(x - mean) / sqrt(var + eps) * g + b in the op order of tensorprog/lower.py lower_layernorm (TEL composite)"""
    with b.p.composite("layernorm", x, axis=eps_axis):
        mean = b.R("mean", f"{out}.mean", x, eps_axis)
        centered = b.E("sub", f"{out}.centered", x, mean)
        var = b.R("mean", f"{out}.var", b.E("mul", f"{out}.squared", centered, centered), eps_axis)
        rstd = b.E("rsqrt", f"{out}.rstd", b.E("add", f"{out}.var_eps", var, "c_eps"))
        normalized = b.E("mul", f"{out}.normalized", centered, rstd)
    return b.E("add", out, b.E("mul", f"{out}.scaled", normalized, g), beta)


def gelu_erf(b: Builder, out, x):
    """0.5 x (1 + erf(x / sqrt 2)): BERT's exact GELU"""
    half, one, r2 = b.constant("c_half", 0.5), b.constant("c_one", 1.0), b.constant("c_sqrt2", math.sqrt(2.0))
    e = b.E("erf", f"{out}.erf", b.E("div", f"{out}.scaled", x, r2))
    return b.E("mul", out, b.E("mul", f"{out}.half_x", x, half), b.E("add", f"{out}.one_plus", e, one))


def linear(b: Builder, out, x, w_name, axes, role, i, over, bias=True):
    y = b.C(f"{out}.mm" if bias else out, x, b.param(w_name, axes, role, i=i, what="weight"), over)
    return b.E("add", out, y, b.param("b" + w_name[1:], axes[:1], role, i=i, what="bias")) if bias else y


def encoder_layer(b: Builder, h: dict, i: int, at: str, x: str, out: str):
    """BERT-family post-norm layer: bidirectional MHA, residual + LayerNorm, GELU FFN, residual + LayerNorm"""
    with b.within(f"{at}.attention", "SelfAttention"):
        q = linear(b, f"Q_{i}.lin", x, f"Wq_{i}", ["Dq", "D"], "q", i, "D")
        k = linear(b, f"K_{i}.lin", x, f"Wk_{i}", ["Dq", "D"], "k", i, "D")
        v = linear(b, f"V_{i}.lin", x, f"Wv_{i}", ["Dq", "D"], "v", i, "D")
        b.permute(f"Q_{i}.perm", b.reshape(f"Q_{i}.heads", q, ["B", "T", "H", "Dh"]), [0, 2, 1, 3])
        b.rename(f"Q_{i}", f"Q_{i}.perm", "T", "q")
        b.permute(f"K_{i}.perm", b.reshape(f"K_{i}.heads", k, ["B", "T", "H", "Dh"]), [0, 2, 1, 3])
        b.rename(f"K_{i}", f"K_{i}.perm", "T", "k")
        b.permute(f"V_{i}.perm", b.reshape(f"V_{i}.heads", v, ["B", "T", "H", "Dh"]), [0, 2, 1, 3])
        b.rename(f"V_{i}", f"V_{i}.perm", "T", "k")
        s = b.E("div", f"S_{i}", b.C(f"S_{i}.qk", f"Q_{i}", f"K_{i}", "Dh"), "c_sqrt_dh")  # no mask: bidirectional
        softmax(b, f"A_{i}", s)
        b.C(f"C_{i}", f"A_{i}", f"V_{i}", "k")
        b.permute(f"C_{i}.perm", b.rename(f"C_{i}.rename", f"C_{i}", "q", "T"), [0, 2, 1, 3])
        b.reshape(f"C_{i}.merged", f"C_{i}.perm", ["B", "T", "Dq"])
        o = linear(b, f"O_{i}", f"C_{i}.merged", f"Wo_{i}", ["D", "Dq"], "o", i, "Dq")
    with b.within(f"{at}.attention.output.LayerNorm", "LayerNorm"):
        xa = layernorm(b, f"Xa_{i}", b.E("add", f"Xa_{i}.res", x, o), b.param(f"gan_{i}", ["D"], "attn_ln", i=i, what="weight"),
                       b.param(f"ban_{i}", ["D"], "attn_ln", i=i, what="bias"))
    with b.within(f"{at}.intermediate", "Intermediate"):
        g = gelu_erf(b, f"G_{i}", linear(b, f"G_{i}.lin", xa, f"Wi_{i}", ["F", "D"], "inter", i, "D"))
    with b.within(f"{at}.output", "Output"):
        m = linear(b, f"M_{i}", g, f"Wm_{i}", ["D", "F"], "out", i, "F")
    with b.within(f"{at}.output.LayerNorm", "LayerNorm"):
        return layernorm(b, out, b.E("add", f"{out}.res", xa, m), b.param(f"gon_{i}", ["D"], "out_ln", i=i, what="weight"),
                         b.param(f"bon_{i}", ["D"], "out_ln", i=i, what="bias"))


def softcap(b: Builder, out, x, cap):
    """cap · tanh(x / cap) (Gemma 2 attention and final logits)"""
    b.E("div", f"{out}.scaled", x, cap)
    b.E("tanh", f"{out}.tanh", f"{out}.scaled")
    return b.E("mul", out, f"{out}.tanh", cap)


def _p7(v: float) -> float:
    return float(f"{v:.7g}")  # JS +v.toPrecision(7)


def rope(b: Builder, out, x, pos, dim, theta, tag=""):
    """Rotary embedding on x [..., pos, dim]: x·cos + rotate_half(x)·sin, rotate_half a fixed map [dim, dim2]
    (HF layout: dimension d pairs with d ± dim/2; frequency θ^(−2(d mod dim/2)/dim) at position t)."""
    n, T = b.size(dim), b.size(pos)
    half = n // 2
    R = b.constant(f"R_{dim}", [[(-1 if d == e + half else 0) if e < half else (1 if d == e - half else 0) for e in range(n)]
                                for d in range(n)], [dim, dim + "2"])
    ang = lambda t, d: t * theta ** (-2 * (d % half) / n)  # noqa: E731
    table = lambda f: [[_p7(f(ang(t, d))) for d in range(n)] for t in range(T)]  # noqa: E731
    cos = b.constant(f"cos_{pos}_{dim}{tag}", table(math.cos), [pos, dim])
    sin = b.constant(f"sin_{pos}_{dim}{tag}", table(math.sin), [pos, dim])
    b.C(f"{out}.rot", x, R, dim)
    b.rename(f"{out}.rot_r", f"{out}.rot", dim + "2", dim)
    b.E("mul", f"{out}.cos", x, cos)
    b.E("mul", f"{out}.sin", f"{out}.rot_r", sin)
    return b.E("add", out, f"{out}.cos", f"{out}.sin")


def softmax(b: Builder, out, s):
    """softmax over k (the viewer's softmax region): exp(s − max) / Σ exp(s − max)"""
    with b.p.composite("softmax", s, axis="k"):
        b.R("max", f"{out}.max", s, "k")
        b.E("sub", f"{out}.shifted", s, f"{out}.max")
        b.E("exp", f"{out}.exp", f"{out}.shifted")
        b.R("sum", f"{out}.sum", f"{out}.exp", "k")
        b.E("div", out, f"{out}.exp", f"{out}.sum")
    return out


def silu(b: Builder, out, x):
    b.E("sigmoid", f"{out}.sig", x)
    return b.E("mul", out, x, f"{out}.sig")


def mask(b: Builder, i):
    """the additive causal mask log(M01), one per layer"""
    return b.E("log", f"Mask_{i}", "M01")


def gqa(b: Builder, h: dict, i: int, at: str, theta_key="theta", tag="", use_rope=True):
    """grouped-query attention: q heads grouped by the kv head they share (Llama, Qwen2/3, Mistral, Phi-3, Gemma, Mixtral).
    Flags: qk_norm (RMSNorm over Dh after the head split, Qwen3 / Gemma 3), attn_cap (Gemma 2 soft-capping)."""
    n = f"N1_{i}"
    theta = h.get(theta_key)
    one = bool(h.get("plus_one"))

    def proj(sym, w, axis, part):
        with b.within(f"{at}.{part}", "Linear"):
            W = b.param(f"W{w}_{i}", [axis, "D"], w, i=i, what="weight")
            y = b.C(f"{sym}_{i}.mm", n, W, "D")
            if h.get("qkv_bias") and part != "o_proj":
                y = b.E("add", f"{sym}_{i}.lin", y, b.param(f"b{w}_{i}", [axis], w, i=i, what="bias"))
            return y

    q, k, v = proj("Q", "q", "Dq", "q_proj"), proj("K", "k", "Dkv", "k_proj"), proj("V", "v", "Dkv", "v_proj")
    with b.within(at, "Attention"):
        qh = b.reshape(f"Q_{i}.heads", q, ["B", "T", "Hkv", "H", "Dh"])
        if h.get("qk_norm"):
            with b.within(f"{at}.q_norm", "RMSNorm"):
                qh = rms(b, f"Q_{i}.normed", qh, b.param(f"gqn_{i}", ["Dh"], "q_norm", i=i), "Dh", one)
        b.permute(f"Q_{i}.perm", qh, [0, 2, 3, 1, 4])
        if use_rope:
            b.rename(f"Q_{i}.q", f"Q_{i}.perm", "T", "q")
            rope(b, f"Q_{i}", f"Q_{i}.q", "q", "Dh", theta, tag)
        else:  # Jamba's attention has no positional encoding
            b.rename(f"Q_{i}", f"Q_{i}.perm", "T", "q")
        kh = b.reshape(f"K_{i}.heads", k, ["B", "T", "Hkv", "Dh"])
        if h.get("qk_norm"):
            with b.within(f"{at}.k_norm", "RMSNorm"):
                kh = rms(b, f"K_{i}.normed", kh, b.param(f"gkn_{i}", ["Dh"], "k_norm", i=i), "Dh", one)
        b.permute(f"K_{i}.perm", kh, [0, 2, 1, 3])
        if use_rope:
            b.rename(f"K_{i}.k", f"K_{i}.perm", "T", "k")
            rope(b, f"K_{i}", f"K_{i}.k", "k", "Dh", theta, tag)
        else:
            b.rename(f"K_{i}", f"K_{i}.perm", "T", "k")
        b.reshape(f"V_{i}.heads", v, ["B", "T", "Hkv", "Dh"])
        b.permute(f"V_{i}.perm", f"V_{i}.heads", [0, 2, 1, 3])
        b.rename(f"V_{i}", f"V_{i}.perm", "T", "k")
        b.C(f"S_{i}.qk", f"Q_{i}", f"K_{i}", "Dh")
        s = b.E("div", f"S_{i}", f"S_{i}.qk", "c_sqrt_dh")
        if h.get("attn_cap"):
            s = softcap(b, f"S_{i}.capped", s, "c_attn_cap")
        b.E("add", f"S_{i}.masked", s, mask(b, i))
        softmax(b, f"A_{i}", f"S_{i}.masked")
        b.C(f"C_{i}", f"A_{i}", f"V_{i}", "k")
        b.rename(f"C_{i}.rename", f"C_{i}", "q", "T")
        b.permute(f"C_{i}.perm", f"C_{i}.rename", [0, 3, 1, 2, 4])
        b.reshape(f"C_{i}.merged", f"C_{i}.perm", ["B", "T", "Dq"])
    with b.within(f"{at}.o_proj", "Linear"):
        return b.C(f"O_{i}", f"C_{i}.merged", b.param(f"Wo_{i}", ["D", "Dq"], "o", i=i, what="weight"), "Dq")


def swiglu(b: Builder, h: dict, i: int, at: str, F="F", prefix="", roles=("gate", "up", "down"),
           syms=("G", "Ga", "U", "Hm", "M")):
    """SwiGLU MLP: down(silu(gate(x)) · up(x)). Parameters are W<prefix>{g,u,d}_i; `roles` are the scheme keys and
    `syms` the (gate, act, up, product, out) symbols (DeepSeek's shared expert: prefix "s", Sg/Sa/Su/Sh/Ms)."""
    gate, act, up, prod, out = (f"{s}_{i}" for s in syms)

    def lin(sym, part, axes, x, over, role):
        with b.within(f"{at}.{part}", "Linear"):
            return b.C(sym, x, b.param(f"W{prefix}{part[0]}_{i}", axes, role, i=i), over)

    lin(gate, "gate_proj", [F, "D"], f"N2_{i}", "D", roles[0])
    if h.get("act") == "gelu_tanh":
        with b.within(f"{at}.act_fn", "GELUTanh"):
            gelu_tanh(b, act, gate)
    else:
        with b.within(f"{at}.act_fn", "SiLU"):
            silu(b, act, gate)
    lin(up, "up_proj", [F, "D"], f"N2_{i}", "D", roles[1])
    with b.within(at, "MLP"):
        b.E("mul", prod, act, up)
    return lin(out, "down_proj", ["D", F], prod, F, roles[2])


def experts(b: Builder, i: int, at: str, x: str, k: int | None = None):
    """every expert, contracted over E downstream and weighted by the router's top-k choice (Sel); with `k` the ops are
    recorded for the routed annotation (D8)"""
    start = len(b.p.ops)
    if k is not None:
        b.routed.append((start, None, b.p.tensors[f"R_{i}"].produced_by, k))
    with b.within(f"{at}.experts", "Experts"):
        W = lambda part, axes: b.param(f"W{part}e_{i}", axes, f"experts_{part}", i=i)  # noqa: E731
        b.C(f"Hg_{i}", x, W("g", ["E", "Fe", "D"]), "D")
        silu(b, f"Hs_{i}", f"Hg_{i}")
        b.C(f"Hu_{i}", x, W("u", ["E", "Fe", "D"]), "D")
        b.E("mul", f"Hh_{i}", f"Hs_{i}", f"Hu_{i}")
        b.C(f"Y_{i}", f"Hh_{i}", W("d", ["E", "D", "Fe"]), "Fe")
    if k is not None:
        b.routed[-1] = (start, len(b.p.ops), b.routed[-1][2], k)


def mlp_swiglu(b, h, i, layer):
    return swiglu(b, h, i, f"{layer}.{h.get('mlp_module', 'mlp')}")


def router(b: Builder, i: int, moe: str):
    with b.within(f"{moe}.gate", "Linear"):
        return b.C(f"R_{i}", f"N2_{i}", b.param(f"Wr_{i}", ["E", "D"], "router", i=i), "D")


def moe_mixtral(b, h, i, layer):
    """sparse MoE (arch.mjs mixtralGraph 236-242): softmax over the router's top-k experts, every expert drawn"""
    moe = f"{layer}.block_sparse_moe"
    router(b, i, moe)
    with b.within(moe, "SparseMoE"):
        b.buffer(f"Sel_{i}", ["B", "T", "E"], f"the router's top-{h['K']} choice for each token: 1 for a chosen expert (runtime)")
        b.E("exp", f"Gk_{i}.exp", f"R_{i}")
        b.E("mul", f"Gk_{i}.chosen", f"Gk_{i}.exp", f"Sel_{i}")
        b.R("sum", f"Gk_{i}.sum", f"Gk_{i}.chosen", "E")
        b.E("div", f"Gk_{i}", f"Gk_{i}.chosen", f"Gk_{i}.sum")
    experts(b, i, moe, f"N2_{i}", h["K"])
    with b.within(moe, "SparseMoE"):
        b.E("mul", f"Yw_{i}", f"Y_{i}", f"Gk_{i}")
        return b.R("sum", f"M_{i}", f"Yw_{i}", "E")


def moe_qwen(b, h, i, layer):
    """Qwen2/3-MoE: softmax router over E, top-k chosen, renormalized over the chosen when norm_topk_prob (then the same
    ops as mixtral's softmax over the chosen), every expert drawn"""
    moe = f"{layer}.{h.get('mlp_module', 'mlp')}"
    router(b, i, moe)
    with b.within(moe, "SparseMoeBlock"):
        b.buffer(f"Sel_{i}", ["B", "T", "E"], f"the router's top-{h['K']} choice of softmax(R): 1 for a chosen expert (runtime)")
        b.E("exp", f"Gk_{i}.exp", f"R_{i}")
        if h.get("norm_topk", 1):
            b.E("mul", f"Gk_{i}.chosen", f"Gk_{i}.exp", f"Sel_{i}")
            b.R("sum", f"Gk_{i}.sum", f"Gk_{i}.chosen", "E")
            b.E("div", f"Gk_{i}", f"Gk_{i}.chosen", f"Gk_{i}.sum")
        else:
            b.R("sum", f"Gk_{i}.esum", f"Gk_{i}.exp", "E")
            b.E("div", f"Gk_{i}.score", f"Gk_{i}.exp", f"Gk_{i}.esum")
            b.E("mul", f"Gk_{i}", f"Gk_{i}.score", f"Sel_{i}")
    experts(b, i, moe, f"N2_{i}", h["K"])
    with b.within(moe, "SparseMoeBlock"):
        b.E("mul", f"Yw_{i}", f"Y_{i}", f"Gk_{i}")
        return b.R("sum", f"M_{i}", f"Yw_{i}", "E")


def moe_deepseek(b, h, i, layer):
    """DeepSeek MoE (arch.mjs deepseekGraph 301-322): sigmoid router, top-k normalized and scaled, routed experts as
    keyless stacked views, plus the shared expert"""
    moe = f"{layer}.mlp"
    router(b, i, moe)
    softmax_router = h.get("scoring") == "softmax"  # DeepSeek-V2; V3 scores with sigmoid
    with b.within(moe, "DeepseekMoE"):
        if softmax_router:
            b.buffer(f"Sel_{i}", ["B", "T", "E"], f"the router's top-{h['K']} choice of softmax(R): 1 for a chosen expert (runtime)")
            b.E("exp", f"Gk_{i}.exp", f"R_{i}")
            b.R("sum", f"Gk_{i}.esum", f"Gk_{i}.exp", "E")
            b.E("div", f"Gk_{i}.score", f"Gk_{i}.exp", f"Gk_{i}.esum")
        else:
            b.buffer(f"Sel_{i}", ["B", "T", "E"], f"the router's top-{h['K']} choice of sigmoid(R) + e_score_correction_bias, "
                                                  "group-limited: 1 for a chosen expert (runtime)")
            b.E("sigmoid", f"Gk_{i}.score", f"R_{i}")
        b.E("mul", f"Gk_{i}.chosen", f"Gk_{i}.score", f"Sel_{i}")
        chosen = f"Gk_{i}.chosen"
        if h.get("norm_topk", 1):
            b.R("sum", f"Gk_{i}.sum", chosen, "E")
            chosen = b.E("div", f"Gk_{i}.norm", chosen, f"Gk_{i}.sum")
        b.E("mul", f"Gk_{i}", chosen, "c_route")
    experts(b, i, moe, f"N2_{i}", h["K"])
    swiglu(b, h, i, f"{moe}.shared_experts", F="Fs" if "Fs" in b.p.dims else "Fe", prefix="s",
           roles=("shared_gate", "shared_up", "shared_down"), syms=("Sg", "Sa", "Su", "Sh", "Ms"))
    with b.within(moe, "DeepseekMoE"):
        b.E("mul", f"Yw_{i}", f"Y_{i}", f"Gk_{i}")
        b.R("sum", f"Mr_{i}", f"Yw_{i}", "E")
        return b.E("add", f"M_{i}", f"Mr_{i}", f"Ms_{i}")


def mla(b: Builder, h: dict, i: int, at: str, direct_q: bool = False):
    """DeepSeek multi-head latent attention (arch.mjs deepseekGraph 266-300): low-rank queries, one compressed key/value
    latent plus a shared rotary key; q·k over (nope ++ rope) is the sum of two contractions"""
    layer = at.rsplit(".", 1)[0]
    nope, rope_d, kv_rank, vhd = b.size("Dn"), b.size("Dr"), b.size("KR"), b.size("Dv")

    def lin(sym, part, axes, x, over, role):
        with b.within(f"{layer}.{part}", "Linear"):
            return b.C(sym, x, b.param(f"W{sym.split('_')[0].lower()}_{i}", axes, role, i=i), over)

    if direct_q:  # DeepSeek-V2-Lite: no query LoRA, one q_proj
        lin(f"Qb_{i}", "self_attn.q_proj", ["QH", "D"], f"N1_{i}", "D", "q")
    else:
        lin(f"Qa_{i}", "self_attn.q_a_proj", ["QR", "D"], f"N1_{i}", "D", "q_a")
        with b.within(f"{at}.q_a_layernorm", "RMSNorm"):
            rms(b, f"Qn_{i}", f"Qa_{i}", b.param(f"gqa_{i}", ["QR"], "q_a_norm", i=i), "QR")
        lin(f"Qb_{i}", "self_attn.q_b_proj", ["QH", "QR"], f"Qn_{i}", "QR", "q_b")
    lin(f"KVa_{i}", "self_attn.kv_a_proj_with_mqa", ["KA", "D"], f"N1_{i}", "D", "kv_a")
    with b.within(at, "MLAttention"):
        b.slice(f"Ckv_{i}", f"KVa_{i}", "KA", 0, kv_rank, "KR")
        b.slice(f"Kp_{i}.t", f"KVa_{i}", "KA", kv_rank, kv_rank + rope_d, "Dr")
    with b.within(f"{at}.kv_a_layernorm", "RMSNorm"):
        rms(b, f"Cn_{i}", f"Ckv_{i}", b.param(f"gkv_{i}", ["KR"], "kv_a_norm", i=i), "KR")
    lin(f"KVb_{i}", "self_attn.kv_b_proj", ["KH", "KR"], f"Cn_{i}", "KR", "kv_b")
    with b.within(at, "MLAttention"):
        b.reshape(f"Qh_{i}", f"Qb_{i}", ["B", "T", "H", "Dqk"])
        b.permute(f"Qh_{i}.perm", f"Qh_{i}", [0, 2, 1, 3])
        b.rename(f"Qh_{i}.q", f"Qh_{i}.perm", "T", "q")
        b.slice(f"Qn_{i}.nope", f"Qh_{i}.q", "Dqk", 0, nope, "Dn")
        b.slice(f"Qp_{i}.pe", f"Qh_{i}.q", "Dqk", nope, nope + rope_d, "Dr")
        rope(b, f"Qr_{i}", f"Qp_{i}.pe", "q", "Dr", h["theta"])
        b.reshape(f"KVh_{i}", f"KVb_{i}", ["B", "T", "H", "Dkvh"])
        b.permute(f"KVh_{i}.perm", f"KVh_{i}", [0, 2, 1, 3])
        b.rename(f"KVh_{i}.k", f"KVh_{i}.perm", "T", "k")
        b.slice(f"Kn_{i}", f"KVh_{i}.k", "Dkvh", 0, nope, "Dn")
        b.slice(f"V_{i}", f"KVh_{i}.k", "Dkvh", nope, nope + vhd, "Dv")
        b.rename(f"Kp_{i}", f"Kp_{i}.t", "T", "k")
        rope(b, f"Kr_{i}", f"Kp_{i}", "k", "Dr", h["theta"])
        b.C(f"S_{i}.nope", f"Qn_{i}.nope", f"Kn_{i}", "Dn")
        b.C(f"S_{i}.pe", f"Qr_{i}", f"Kr_{i}", "Dr")
        b.E("add", f"S_{i}.qk", f"S_{i}.nope", f"S_{i}.pe")
        b.E("div", f"S_{i}", f"S_{i}.qk", "c_sqrt_dh")
        b.E("add", f"S_{i}.masked", f"S_{i}", mask(b, i))
        softmax(b, f"A_{i}", f"S_{i}.masked")
        b.C(f"C_{i}", f"A_{i}", f"V_{i}", "k")
        b.rename(f"C_{i}.rename", f"C_{i}", "q", "T")
        b.permute(f"C_{i}.perm", f"C_{i}.rename", [0, 2, 1, 3])
        b.reshape(f"C_{i}.merged", f"C_{i}.perm", ["B", "T", "HV"])
    return lin(f"O_{i}", "self_attn.o_proj", ["D", "HV"], f"C_{i}.merged", "HV", "o")


def gqa_local(b, h, i, at):
    """Gemma 3's sliding-window layers: the local rotary base (`theta_local`), its own rope tables"""
    return gqa(b, h, i, at, "theta_local", "_local")


def _shift(b: Builder):
    """the causal-convolution tap table conv_shift[Kc, T, k]: tap j of output t reads input t + j - (Kc - 1)"""
    K, T = b.size("Kc"), b.size("T")
    return b.constant("conv_shift", [[[1.0 if s == t + j - (K - 1) else 0.0 for s in range(T)] for t in range(T)] for j in range(K)],
                      ["Kc", "T", "k"])


def causal_conv(b: Builder, out, x, w, C):
    """depthwise causal conv over T of x [B, T, C] with weight w [C, U, Kc] (tensorprog/qwen3_5.py's spelling)"""
    taps = b.C(f"{out}.taps", b.rename(f"{out}.k", x, "T", "k"), _shift(b), "k")  # [B, C, Kc, T]
    ww = b.reshape(f"{out}.w", w, [C, "Kc"])
    conv = b.R("sum", f"{out}.sum", b.E("mul", f"{out}.prod", taps, ww), "Kc")  # [B, C, T]
    return b.permute(out, conv, [0, 2, 1])


def lfm2_conv(b: Builder, h: dict, i: int, at: str):
    """LFM2 gated short convolution: B, C, x = in_proj(x); y = C · conv(B · x); out_proj"""
    layer, D = at.rsplit(".", 1)[0], b.size("D")
    cv = f"{layer}.conv"
    with b.within(f"{cv}.in_proj", "Linear"):
        bcx = b.C(f"Bcx_{i}", f"N1_{i}", b.param(f"Wbcx_{i}", ["D3", "D"], "conv_in", i=i), "D")
    with b.within(cv, "ShortConv"):
        bb = b.slice(f"Bc_{i}", bcx, "D3", 0, D, "D")
        cc = b.slice(f"Cc_{i}", bcx, "D3", D, 2 * D, "D")
        xx = b.slice(f"Xc_{i}", bcx, "D3", 2 * D, 3 * D, "D")
        bx = b.E("mul", f"Bx_{i}", bb, xx)
    with b.within(f"{cv}.conv", "CausalConv1d"):
        conv = causal_conv(b, f"Cv_{i}", bx, b.param(f"Wconv_{i}", ["D", "U", "Kc"], "conv_w", i=i), "D")
    with b.within(cv, "ShortConv"):
        y = b.E("mul", f"Yc_{i}", cc, conv)
    with b.within(f"{cv}.out_proj", "Linear"):  # weight [D, D]: the input side is renamed Dc
        return b.C(f"O_{i}", b.rename(f"Yc_{i}.r", y, "D", "Dc"), b.param(f"Wco_{i}", ["D", "Dc"], "conv_out", i=i), "Dc")


def _onehot(b: Builder, t: int):
    T = b.size("T")
    e = b.constant(f"e_{t}", [1.0 if s == t else 0.0 for s in range(T)], ["T"])
    eu = b.constant(f"eu_{t}", [[1.0 if s == t else 0.0] for s in range(T)], ["T", "U"])
    return e, eu


def mamba(b: Builder, h: dict, i: int, at: str):
    """Mamba-1 selective scan, unrolled over T (Jamba, Mamba): in_proj → x, z; causal conv + SiLU; x_proj → dt, B, C
    (Jamba: RMSNorms on each); dt = softplus(dt_proj); h_t = exp(dt_t A) h_{t-1} + (dt_t x_t) ⊗ B_t; y_t = C_t · h_t;
    y = (y + D x) · silu(z); out_proj. The outer products are contractions over the size-1 axis U. Step ops are named
    Ss_i.t<t>.* (scan annotation)."""
    layer = at.rsplit(".", 1)[0]
    m = f"{layer}.mamba"
    Di, N, R = b.size("Di"), b.size("N"), b.size("R")
    one = b.constant("c_one", 1.0)
    with b.within(f"{m}.in_proj", "Linear"):
        xz = b.C(f"Xz_{i}", f"N1_{i}", b.param(f"Wmi_{i}", ["Di2", "D"], "mamba_in", i=i), "D")
    with b.within(m, "Mamba"):
        x = b.slice(f"Xm_{i}.x", xz, "Di2", 0, Di, "Di")
        z = b.slice(f"Xm_{i}.z", xz, "Di2", Di, 2 * Di, "Di")
    with b.within(f"{m}.conv1d", "CausalConv1d"):
        conv = causal_conv(b, f"Cm_{i}", x, b.param(f"Wmc_{i}", ["Di", "U", "Kc"], "mamba_conv", i=i), "Di")
        if h.get("conv_bias"):
            conv = b.E("add", f"Cm_{i}.b", conv, b.param(f"bmc_{i}", ["Di"], "mamba_conv_bias", i=i))
        xc = silu(b, f"Xs_{i}", conv)
    with b.within(f"{m}.x_proj", "Linear"):
        xp = b.C(f"Xp_{i}", xc, b.param(f"Wmx_{i}", ["XP", "Di"], "mamba_x", i=i), "Di")
    with b.within(m, "Mamba"):
        dt = b.slice(f"Dt_{i}.raw", xp, "XP", 0, R, "R")
        bm = b.slice(f"Bm_{i}", xp, "XP", R, R + N, "N")
        cm = b.slice(f"Cs_{i}", xp, "XP", R + N, R + 2 * N, "N")
    if h.get("mamba_norms"):  # Jamba
        for nm, src, ax, role in (("dt", dt, "R", "mamba_dt_norm"), ("b", bm, "N", "mamba_b_norm"), ("c", cm, "N", "mamba_c_norm")):
            with b.within(f"{m}.{nm}_layernorm", "RMSNorm"):
                out = rms(b, f"{src}.n", src, b.param(f"g{nm}m_{i}", [ax], role, i=i), ax)
            dt, bm, cm = (out, bm, cm) if nm == "dt" else (dt, out, cm) if nm == "b" else (dt, bm, out)
    with b.within(f"{m}.dt_proj", "Linear"):
        dtp = b.E("add", f"Dt_{i}.lin", b.C(f"Dt_{i}.mm", dt, b.param(f"Wdt_{i}", ["Di", "R"], "mamba_dt", i=i), "R"),
                  b.param(f"bdt_{i}", ["Di"], "mamba_dt_bias", i=i))
    with b.within(m, "Mamba"):
        delta = b.E("log", f"Dt_{i}", b.E("add", f"Dt_{i}.exp1", b.E("exp", f"Dt_{i}.exp", dtp), one))  # softplus
        A = b.E("sub", f"Am_{i}", b.constant("c_zero", 0.0), b.E("exp", f"Am_{i}.exp", b.param(f"A_log_{i}", ["Di", "N"], "mamba_A", i=i)))
        Au = b.reshape(f"Am_{i}.u", A, ["U", "Di", "N"])
        state = out = None
        for t in range(b.size("T")):
            e, eu = _onehot(b, t)
            s = f"Ss_{i}.t{t}"
            dt_t = b.C(f"{s}.dt", delta, e, "T")  # [B, Di]
            x_t = b.C(f"{s}.x", xc, e, "T")
            b_t = b.reshape(f"{s}.Bu", b.C(f"{s}.B", bm, e, "T"), ["B", "U", "N"])
            c_t = b.C(f"{s}.C", cm, e, "T")  # [B, N]
            dbx = b.C(f"{s}.dBx", b.reshape(f"{s}.dxu", b.E("mul", f"{s}.dx", dt_t, x_t), ["B", "Di", "U"]), b_t, "U")
            if state is None:  # h_0 = dt_0 x_0 ⊗ B_0: the first step has nothing to decay
                state = dbx
            else:
                decay = b.E("exp", f"{s}.dA", b.C(f"{s}.dtA", b.reshape(f"{s}.dtu", dt_t, ["B", "Di", "U"]), Au, "U"))  # [B, Di, N]
                state = b.E("add", f"{s}.h", b.E("mul", f"{s}.decayed", state, decay), dbx)
            y_t = b.C(f"{s}.y", state, c_t, "N")  # [B, Di]
            placed = b.C(f"{s}.o", b.reshape(f"{s}.yu", y_t, ["B", "Di", "U"]), eu, "U")  # [B, Di, T]
            out = placed if out is None else b.E("add", f"{s}.acc", out, placed)
        y = b.permute(f"Ym_{i}", out, [0, 2, 1])
        y = b.E("add", f"Ym_{i}.skip", y, b.E("mul", f"Ym_{i}.dx", xc, b.param(f"Dm_{i}", ["Di"], "mamba_D", i=i)))
        y = b.E("mul", f"Ym_{i}.gated", y, silu(b, f"Zm_{i}", z))
    with b.within(f"{m}.out_proj", "Linear"):
        return b.C(f"O_{i}", y, b.param(f"Wmo_{i}", ["D", "Di"], "mamba_out", i=i), "Di")


def gqa_norope(b, h, i, at):
    return gqa(b, h, i, at, use_rope=False)


def mla_direct(b, h, i, at):
    return mla(b, h, i, at, direct_q=True)


ATTENTION = {"gqa": gqa, "gqa_local": gqa_local, "gqa_norope": gqa_norope, "mla": mla, "mla_direct": mla_direct,
             "lfm2_conv": lfm2_conv, "mamba": mamba}
MLP = {"swiglu": mlp_swiglu, "moe_mixtral": moe_mixtral, "moe_deepseek": moe_deepseek, "moe_qwen": moe_qwen}
