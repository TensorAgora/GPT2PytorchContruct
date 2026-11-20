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


def gqa(b: Builder, h: dict, i: int, at: str, theta_key="theta", tag=""):
    """grouped-query attention: q heads grouped by the kv head they share (Llama, Qwen2/3, Mistral, Phi-3, Gemma, Mixtral).
    Flags: qk_norm (RMSNorm over Dh after the head split, Qwen3 / Gemma 3), attn_cap (Gemma 2 soft-capping)."""
    n = f"N1_{i}"
    theta = h[theta_key]
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
        b.rename(f"Q_{i}.q", f"Q_{i}.perm", "T", "q")
        rope(b, f"Q_{i}", f"Q_{i}.q", "q", "Dh", theta, tag)
        kh = b.reshape(f"K_{i}.heads", k, ["B", "T", "Hkv", "Dh"])
        if h.get("qk_norm"):
            with b.within(f"{at}.k_norm", "RMSNorm"):
                kh = rms(b, f"K_{i}.normed", kh, b.param(f"gkn_{i}", ["Dh"], "k_norm", i=i), "Dh", one)
        b.permute(f"K_{i}.perm", kh, [0, 2, 1, 3])
        b.rename(f"K_{i}.k", f"K_{i}.perm", "T", "k")
        rope(b, f"K_{i}", f"K_{i}.k", "k", "Dh", theta, tag)
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
    return swiglu(b, h, i, f"{layer}.mlp")


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
    moe = f"{layer}.mlp"
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


def mla_direct(b, h, i, at):
    return mla(b, h, i, at, direct_q=True)


ATTENTION = {"gqa": gqa, "gqa_local": gqa_local, "mla": mla, "mla_direct": mla_direct}
MLP = {"swiglu": mlp_swiglu, "moe_mixtral": moe_mixtral, "moe_deepseek": moe_deepseek, "moe_qwen": moe_qwen}
