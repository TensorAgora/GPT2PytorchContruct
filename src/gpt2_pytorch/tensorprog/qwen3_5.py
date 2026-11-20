"""Lower a Qwen3.5 hybrid text model (Qwen3_5ForConditionalGeneration's language model) from its config.json alone.

Every layer is spelled in the same primitives as the DistilGPT2 program (program.py), following transformers'
modeling_qwen3_5.py arithmetic exactly:
- RMSNorm is zero-centred: x * rsqrt(mean(x^2) + eps) * (1 + w). The gated RMSNorm inside linear attention uses w as is.
- linear_attention layers are Gated DeltaNet: in_proj_qkv -> depthwise causal conv1d (a contraction with a constant
  shift table) -> SiLU -> split q, k, v; q and k L2-normalised, q / sqrt(Dk); beta = sigmoid(b);
  decay = exp(-exp(A_log) * softplus(a + dt_bias)); the delta-rule recurrence unrolled over the example's T tokens
  (one-hot contractions pick token t and put o_t back); gated RMSNorm with silu(z); out_proj.
- full_attention layers: q_proj holds (query, gate) per head; q_norm / k_norm; partial RoPE (the first
  rotary_dim of each head, pass-through for the rest, spelled full width so no concat is needed); causal GQA softmax;
  context * sigmoid(gate); o_proj. Text-only input gives all three M-RoPE position rows the same ids, so M-RoPE is RoPE.
- SwiGLU MLP, final norm, untied lm_head.
The vision tower and the MTP head do not run in a text forward pass and are not spelled.
"""
import math
from contextlib import contextmanager

from .program import Program

ROLES = {
    "B": "lane", "T": "token", "q": "token", "k": "token", "D": "feature", "V": "vocab", "F": "feature",
    # linear attention
    "C": "feature", "Dqk": "feature", "Dvv": "feature", "Hv": "lane", "Hk": "lane", "R": "lane", "Dk": "feature",
    "Dv": "feature", "Kc": "lane", "U": "lane",
    # full attention
    "Dqg": "feature", "Hkv": "lane", "G": "lane", "QG": "feature", "Dh": "feature", "Dh2": "feature", "Dkv": "feature",
    "Dq": "feature",
}


class _Lowering:
    def __init__(self, cfg: dict, batch: int, seq: int, prefix: str):
        c = self.c = cfg.get("text_config", cfg)
        self.prefix = prefix
        hk, hv, dh = c["linear_num_key_heads"], c["linear_num_value_heads"], c["head_dim"]
        self.T = seq
        self.p = Program({
            "B": batch, "T": seq, "q": seq, "k": seq, "D": c["hidden_size"], "V": c["vocab_size"], "F": c["intermediate_size"],
            "C": 2 * hk * c["linear_key_head_dim"] + hv * c["linear_value_head_dim"], "Dqk": hk * c["linear_key_head_dim"],
            "Dvv": hv * c["linear_value_head_dim"], "Hv": hv, "Hk": hk, "R": hv // hk, "Dk": c["linear_key_head_dim"],
            "Dv": c["linear_value_head_dim"], "Kc": c["linear_conv_kernel_dim"], "U": 1,
            "Dqg": 2 * c["num_attention_heads"] * dh, "Hkv": c["num_key_value_heads"],
            "G": c["num_attention_heads"] // c["num_key_value_heads"], "QG": 2 * dh, "Dh": dh, "Dh2": dh,
            "Dkv": c["num_key_value_heads"] * dh, "Dq": c["num_attention_heads"] * dh,
        }, symbols={"B": "B", "T": "T", "q": "T", "k": "T"})
        assert hv % hk == 0 and c["num_attention_heads"] % c["num_key_value_heads"] == 0

    @contextmanager
    def at(self, path: str, module_type: str):
        saved = self.p.module, self.p.module_type
        self.p.module, self.p.module_type = path, module_type
        try:
            yield
        finally:
            self.p.module, self.p.module_type = saved

    def param(self, name, axes, key):
        return self.p.declare(name, "parameter", axes, checkpoint_key=key, state_dict_key=key)

    def c_(self, name, value, axes=()):
        return self.p.constant(name, value, axes=axes)

    # ---- pieces -----------------------------------------------------------------------------------------------
    def rms(self, out, x, w, axis, centred=True):
        p, E = self.p, self.p.elementwise
        sq = E("mul", f"{out}.sq", x, x)
        ms = p.reduce("mean", f"{out}.ms", sq, axis)
        rstd = E("rsqrt", f"{out}.rstd", E("add", f"{out}.ms_eps", ms, self.c_("c_eps", self.c["rms_norm_eps"])))
        normed = E("mul", f"{out}.normalized", x, rstd)
        if centred:
            w = E("add", f"{w}.plus1", w, self.c_("c_one", 1.0))
        return E("mul", out, normed, w)

    def silu(self, out, x):
        return self.p.elementwise("mul", out, x, self.p.elementwise("sigmoid", f"{out}.sig", x))

    def linear(self, path, out, x, name, axes, key):
        with self.at(path, "Linear"):
            return self.p.contract(out, x, self.param(name, axes, key), [axes[1]])

    def l2norm(self, out, x, axis):
        E = self.p.elementwise
        ss = self.p.reduce("sum", f"{out}.ss", E("mul", f"{out}.sq", x, x), axis)
        return E("mul", out, x, E("rsqrt", f"{out}.rstd", E("add", f"{out}.ss_eps", ss, self.c_("c_l2_eps", 1e-6))))

    def onehot(self, t, with_u):
        rows = [[1.0 if s == t else 0.0] for s in range(self.T)] if with_u else [1.0 if s == t else 0.0 for s in range(self.T)]
        return self.c_(f"e{'u' if with_u else ''}_{t}", rows, ("T", "U") if with_u else ("T",))

    def rope_tables(self, pos):
        """Full-width tables over Dh: RoPE on the first rotary_dim, cos 1 / sin 0 / no rotation on the rest."""
        c, dh = self.c, self.p.dims["Dh"]
        rp = c.get("rope_parameters") or {}
        theta = rp.get("rope_theta", c.get("rope_theta", 10000.0))
        rd = int(dh * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        half = rd // 2
        ang = lambda t, d: t * theta ** (-2 * (d % half) / rd)  # noqa: E731
        cos = [[math.cos(ang(t, d)) if d < rd else 1.0 for d in range(dh)] for t in range(self.T)]
        sin = [[math.sin(ang(t, d)) if d < rd else 0.0 for d in range(dh)] for t in range(self.T)]
        rot = [[-1.0 if e < half and d == e + half else 1.0 if half <= e < rd and d == e - half else 0.0 for e in range(dh)]
               for d in range(dh)]
        return (self.c_(f"cos_{pos}", cos, (pos, "Dh")), self.c_(f"sin_{pos}", sin, (pos, "Dh")),
                self.c_("R_rot", rot, ("Dh", "Dh2")))

    def rope(self, out, x, pos):
        p, E = self.p, self.p.elementwise
        cos, sin, rot = self.rope_tables(pos)
        r = p.rename(f"{out}.rot_r", p.contract(f"{out}.rot", x, rot, ["Dh"]), "Dh2", "Dh")
        return E("add", out, E("mul", f"{out}.cos", x, cos), E("mul", f"{out}.sin", r, sin))

    # ---- layers -----------------------------------------------------------------------------------------------
    def linear_attention(self, i, x, at, key):
        p, E, T = self.p, self.p.elementwise, self.T
        k = lambda part: f"{key}.{part}"  # noqa: E731
        mixed = self.linear(f"{at}.in_proj_qkv", f"Hqkv_{i}", x, f"Wqkv_{i}", ("C", "D"), k("in_proj_qkv.weight"))
        z = self.linear(f"{at}.in_proj_z", f"Z_{i}", x, f"Wz_{i}", ("Dvv", "D"), k("in_proj_z.weight"))
        b = self.linear(f"{at}.in_proj_b", f"Bt_{i}", x, f"Wb_{i}", ("Hv", "D"), k("in_proj_b.weight"))
        a = self.linear(f"{at}.in_proj_a", f"At_{i}", x, f"Wa_{i}", ("Hv", "D"), k("in_proj_a.weight"))
        with self.at(f"{at}.conv1d", "CausalConv1d"):
            K = p.dims["Kc"]
            shift = self.c_("conv_shift", [[[1.0 if s == t + j - (K - 1) else 0.0 for s in range(T)] for t in range(T)] for j in range(K)],
                            ("Kc", "T", "k"))
            xs = p.contract(f"Cv_{i}.taps", p.rename(f"Cv_{i}.k", mixed, "T", "k"), shift, ["k"])  # [B, C, Kc, T]
            w = p.reshape(f"Cv_{i}.w", self.param(f"Wconv_{i}", ("C", "U", "Kc"), k("conv1d.weight")), ("C", "Kc"))
            conv = p.reduce("sum", f"Cv_{i}.sum", E("mul", f"Cv_{i}.prod", xs, w), "Kc")
            conv = p.permute(f"Cv_{i}", self.silu(f"Cv_{i}.act", conv), ("B", "T", "C"))
        with self.at(at, "GatedDeltaNet"):
            dqk, dvv = p.dims["Dqk"], p.dims["Dvv"]
            q = p.reshape(f"Ql_{i}.heads", p.slice(f"Ql_{i}.cut", conv, "C", 0, dqk, "Dqk"), ("B", "T", "Hk", "Dk"))
            kk = p.reshape(f"Kl_{i}.heads", p.slice(f"Kl_{i}.cut", conv, "C", dqk, 2 * dqk, "Dqk"), ("B", "T", "Hk", "Dk"))
            v = p.reshape(f"Vl_{i}", p.slice(f"Vl_{i}.cut", conv, "C", 2 * dqk, 2 * dqk + dvv, "Dvv"), ("B", "T", "Hk", "R", "Dv"))
            q = E("div", f"Ql_{i}", self.l2norm(f"Ql_{i}.l2", q, "Dk"), self.c_("c_sqrt_dk", math.sqrt(p.dims["Dk"])))
            kk = self.l2norm(f"Kl_{i}", kk, "Dk")
            beta = p.reshape(f"Beta_{i}", E("sigmoid", f"Beta_{i}.sig", b), ("B", "T", "Hk", "R"))
            ab = E("add", f"Gd_{i}.a", a, self.param(f"dt_bias_{i}", ("Hv",), k("dt_bias")))
            softplus = E("log", f"Gd_{i}.softplus", E("add", f"Gd_{i}.exp1", E("exp", f"Gd_{i}.exp", ab), self.c_("c_one", 1.0)))
            rate = E("mul", f"Gd_{i}.rate", softplus, E("exp", f"Gd_{i}.A", self.param(f"A_log_{i}", ("Hv",), k("A_log"))))
            decay = p.reshape(f"Gd_{i}", E("exp", f"Gd_{i}.exp_neg", E("sub", f"Gd_{i}.neg", self.c_("c_zero", 0.0), rate)),
                              ("B", "T", "Hk", "R"))
            # the delta rule, token by token: S [B,Hk,R,Dv,Dk]
            S = out = None
            for t in range(T):
                eu, e = self.onehot(t, True), self.onehot(t, False)
                s = f"Dn_{i}.t{t}"
                qt, kt = p.contract(f"{s}.q", q, eu, ["T"]), p.contract(f"{s}.k", kk, eu, ["T"])  # [B,Hk,Dk,U]
                vt, bt = p.contract(f"{s}.v", v, eu, ["T"]), p.contract(f"{s}.beta", beta, e, ["T"])
                if S is None:
                    delta = E("mul", f"{s}.delta", vt, bt)
                else:
                    S = E("mul", f"{s}.decayed", S, p.contract(f"{s}.g", decay, e, ["T"]))
                    mem = p.contract(f"{s}.mem", S, kt, ["Dk"])  # [B,Hk,R,Dv,U]
                    delta = E("mul", f"{s}.delta", E("sub", f"{s}.err", vt, mem), bt)
                write = p.contract(f"{s}.write", delta, kt, ["U"])  # k_t ⊗ delta: [B,Hk,R,Dv,Dk]
                S = write if S is None else E("add", f"{s}.S", S, write)
                placed = p.contract(f"{s}.o", p.contract(f"{s}.read", S, qt, ["Dk"]), eu, ["U"])  # [B,Hk,R,Dv,T]
                out = placed if out is None else E("add", f"{s}.acc", out, placed)
            o = p.permute(f"Ol_{i}", out, ("B", "T", "Hk", "R", "Dv"))
        with self.at(f"{at}.norm", "RMSNormGated"):
            zg = self.silu(f"Zg_{i}", p.reshape(f"Z_{i}.heads", z, ("B", "T", "Hk", "R", "Dv")))
            normed = self.rms(f"On_{i}.norm", o, self.param(f"gn_{i}", ("Dv",), k("norm.weight")), "Dv", centred=False)
            gated = p.reshape(f"On_{i}.merged", E("mul", f"On_{i}", normed, zg), ("B", "T", "Dvv"))
        return self.linear(f"{at}.out_proj", f"O_{i}", gated, f"Wout_{i}", ("D", "Dvv"), k("out_proj.weight"))

    def full_attention(self, i, x, at, key):
        p, E = self.p, self.p.elementwise
        k = lambda part: f"{key}.{part}"  # noqa: E731
        dh = p.dims["Dh"]
        qg = self.linear(f"{at}.q_proj", f"Qg_{i}", x, f"Wq_{i}", ("Dqg", "D"), k("q_proj.weight"))
        kx = self.linear(f"{at}.k_proj", f"K_{i}.proj", x, f"Wk_{i}", ("Dkv", "D"), k("k_proj.weight"))
        vx = self.linear(f"{at}.v_proj", f"V_{i}.proj", x, f"Wv_{i}", ("Dkv", "D"), k("v_proj.weight"))
        with self.at(at, "Attention"):
            heads = p.reshape(f"Qg_{i}.heads", qg, ("B", "T", "Hkv", "G", "QG"))
            query = p.slice(f"Q_{i}.query", heads, "QG", 0, dh, "Dh")
            gate = p.slice(f"Gate_{i}", heads, "QG", dh, 2 * dh, "Dh")
        with self.at(f"{at}.q_norm", "RMSNorm"):
            qn = self.rms(f"Q_{i}.norm", query, self.param(f"gq_{i}", ("Dh",), k("q_norm.weight")), "Dh")
        with self.at(f"{at}.k_norm", "RMSNorm"):
            kn = self.rms(f"K_{i}.norm", p.reshape(f"K_{i}.heads", kx, ("B", "T", "Hkv", "Dh")),
                          self.param(f"gk_{i}", ("Dh",), k("k_norm.weight")), "Dh")
        with self.at(at, "Attention"):
            qr = self.rope(f"Q_{i}", p.rename(f"Q_{i}.q", p.permute(f"Q_{i}.perm", qn, ("B", "Hkv", "G", "T", "Dh")), "T", "q"), "q")
            kr = self.rope(f"K_{i}", p.rename(f"K_{i}.k", p.permute(f"K_{i}.perm", kn, ("B", "Hkv", "T", "Dh")), "T", "k"), "k")
            vh = p.rename(f"V_{i}", p.permute(f"V_{i}.perm", p.reshape(f"V_{i}.heads", vx, ("B", "T", "Hkv", "Dh")), ("B", "Hkv", "T", "Dh")), "T", "k")
            s = E("div", f"S_{i}", p.contract(f"S_{i}.qk", qr, kr, ["Dh"]), self.c_("c_sqrt_dh", math.sqrt(dh)))
            causal = self.c_("M01", [[1.0 if kk <= qq else 0.0 for kk in range(self.T)] for qq in range(self.T)], ("q", "k"))
            masked = E("add", f"S_{i}.masked", s, E("log", f"Mask_{i}", causal))
            with p.composite("softmax", masked, axis="k"):
                top = p.reduce("max", f"A_{i}.max", masked, "k")
                ex = E("exp", f"A_{i}.exp", E("sub", f"A_{i}.shifted", masked, top))
                att = E("div", f"A_{i}", ex, p.reduce("sum", f"A_{i}.sum", ex, "k"))
            ctx = p.permute(f"C_{i}.perm", p.rename(f"C_{i}.T", p.contract(f"C_{i}", att, vh, ["k"]), "q", "T"), ("B", "T", "Hkv", "G", "Dh"))
            gated = p.reshape(f"C_{i}.merged", E("mul", f"C_{i}.gated", ctx, E("sigmoid", f"Gate_{i}.sig", gate)), ("B", "T", "Dq"))
        return self.linear(f"{at}.o_proj", f"O_{i}", gated, f"Wo_{i}", ("D", "Dq"), k("o_proj.weight"))

    def mlp(self, i, x, at, key):
        g = self.linear(f"{at}.gate_proj", f"G_{i}", x, f"Wg_{i}", ("F", "D"), f"{key}.gate_proj.weight")
        with self.at(f"{at}.act_fn", "SiLU"):
            ga = self.silu(f"Ga_{i}", g)
        u = self.linear(f"{at}.up_proj", f"U_{i}", x, f"Wu_{i}", ("F", "D"), f"{key}.up_proj.weight")
        with self.at(at, "MLP"):
            h = self.p.elementwise("mul", f"Hm_{i}", ga, u)
        return self.linear(f"{at}.down_proj", f"M_{i}", h, f"Wd_{i}", ("D", "F"), f"{key}.down_proj.weight")

    def run(self) -> Program:
        p, E, c, pre = self.p, self.p.elementwise, self.c, self.prefix
        top = "model"
        ids = p.declare("ids", "input", ("B", "T"), "int64")
        with self.at(f"{top}.embed_tokens", "Embedding"):
            x = p.gather("X_0", self.param("E", ("V", "D"), f"{pre}embed_tokens.weight"), ids, "V")
        for i, kind in enumerate(c["layer_types"]):
            at, key = f"{top}.layers.{i}", f"{pre}layers.{i}"
            with self.at(f"{at}.input_layernorm", "RMSNorm"):
                n1 = self.rms(f"N1_{i}", x, self.param(f"g1_{i}", ("D",), f"{key}.input_layernorm.weight"), "D")
            if kind == "linear_attention":
                o = self.linear_attention(i, n1, f"{at}.linear_attn", f"{key}.linear_attn")
            elif kind == "full_attention":
                o = self.full_attention(i, n1, f"{at}.self_attn", f"{key}.self_attn")
            else:
                raise NotImplementedError(f"layer {i}: layer type {kind!r} has no lowering")
            with self.at(at, "DecoderLayer"):
                xm = E("add", f"Xm_{i}", x, o)
            with self.at(f"{at}.post_attention_layernorm", "RMSNorm"):
                n2 = self.rms(f"N2_{i}", xm, self.param(f"g2_{i}", ("D",), f"{key}.post_attention_layernorm.weight"), "D")
            m = self.mlp(i, n2, f"{at}.mlp", f"{key}.mlp")
            with self.at(at, "DecoderLayer"):
                x = E("add", f"X_{i + 1}", xm, m)
        with self.at(f"{top}.norm", "RMSNorm"):
            xf = self.rms("Xf", x, self.param("gf", ("D",), f"{pre}norm.weight"), "D")
        assert not c.get("tie_word_embeddings"), "tied lm_head: not in this checkpoint"
        with self.at("lm_head", "Linear"):
            p.contract("L", xf, self.param("Wlm", ("V", "D"), "lm_head.weight"), ["D"])
        return p


def lower_qwen3_5(cfg: dict, batch: int = 1, seq: int = 8, prefix: str = "model.language_model.") -> Program:
    """config.json (whole, or its text_config) -> Program. `prefix` is the checkpoint key prefix of the language model."""
    return _Lowering(cfg, batch, seq, prefix).run()
