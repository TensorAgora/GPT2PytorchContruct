"""Lower the pure-PyTorch model, module by module, into tensor operations (see program.py).

One rule per nn.Module type; a module type without a rule is an error, never silently skipped. Rules mirror the
arithmetic of the module's forward() exactly. Tensors that exist in TEL's emitted forward pass keep TEL's symbol
names (X_0, N1_l, Hq_l, Q_l, S_l, A_l, ..., Xf, L); finer intermediates are `<symbol>.<part>`.
"""
import math
from collections import Counter

from torch import nn

from ..checkpoint.mapping import checkpoint_key_for, map_key
from ..config import GPT2Config
from ..model import (
    CausalSelfAttention, DistilGPT2LMHeadModel, GPT2Embeddings, GPT2GELU, GPT2MLP, GPT2Model, PositionEmbedding,
    TokenEmbedding, TransformerBlock,
)
from .program import Program


class Lowering:
    def __init__(self, model: DistilGPT2LMHeadModel, batch: int, seq: int):
        c: GPT2Config = model.config
        self.config = c
        self.p = Program({
            "B": batch, "T": seq, "q": seq, "k": seq, "D": c.hidden_size, "Dc": c.hidden_size, "H": c.num_heads,
            "Dh": c.head_dim, "F": c.intermediate_size, "V": c.vocab_size, "P": c.max_position_embeddings,
            "QKV": 3 * c.hidden_size,
        })
        self.registered: dict[int, str] = {}  # id(tensor) -> symbol (detects tied weights)
        self.instances: Counter = Counter()

    def call(self, path: str, module: nn.Module, *args, **kwargs):
        rule = RULES.get(type(module))
        if rule is None:
            raise NotImplementedError(f"no tensor lowering for {type(module).__name__} at '{path}'")
        self.instances[type(module).__name__] += 1
        saved = self.p.module, self.p.module_type
        self.p.module, self.p.module_type = path, type(module).__name__
        try:
            return rule(self, module, path, *args, **kwargs)
        finally:
            self.p.module, self.p.module_type = saved

    def param(self, tensor, target: str, symbol: str, axes, role="parameter") -> str:
        """Declare a checkpoint tensor once. A tied weight resolves to the symbol it was first declared as."""
        if id(tensor) in self.registered:
            self.p.aliases[target] = self.registered[id(tensor)]
            return self.registered[id(tensor)]
        self.registered[id(tensor)] = symbol
        key = checkpoint_key_for(target, self.config.num_layers)
        return self.p.declare(symbol, role, axes, checkpoint_key=key)

    def checkpoint_is_in_out(self, target: str) -> bool:
        """HF Conv1D weights are stored [in, out]; embeddings and lm_head are [rows, features]."""
        key = checkpoint_key_for(target, self.config.num_layers)
        return key is not None and map_key(key).action == "transpose"


# ---- leaf modules --------------------------------------------------------------------------------------------
def lower_token_embedding(ctx, m: TokenEmbedding, path, ids, out):
    table = ctx.param(m.weight, f"{path}.weight", "E", ("V", "D"))
    return ctx.p.gather(out, table, ids, "V")


def lower_position_embedding(ctx, m: PositionEmbedding, path, pos, out):
    table = ctx.param(m.weight, f"{path}.weight", "Wp", ("P", "D"))
    return ctx.p.gather(out, table, pos, "P")


def lower_dropout(ctx, m: nn.Dropout, path, x):
    ctx.p.elided.append(path)  # identity in eval(): no tensor op
    return x


def lower_linear(ctx, m: nn.Linear, path, x, out, in_axis, out_axis, w, b=None):
    """y = x · W + b: W and b are tensors, and so is the product x · W."""
    in_out = ctx.checkpoint_is_in_out(f"{path}.weight")
    weight = ctx.param(m.weight, f"{path}.weight", w, (in_axis, out_axis) if in_out else (out_axis, in_axis))
    product = ctx.p.contract(f"{out}.mm" if b else out, x, weight, over=[in_axis])
    if not b:
        return product
    bias = ctx.param(m.bias, f"{path}.bias", b, (out_axis,))
    return ctx.p.elementwise("add", out, product, bias)


def lower_layernorm(ctx, m: nn.LayerNorm, path, x, out, g, b):
    """(x - mean) / sqrt(var + eps) * g + b, biased variance, every step a tensor."""
    p = ctx.p
    gamma = ctx.param(m.weight, f"{path}.weight", g, ("D",))
    beta = ctx.param(m.bias, f"{path}.bias", b, ("D",))
    eps = p.constant("c_eps", m.eps)
    mean = p.reduce("mean", f"{out}.mean", x, "D")
    centered = p.elementwise("sub", f"{out}.centered", x, mean)
    squared = p.elementwise("mul", f"{out}.squared", centered, centered)
    var = p.reduce("mean", f"{out}.var", squared, "D")
    var_eps = p.elementwise("add", f"{out}.var_eps", var, eps)
    rstd = p.elementwise("rsqrt", f"{out}.rstd", var_eps)
    normalized = p.elementwise("mul", f"{out}.normalized", centered, rstd)
    scaled = p.elementwise("mul", f"{out}.scaled", normalized, gamma)
    return p.elementwise("add", out, scaled, beta)


def lower_gelu(ctx, m: GPT2GELU, path, x, out):
    """0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), in the order GPT2GELU.forward evaluates it."""
    p = ctx.p
    a, b = p.constant("c_gelu_a", 0.044715), p.constant("c_gelu_b", math.sqrt(2.0 / math.pi))
    half, one = p.constant("c_half", 0.5), p.constant("c_one", 1.0)
    squared = p.elementwise("mul", f"{out}.squared", x, x)
    cubic = p.elementwise("mul", f"{out}.cubic", squared, x)
    cubic_a = p.elementwise("mul", f"{out}.cubic_a", cubic, a)
    inner_sum = p.elementwise("add", f"{out}.inner_sum", x, cubic_a)
    inner = p.elementwise("mul", f"{out}.inner", inner_sum, b)
    tanh = p.elementwise("tanh", f"{out}.tanh", inner)
    one_plus = p.elementwise("add", f"{out}.one_plus", tanh, one)
    half_x = p.elementwise("mul", f"{out}.half_x", x, half)
    return p.elementwise("mul", out, half_x, one_plus)


# ---- composite modules ---------------------------------------------------------------------------------------
def lower_mlp(ctx, m: GPT2MLP, path, x, layer):
    g = ctx.call(f"{path}.fc", m.fc, x, f"G_{layer}", "D", "F", f"W_fc_{layer}", f"b_fc_{layer}")
    ga = ctx.call(f"{path}.activation", m.activation, g, f"Ga_{layer}")
    mo = ctx.call(f"{path}.projection", m.projection, ga, f"M_{layer}", "F", "D", f"W_pr_{layer}", f"b_pr_{layer}")
    return ctx.call(f"{path}.dropout", m.dropout, mo)


def lower_attention(ctx, m: CausalSelfAttention, path, x, layer):
    p, c = ctx.p, ctx.config
    l = layer
    # [B,T,D] -> [B,T,QKV]
    hq = ctx.call(f"{path}.qkv_projection", m.qkv_projection, x, f"Hq_{l}", "D", "QKV", f"W_qkv_{l}", f"b_qkv_{l}")
    heads = {}
    for name, part, axis in (("Q", 0, "q"), ("K", 1, "k"), ("V", 2, "k")):  # chunk(3): q | k | v
        s = p.slice(f"{name}_{l}.slice", hq, "QKV", part * c.hidden_size, (part + 1) * c.hidden_size, "Dc")
        r = p.reshape(f"{name}_{l}.heads", s, ("B", "T", "H", "Dh"))
        t = p.permute(f"{name}_{l}.perm", r, ("B", "H", "T", "Dh"))
        heads[name] = p.rename(f"{name}_{l}", t, "T", axis)
    qk = p.contract(f"S_{l}.qk", heads["Q"], heads["K"], over=["Dh"])
    s = p.elementwise("div", f"S_{l}", qk, p.constant("c_sqrt_dh", math.sqrt(c.head_dim)))
    # causal mask: the checkpoint's own attn.bias buffer (1 = visible), cut to [T,T]; log(1)=0, log(0)=-inf
    mask = ctx.param(m.causal_mask, f"{path}.causal_mask", f"Mb_{l}", ("q", "k"), role="buffer")
    p.tensors[mask].view = "[0,0,:T,:T]"
    log_mask = p.elementwise("log", f"Mb_{l}.log", mask)
    masked = p.elementwise("add", f"S_{l}.masked", s, log_mask)
    # softmax over k
    top = p.reduce("max", f"A_{l}.max", masked, "k")
    shifted = p.elementwise("sub", f"A_{l}.shifted", masked, top)
    exp = p.elementwise("exp", f"A_{l}.exp", shifted)
    total = p.reduce("sum", f"A_{l}.sum", exp, "k")
    probs = p.elementwise("div", f"A_{l}", exp, total)
    ctx.call(f"{path}.attention_dropout", m.attention_dropout, probs)
    context = p.contract(f"C_{l}", probs, heads["V"], over=["k"])  # [B,H,q,Dh]
    # merge heads [B,H,q,Dh] -> [B,T,Dc]
    renamed = p.rename(f"C_{l}.rename", context, "q", "T")
    perm = p.permute(f"C_{l}.perm", renamed, ("B", "T", "H", "Dh"))
    merged = p.reshape(f"C_{l}.merged", perm, ("B", "T", "Dc"))
    o = ctx.call(f"{path}.output_projection", m.output_projection, merged, f"O_{l}", "Dc", "D", f"W_o_{l}", f"b_o_{l}")
    return ctx.call(f"{path}.residual_dropout", m.residual_dropout, o)


def lower_block(ctx, m: TransformerBlock, path, x, layer):
    p = ctx.p
    n1 = ctx.call(f"{path}.ln_1", m.ln_1, x, f"N1_{layer}", f"g1_{layer}", f"b1_{layer}")
    o = ctx.call(f"{path}.attention", m.attention, n1, layer)
    xm = p.elementwise("add", f"Xm_{layer}", x, o)  # residual
    n2 = ctx.call(f"{path}.ln_2", m.ln_2, xm, f"N2_{layer}", f"g2_{layer}", f"b2_{layer}")
    mo = ctx.call(f"{path}.mlp", m.mlp, n2, layer)
    return p.elementwise("add", f"X_{layer + 1}", xm, mo)  # residual


def lower_embeddings(ctx, m: GPT2Embeddings, path, ids, pos):
    tok = ctx.call(f"{path}.token_embedding", m.token_embedding, ids, "X_0.token")
    position = ctx.call(f"{path}.position_embedding", m.position_embedding, pos, "X_0.position")
    x = ctx.p.elementwise("add", "X_0", tok, position)
    return ctx.call(f"{path}.dropout", m.dropout, x)


def lower_backbone(ctx, m: GPT2Model, path, ids, pos):
    x = ctx.call(f"{path}.embeddings", m.embeddings, ids, pos)
    for i, block in enumerate(m.blocks):
        x = ctx.call(f"{path}.blocks.{i}", block, x, i)
    return ctx.call(f"{path}.final_layer_norm", m.final_layer_norm, x, "Xf", "gf", "bf")


def lower_model(ctx, m: DistilGPT2LMHeadModel, path, ids, pos):
    xf = ctx.call("transformer", m.transformer, ids, pos)
    # tied head: lm_head.weight is the embedding table E[V,D], so L = Xf · E over D
    return ctx.call("lm_head", m.lm_head, xf, "L", "D", "V", "E")


RULES = {
    DistilGPT2LMHeadModel: lower_model, GPT2Model: lower_backbone, GPT2Embeddings: lower_embeddings,
    TokenEmbedding: lower_token_embedding, PositionEmbedding: lower_position_embedding, TransformerBlock: lower_block,
    CausalSelfAttention: lower_attention, GPT2MLP: lower_mlp, GPT2GELU: lower_gelu, nn.LayerNorm: lower_layernorm,
    nn.Linear: lower_linear, nn.Dropout: lower_dropout,
}


def lower(model: DistilGPT2LMHeadModel, batch: int = 2, seq: int = 8) -> tuple[Program, Counter]:
    """Tensor program for `model`, with example shape (batch, seq). Returns (program, module-type instance counts)."""
    ctx = Lowering(model, batch, seq)
    ids = ctx.p.declare("ids", "input", ("B", "T"), "int64")
    pos = ctx.p.declare("pos", "input", ("T",), "int64")
    ctx.call("", model, ids, pos)
    return ctx.p, ctx.instances
