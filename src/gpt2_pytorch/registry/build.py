"""Path A build (master §6): descriptor × resolved hyperparameters × naming scheme → Program → viz_graph().

python -m gpt2_pytorch.registry.build --descriptor ID --config config.json [--name NAME] [--scheme hf] --out graph.json

Returns the graph and the parameters it expects in the checkpoint ([{name, role, shape, checkpoint_key, view}]),
which bind.py checks against a manifest. Graphs are written the way JSON.stringify writes them (the goldens' format).
"""
import argparse
import json
import math
import re
from decimal import Decimal
from pathlib import Path

from ..tensorprog.viz import viz_graph
from .blocks import ATTENTION, EXAMPLE, MLP, ROLES, Builder, rms, softcap
from .descriptor import DescriptorError, evaluate, hyper_from_config, layer_blocks, load

MAX_LAYERS, MAX_DIM = 1024, 1 << 31


def _values(hyper: dict) -> dict:
    return {k: (v[0] if isinstance(v, tuple) else v) for k, v in hyper.items()}


def _finish(prog, d: dict, name: str, example: dict, extra=None, roles=True) -> tuple[dict, list]:
    for t in prog.tensors.values():
        t.pytorch_name = None  # program.py's mapping names GPT-2 debug tensors only
    doc = prog.to_dict(name, example) | {"verification": []} | ({"roles": ROLES} if roles else {})
    g = viz_graph(json.loads(json.dumps(doc)), f"gpt2_pytorch.registry {d['id']}@{d['version']}")
    g.update(extra or {})
    return g, expected(g)


def expected(g: dict) -> list[dict]:
    return [{k: t[k] for k in ("name", "role", "shape", "checkpoint_key", "view")}
            for t in g["tensors"] if t["role"] in ("parameter", "buffer")]


def decoder(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """Pre-norm decoder (arch.mjs llamaGraph 199-219): embed; per layer norm, attention, residual, norm, mlp, residual;
    final norm; head (tied to the embedding when `tie`)."""
    r = d["recipe"]
    cfg = cfg.get(r["config"]) or cfg if r.get("config") else cfg  # assemblies: selectors read the text config
    h = {**r.get("flags", {}), **h}
    h["plus_one"] = bool(h.get("norm_plus_one")) and scheme != "gguf"  # GGUF stores Gemma norms with the +1 folded in
    one = h["plus_one"]
    mods = {"ffn_norm": "post_attention_layernorm", **r.get("modules", {})}
    dims = {a: int(evaluate(str(e), {**EXAMPLE, **h})) for a, e in r["dims"].items()}
    if not 0 < int(h["L"]) <= MAX_LAYERS or any(not 0 < n < MAX_DIM for n in dims.values()):  # hostile configs (C5)
        raise DescriptorError(f"{d['id']}: implausible sizes L={h['L']} {dims}")
    fmt = {**h, **dims}
    fmt |= {k: evaluate(str(e), fmt) for k, e in r.get("format_values", {}).items()}
    b = Builder(dims, r["root"], d["schemes"][scheme], fmt)
    root, L = r["root"], h["L"]
    at = lambda i: f"{root}.layers.{i}"  # noqa: E731
    slots = {s: layer_blocks(sel, cfg, L, names) for s, sel in r["layers"].items()}
    b.p.declare("ids", "input", ("B", "T"), "int64")
    b.constant("c_eps", h["eps"])
    b.constant("c_sqrt_dh", math.sqrt(evaluate(r.get("score_dim", "Dh"), {**h, **dims})))
    b.constant("M01", [[1 if k <= q else 0 for k in range(EXAMPLE["T"])] for q in range(EXAMPLE["T"])], ["q", "k"])
    for c, v in r.get("constants", {}).items():
        b.constant(c, evaluate(str(v), {**h, **dims}))
    E = b.param("E", ["V", "D"], "embed")
    with b.within(f"{root}.embed_tokens", "Embedding"):
        if h.get("embed_scale"):  # Gemma: the embedding times sqrt(D)
            b.gather("X_0.emb", E, "ids", "V")
            b.E("mul", "X_0", "X_0.emb", "c_emb")
        else:
            b.gather("X_0", E, "ids", "V")
    for i in range(L):
        with b.within(f"{at(i)}.input_layernorm", "RMSNorm"):
            rms(b, f"N1_{i}", f"X_{i}", b.param(f"g1_{i}", ["D"], "attn_norm", i=i), "D", one)
        o = ATTENTION[slots["attention"][i]](b, h, i, f"{at(i)}.self_attn")
        if "post_attn_norm" in d["schemes"][scheme]:  # Gemma 2/3: a norm on the attention output
            with b.within(f"{at(i)}.post_attention_layernorm", "RMSNorm"):
                o = rms(b, f"Oa_{i}", o, b.param(f"gpa_{i}", ["D"], "post_attn_norm", i=i), "D", one)
        with b.within(at(i), "DecoderLayer"):
            b.E("add", f"Xm_{i}", f"X_{i}", o)
        with b.within(f"{at(i)}.{mods['ffn_norm']}", "RMSNorm"):
            rms(b, f"N2_{i}", f"Xm_{i}", b.param(f"g2_{i}", ["D"], "ffn_norm", i=i), "D", one)
        m = MLP[slots["mlp"][i]](b, h, i, at(i))
        if "post_ffn_norm" in d["schemes"][scheme]:
            with b.within(f"{at(i)}.post_feedforward_layernorm", "RMSNorm"):
                m = rms(b, f"Mf_{i}", m, b.param(f"gpf_{i}", ["D"], "post_ffn_norm", i=i), "D", one)
        with b.within(at(i), "DecoderLayer"):
            b.E("add", f"X_{i + 1}", f"Xm_{i}", m)
    with b.within(f"{root}.norm", "RMSNorm"):
        rms(b, "Xf", f"X_{L}", b.param("gf", ["D"], r.get("final_norm_role", "norm")), "D", one)  # LFM2: embd_norm
    tie = bool(h.get("tie")) and d.get("ties", {}).get("head") == "embed"
    head = E if tie else b.param("Wlm", ["V", "D"], "head")
    if tie:
        b.p.aliases[b.key("head")[0]] = E
    with b.within("lm_head", "Linear"):
        if h.get("final_cap"):  # Gemma 2: logits soft-capped
            softcap(b, "L", b.C("L.raw", "Xf", head, "D"), "c_final_cap")
        else:
            b.C("L", "Xf", head, "D")
    extra = {"lanes": r["lanes"]} if r.get("lanes") else {}
    g, exp = _finish(b.p, d, name, EXAMPLE, extra)
    if b.routed:
        annotate_routed(g, b.routed)
    annotate_scan(g)
    return g, exp


def annotate_routed(g: dict, routed) -> None:
    """D8: expert ops and their multiplications carry execution {kind: routed, of: E, active: K, router}; totals gain
    active_flops (each routed contraction counted for K of its E experts)."""
    E = g["dims"]["E"]
    marks = {}
    for start, end, router, k in routed:
        for j in range(start, end):
            marks[j] = {"kind": "routed", "of": "E", "active": k, "router": router}
            g["ops"][j]["execution"] = marks[j]
    removed = 0
    for m in g["multiplications"]:
        if m["op"] in marks:
            m["execution"] = marks[m["op"]]
            removed += m["flops"] * (E - marks[m["op"]]["active"]) // E
    g["totals"]["active_flops"] = g["totals"]["contraction_flops"] - removed


def gpt2_lowering(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """GPT-2 through the verified tensorprog/lower.py, on a meta-device model built from the config (no weights)."""
    import torch

    from ..config import GPT2Config
    from ..model import DistilGPT2LMHeadModel
    from ..tensorprog import lower

    c = GPT2Config(vocab_size=h["V"], max_position_embeddings=h["P"], hidden_size=h["D"], num_layers=h["L"],
                   num_heads=h["H"], intermediate_size=h["F"], layer_norm_epsilon=h["eps"])
    with torch.device("meta"):
        model = DistilGPT2LMHeadModel(c)
    model.eval()
    prog, _ = lower(model, 2, 8)
    prov = d["provenance"]
    verified = all(h[k] == v for k, v in prov["verified_dims"].items())
    doc = prog.to_dict(name, {"B": 2, "T": 8}) | {"verification": prov["verification"] if verified else []}
    g = viz_graph(json.loads(json.dumps(doc)), prov["source"])
    return g, expected(g)


def spell_qwen3_5(kv: dict, shapes: dict) -> dict:
    """The text config.json the qwen3_5 lowering reads, spelled from the metadata llama.cpp writes for arch qwen35."""
    g = lambda k: kv[f"qwen35.{k}"]  # noqa: E731
    # block_count counts the MTP block(s) after the stack (nextn_predict_layers), stored or stripped
    n, every, hv = g("block_count") - kv.get("qwen35.nextn_predict_layers", 0), g("full_attention_interval"), g("ssm.time_step_rank")
    return {"model_type": "qwen3_5_text", "hidden_size": g("embedding_length"), "num_hidden_layers": n,
            "num_attention_heads": g("attention.head_count"), "num_key_value_heads": g("attention.head_count_kv"),
            "head_dim": g("attention.key_length"), "intermediate_size": g("feed_forward_length"),
            "vocab_size": shapes["token_embd.weight"][0], "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
            "linear_num_key_heads": g("ssm.group_count"), "linear_key_head_dim": g("ssm.state_size"),
            "linear_num_value_heads": hv, "linear_value_head_dim": g("ssm.inner_size") // hv,
            "linear_conv_kernel_dim": g("ssm.conv_kernel"),
            "layer_types": ["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)],
            "rope_theta": g("rope.freq_base"), "partial_rotary_factor": g("rope.dimension_count") / g("attention.key_length"),
            "tie_word_embeddings": "output.weight" not in shapes}


def qwen3_5(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """Qwen3.5 / Qwen3.6 hybrid text model through the existing tensorprog/qwen3_5.py lowering (called, not rewritten).
    FP8 `*_scale_inv` views need the header (sidecars, 06e); from a config alone they stay empty."""
    from ..tensorprog.qwen3_5 import ROLES as Q_ROLES, lower_qwen3_5

    prog = lower_qwen3_5(cfg, batch=1, seq=EXAMPLE["T"], prefix=d["schemes"][scheme]["prefix"])
    doc = prog.to_dict(name, {"B": 1, "T": EXAMPLE["T"]}) | {"verification": [], "roles": Q_ROLES}
    g = viz_graph(json.loads(json.dumps(doc)), f"gpt2_pytorch.registry {d['id']}@{d['version']}")
    annotate_scan(g)
    return g, expected(g)


SCAN_STEP = re.compile(r"^(?P<sym>[A-Za-z]+_\d+)\.t(?P<t>\d+)\.")


def annotate_scan(g: dict) -> None:
    """D8: the ops of an unrolled recurrence step (`<sym>_<layer>.t<step>.*`: the gated delta rule's Dn_i, Mamba's
    Ss_i) carry execution {kind: scan, axis: T, step, body: <the statement group of the recurrence>}."""
    marks = {}
    for o in g["ops"]:
        if m := SCAN_STEP.match(o["out"]):
            o["execution"] = marks[o["index"]] = {"kind": "scan", "axis": "T", "step": int(m["t"]), "body": f"{o['module']}:{m['sym']}"}
    for mm in g["multiplications"]:
        if mm["op"] in marks:
            mm["execution"] = marks[mm["op"]]


def encoder(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """BERT-family encoder: word + position (+ token type) embeddings and LayerNorm, post-norm layers; the last hidden
    state is L. Task heads (pooler, MLM, classifiers) are components declared without a graph."""
    from .blocks import encoder_layer, layernorm
    r = d["recipe"]
    dims = {a: int(evaluate(str(e), {**EXAMPLE, **h})) for a, e in r["dims"].items()}
    b = Builder(dims, r["root"], d["schemes"][scheme], {**h, **dims})
    root, L = r["root"], int(h["L"])
    b.p.declare("ids", "input", ("B", "T"), "int64")
    b.p.declare("pos", "input", ("T",), "int64")
    if "type_embed" in b.scheme:
        b.p.declare("type_ids", "input", ("B", "T"), "int64")
    b.constant("c_eps", h["eps"])
    b.constant("c_sqrt_dh", math.sqrt(dims["Dh"]))
    with b.within(f"{root}.embeddings", "Embeddings"):
        x = b.E("add", "X_0.wp", b.gather("X_0.w", b.param("E", ["V", "D"], "embed"), "ids", "V"),
                b.gather("X_0.p", b.param("Ep", ["P", "D"], "pos_embed"), "pos", "P"))
        if "type_embed" in b.scheme:
            x = b.E("add", "X_0.sum", x, b.gather("X_0.t", b.param("Et", ["Tv", "D"], "type_embed"), "type_ids", "Tv"))
    with b.within(f"{root}.embeddings.LayerNorm", "LayerNorm"):
        layernorm(b, "X_0", x, b.param("ge", ["D"], "emb_ln", what="weight"), b.param("be", ["D"], "emb_ln", what="bias"))
    for i in range(L):
        encoder_layer(b, h, i, f"{root}.encoder.layer.{i}", f"X_{i}", "L" if i == L - 1 else f"X_{i + 1}")
    return _finish(b.p, d, name, EXAMPLE)


def t5_buckets(T: int, bidirectional: bool, n: int, max_distance: int) -> list:
    """HF T5Attention._relative_position_bucket for memory - query positions on a T x T grid"""
    out = []
    for q in range(T):
        row = []
        for k in range(T):
            rel, nb, bucket = k - q, n, 0
            if bidirectional:
                nb //= 2
                bucket += nb if rel > 0 else 0
                rel = abs(rel)
            else:
                rel = -min(rel, 0)
            exact = nb // 2
            if rel < exact:
                bucket += rel
            else:
                bucket += min(nb - 1, exact + int(math.log(rel / exact) / math.log(max_distance / exact) * (nb - exact)))
            row.append(bucket)
        out.append(row)
    return out


def seq2seq(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """Encoder-decoder (T5 / mT5: pre-norm RMS, relative position bias, ReLU or gated-GELU FFN; BART: post-norm
    LayerNorm, learned positions, GELU). Encoder ids and decoder ids share T; L is the decoder's logits."""
    from .blocks import gelu_erf, gelu_tanh, layernorm, silu, softmax
    r = d["recipe"]
    t5 = r["flavour"] == "t5"
    dims = {a: int(evaluate(str(e), {**EXAMPLE, **h})) for a, e in r["dims"].items()}
    b = Builder(dims, r["root"], d["schemes"][scheme], {**h, **dims})
    root = r["root"]
    b.p.declare("ids", "input", ("B", "T"), "int64")
    b.p.declare("dec_ids", "input", ("B", "T"), "int64")
    if not t5:
        b.p.declare("pos", "input", ("T",), "int64")
    b.constant("c_eps", h["eps"])
    b.constant("c_sqrt_dh", math.sqrt(dims["Dh"]))
    b.constant("M01", [[1 if k <= q else 0 for k in range(EXAMPLE["T"])] for q in range(EXAMPLE["T"])], ["q", "k"])
    E = b.param("E", ["V", "D"], "embed")
    act = {"gelu": gelu_erf, "gelu_new": gelu_tanh, "swish": silu, "silu": silu}

    def lin(out, x, w, axes, role, i, over, bias):
        y = b.C(out if not bias else f"{out}.mm", x, b.param(w, axes, role, i=i, what="weight"), over)
        return b.E("add", out, y, b.param("b" + w[1:], axes[:1], role, i=i, what="bias")) if bias else y

    def mha(tag, i, xq, xkv, at, roles, mask=None, rel=None):
        bias = not t5
        with b.within(at, "Attention"):
            q = lin(f"{tag}Q_{i}", xq, f"W{tag}q_{i}", ["Dq", "D"], roles[0], i, "D", bias)
            k = lin(f"{tag}K_{i}", xkv, f"W{tag}k_{i}", ["Dq", "D"], roles[1], i, "D", bias)
            v = lin(f"{tag}V_{i}", xkv, f"W{tag}v_{i}", ["Dq", "D"], roles[2], i, "D", bias)
            qh = b.rename(f"{tag}Q_{i}.q", b.permute(f"{tag}Q_{i}.perm", b.reshape(f"{tag}Q_{i}.heads", q, ["B", "T", "H", "Dh"]), [0, 2, 1, 3]), "T", "q")
            kh = b.rename(f"{tag}K_{i}.k", b.permute(f"{tag}K_{i}.perm", b.reshape(f"{tag}K_{i}.heads", k, ["B", "T", "H", "Dh"]), [0, 2, 1, 3]), "T", "k")
            vh = b.rename(f"{tag}V_{i}.k", b.permute(f"{tag}V_{i}.perm", b.reshape(f"{tag}V_{i}.heads", v, ["B", "T", "H", "Dh"]), [0, 2, 1, 3]), "T", "k")
            s = b.C(f"{tag}S_{i}.qk", qh, kh, "Dh")
            if not t5:  # T5 folds the 1/sqrt(d) scale into its initialisation
                s = b.E("div", f"{tag}S_{i}", s, "c_sqrt_dh")
            if rel:
                s = b.E("add", f"{tag}S_{i}.rel", s, rel)
            if mask:
                s = b.E("add", f"{tag}S_{i}.masked", s, b.E("log", f"{tag}Mask_{i}", "M01"))
            softmax(b, f"{tag}A_{i}", s)
            c = b.C(f"{tag}C_{i}", f"{tag}A_{i}", vh, "k")
            merged = b.reshape(f"{tag}C_{i}.merged", b.permute(f"{tag}C_{i}.perm", b.rename(f"{tag}C_{i}.rename", c, "q", "T"), [0, 2, 1, 3]), ["B", "T", "Dq"])
            return lin(f"{tag}O_{i}", merged, f"W{tag}o_{i}", ["D", "Dq"], roles[3], i, "Dq", bias)

    def norm(out, x, role, i, at):
        with b.within(at, "LayerNorm" if not t5 else "T5LayerNorm"):
            if t5:
                return rms(b, out, x, b.param(f"g{out}", ["D"], role, i=i), "D")
            return layernorm(b, out, x, b.param(f"g{out}", ["D"], role, i=i, what="weight"), b.param(f"b{out}", ["D"], role, i=i, what="bias"))

    def ffn(tag, i, x, at, role_prefix):
        with b.within(at, "FeedForward"):
            # gated-GELU (v1.1 / mT5) per the config, or, with no config (GGUF), per the stored gate tensor
            gated = cfg.get("feed_forward_proj", "relu").startswith("gated") if cfg else b.key(f"{role_prefix}wi0", i=i)[0] in names
            if t5 and gated:
                g = gelu_tanh(b, f"{tag}G_{i}", b.C(f"{tag}G_{i}.mm", x, b.param(f"W{tag}g_{i}", ["F", "D"], f"{role_prefix}wi0", i=i), "D"))
                u = b.C(f"{tag}U_{i}", x, b.param(f"W{tag}u_{i}", ["F", "D"], f"{role_prefix}wi1", i=i), "D")
                hdn = b.E("mul", f"{tag}H_{i}", g, u)
            elif t5:
                hdn = b.E("relu", f"{tag}H_{i}", b.C(f"{tag}H_{i}.mm", x, b.param(f"W{tag}i_{i}", ["F", "D"], f"{role_prefix}wi", i=i), "D"))
            else:
                f = act[cfg.get("activation_function", "gelu")]
                hdn = f(b, f"{tag}H_{i}", lin(f"{tag}H_{i}.lin", x, f"W{tag}i_{i}", ["F", "D"], f"{role_prefix}fc1", i, "D", True))
            return lin(f"{tag}M_{i}", hdn, f"W{tag}m_{i}", ["D", "F"], f"{role_prefix}fc2" if not t5 else f"{role_prefix}wo", i, "F", not t5)

    def rel_bias(tag, bidirectional, role):
        b.p.declare(f"{tag}bucket", "constant", ["q", "k"], "int64", value=t5_buckets(EXAMPLE["T"], bidirectional, dims["Nb"], h["Md"]))
        with b.within(f"{root}.{'encoder' if bidirectional else 'decoder'}.block.0.layer.0.SelfAttention", "RelativeBias"):
            g = b.gather(f"{tag}Pb", b.param(f"W{tag}rb", ["Nb", "H"], role), f"{tag}bucket", "Nb")  # [q, k, H]
            return b.permute(f"{tag}Pb.perm", g, [2, 0, 1])

    def embed(tag, ids, stack):
        with b.within(f"{root}.{stack}.embed_tokens", "Embedding"):
            x = b.gather(f"{tag}X_0.w", E, ids, "V")
            if h.get("scale"):
                x = b.E("mul", f"{tag}X_0.ws", x, b.constant("c_emb", math.sqrt(dims["D"])))
            if t5:
                return x
            x = b.E("add", f"{tag}X_0.wp", x, b.gather(f"{tag}X_0.p", b.param(f"{tag}Ep", ["Pp", "D"], f"{stack}_pos"), "pos", "Pp"))
        return norm(f"{tag}X_0", x, f"{stack}_emb_ln", 0, f"{root}.{stack}.layernorm_embedding") if f"{stack}_emb_ln" in b.scheme else x

    # encoder
    x = embed("e", "ids", "encoder")
    rel = rel_bias("e", True, "enc_rel_bias") if t5 else None
    for i in range(int(h["L"])):
        at = f"{root}.encoder.block.{i}" if t5 else f"{root}.encoder.layers.{i}"
        if t5:
            n1 = norm(f"eN1_{i}", x, "enc_ln1", i, f"{at}.layer.0.layer_norm")
            x = b.E("add", f"eXa_{i}", x, mha("e", i, n1, n1, f"{at}.layer.0.SelfAttention", ("enc_q", "enc_k", "enc_v", "enc_o"), rel=rel))
            n2 = norm(f"eN2_{i}", x, "enc_ln2", i, f"{at}.layer.1.layer_norm")
            x = b.E("add", f"eX_{i + 1}", x, ffn("e", i, n2, f"{at}.layer.1.DenseReluDense", "enc_"))
        else:
            x = norm(f"eXa_{i}", b.E("add", f"eXa_{i}.res", x, mha("e", i, x, x, f"{at}.self_attn", ("enc_q", "enc_k", "enc_v", "enc_o"))),
                     "enc_ln1", i, f"{at}.self_attn_layer_norm")
            x = norm(f"eX_{i + 1}", b.E("add", f"eX_{i + 1}.res", x, ffn("e", i, x, at, "enc_")), "enc_ln2", i, f"{at}.final_layer_norm")
    xe = norm("eXf", x, "enc_final", 0, f"{root}.encoder.final_layer_norm") if t5 else x
    # decoder
    x = embed("d", "dec_ids", "decoder")
    rel = rel_bias("d", False, "dec_rel_bias") if t5 else None
    for i in range(int(h["Ld"])):
        at = f"{root}.decoder.block.{i}" if t5 else f"{root}.decoder.layers.{i}"
        self_roles, cross_roles = ("dec_q", "dec_k", "dec_v", "dec_o"), ("x_q", "x_k", "x_v", "x_o")
        if t5:
            n1 = norm(f"dN1_{i}", x, "dec_ln1", i, f"{at}.layer.0.layer_norm")
            x = b.E("add", f"dXa_{i}", x, mha("d", i, n1, n1, f"{at}.layer.0.SelfAttention", self_roles, mask=True, rel=rel))
            n2 = norm(f"dN2_{i}", x, "dec_ln2", i, f"{at}.layer.1.layer_norm")
            x = b.E("add", f"dXc_{i}", x, mha("x", i, n2, xe, f"{at}.layer.1.EncDecAttention", cross_roles))
            n3 = norm(f"dN3_{i}", x, "dec_ln3", i, f"{at}.layer.2.layer_norm")
            x = b.E("add", f"dX_{i + 1}", x, ffn("d", i, n3, f"{at}.layer.2.DenseReluDense", "dec_"))
        else:
            x = norm(f"dXa_{i}", b.E("add", f"dXa_{i}.res", x, mha("d", i, x, x, f"{at}.self_attn", self_roles, mask=True)),
                     "dec_ln1", i, f"{at}.self_attn_layer_norm")
            x = norm(f"dXc_{i}", b.E("add", f"dXc_{i}.res", x, mha("x", i, x, xe, f"{at}.encoder_attn", cross_roles)),
                     "dec_ln2", i, f"{at}.encoder_attn_layer_norm")
            x = norm(f"dX_{i + 1}", b.E("add", f"dX_{i + 1}.res", x, ffn("d", i, x, at, "dec_")), "dec_ln3", i, f"{at}.final_layer_norm")
    xd = norm("dXf", x, "dec_final", 0, f"{root}.decoder.final_layer_norm") if t5 else x
    tie = bool(h.get("tie"))
    head = E if tie else b.param("Wlm", ["V", "D"], "head")
    for role in ("enc_embed", "dec_embed") + (("head",) if tie else ()):  # stored copies of the shared table are tied
        if role in b.scheme:
            b.p.aliases[b.key(role)[0]] = E
    with b.within("lm_head", "Linear"):
        if t5 and tie:  # T5 v1.0 rescales before the tied head
            xd = b.E("mul", "dXf.scaled", xd, b.constant("c_tie_scale", dims["D"] ** -0.5))
        # BART's final_logits_bias is an all-zero buffer HF lets checkpoints omit: drawn only when stored
        if "logits_bias" in b.scheme and (not names or b.key("logits_bias")[0] in names):
            raw = b.C("L.raw", xd, head, "D")
            b.E("add", "L", raw, b.reshape("Lb", b.param("Wlb", ["U", "V"], "logits_bias"), ["V"]))
        else:
            b.C("L", xd, head, "D")
    return _finish(b.p, d, name, EXAMPLE)


TEMPLATES = {"decoder": decoder, "encoder": encoder, "seq2seq": seq2seq, "gpt2_lowering": gpt2_lowering, "qwen3_5": qwen3_5}


def build(d: dict, hyper: dict, scheme: str = "hf", name: str = "model", cfg: dict | None = None, names=()):
    """(graph, expected params) for descriptor `d` with resolved `hyper` ({name: value or (value, source)})."""
    h = _values(hyper)
    if "L" in h and not 0 < int(h["L"]) <= MAX_LAYERS:  # hostile configs fail before any template runs (C5)
        raise DescriptorError(f"{d['id']}: implausible layer count {h['L']}")
    return TEMPLATES[d["recipe"]["template"]](d, h, scheme, name, cfg or {}, names)


# ---- JSON the way JSON.stringify writes it -------------------------------------------------------------------------
def _js_number(x: float) -> str:
    if x != x or x in (math.inf, -math.inf):
        return "null"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    sign, digits, exp = Decimal(repr(x)).as_tuple()
    ds = "".join(map(str, digits)).rstrip("0") or "0"
    k, n, s = len(ds), exp + len(digits), "-" if sign else ""  # value = 0.ds × 10^n
    if k <= n <= 21:
        return s + ds + "0" * (n - k)
    if 0 < n <= 21:
        return s + ds[:n] + "." + ds[n:]
    if -6 < n <= 0:
        return s + "0." + "0" * -n + ds
    return s + ds[0] + ("." + ds[1:] if k > 1 else "") + "e" + ("+" if n > 1 else "-") + str(abs(n - 1))


def dump_js(o) -> str:
    if o is None or o is True or o is False:
        return {None: "null", True: "true", False: "false"}[o]
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        return _js_number(o)
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(dump_js(v) for v in o) + "]"
    return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + dump_js(v) for k, v in o.items()) + "}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--name", help="model name (default: the config file name before .config)")
    ap.add_argument("--scheme", default="hf")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    d, cfg = load(a.descriptor), json.loads(a.config.read_text())
    g, exp = build(d, hyper_from_config(d, cfg), a.scheme, a.name or a.config.name.split(".config")[0], cfg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(dump_js(g))
    print(f"{a.out}: {len(g['ops'])} ops, {len(exp)} parameters/buffers")


if __name__ == "__main__":
    main()

