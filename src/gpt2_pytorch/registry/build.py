"""Path A build (master §6): descriptor × resolved hyperparameters × naming scheme → Program → viz_graph().

python -m gpt2_pytorch.registry.build --descriptor ID --config config.json [--name NAME] [--scheme hf] --out graph.json

Returns the graph and the parameters it expects in the checkpoint ([{name, role, shape, checkpoint_key, view}]),
which bind.py checks against a manifest. Graphs are written the way JSON.stringify writes them (the goldens' format).
"""
import argparse
import json
import math
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
        rms(b, "Xf", f"X_{L}", b.param("gf", ["D"], "norm"), "D", one)
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
    return _finish(b.p, d, name, EXAMPLE, extra)


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


def qwen3_5(d: dict, h: dict, scheme: str, name: str, cfg: dict, names=()):
    """Qwen3.5 / Qwen3.6 hybrid text model through the existing tensorprog/qwen3_5.py lowering (called, not rewritten).
    FP8 `*_scale_inv` views need the header (sidecars, 06e); from a config alone they stay empty."""
    from ..tensorprog.qwen3_5 import ROLES as Q_ROLES, lower_qwen3_5

    prog = lower_qwen3_5(cfg, batch=1, seq=EXAMPLE["T"], prefix=d["schemes"][scheme]["prefix"])
    doc = prog.to_dict(name, {"B": 1, "T": EXAMPLE["T"]}) | {"verification": [], "roles": Q_ROLES}
    g = viz_graph(json.loads(json.dumps(doc)), f"gpt2_pytorch.registry {d['id']}@{d['version']}")
    return g, expected(g)


TEMPLATES = {"decoder": decoder, "gpt2_lowering": gpt2_lowering, "qwen3_5": qwen3_5}


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

