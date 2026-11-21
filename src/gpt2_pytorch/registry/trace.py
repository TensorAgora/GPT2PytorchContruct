"""Path B, library trace (master §6, R-12, D4, D5): a transformers / timm / diffusers class built from config.json on
the `meta` device (no weights, never trust_remote_code), exported with torch.export, and normalized onto Program:
linear layers become Contractions (+ bias), elementwise and unary ATen ops map to Elementwise, view / reshape / flatten
to Reshape, embedding to Gather; every other ATen op is an `Opaque` op typed by its ports. Axes are named right-aligned
by size (r<pos>_<size>), so numpy broadcasting is TEL's subsequence rule. Parameters keep their state_dict names, which
are the checkpoint keys, and bind through bind.py like path A's.
"""
import operator
import threading

from ..tensorprog.program import Program
from ..tensorprog.viz import viz_graph

BINARY = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}
UNARY = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "exp": "exp", "log": "log", "rsqrt": "rsqrt", "erf": "erf"}
IDENTITY = {"dropout", "alias", "clone", "contiguous", "detach", "lift_fresh_copy", "_assert_tensor_metadata",
            "_to_copy", "to", "type_as"}  # dtype casts: the graph's dtypes are the checkpoint's


# torch.export's fake-tensor state is process-global: batch threads hand traces to a pool of spawned processes, which
# also run them in parallel and, recycled every 25 traces, give back the memory export keeps
WORKERS = 4
_POOL, _POOL_LOCK = None, threading.Lock()


class TraceError(Exception):
    pass


def transformers_has(model_type) -> bool:
    """whether the installed transformers ships a class for this model_type (path B never loads remote code)"""
    try:
        from transformers import CONFIG_MAPPING
    except ImportError:
        return False
    return model_type in CONFIG_MAPPING


def transformers_class(cfg: dict, kind: str):
    """(class name, build) for a kind, build() making the model: `arch` the class the checkpoint was saved from (task head included), `auto` a causal
    LM when there is one (seq2seq for encoder-decoders) else the bare model, `base` the bare model, or causal / seq2seq.
    Only classes transformers ships; trust_remote_code is never set."""
    import transformers as tr
    config = tr.AutoConfig.for_model(cfg.get("model_type"), **{k: v for k, v in cfg.items() if k not in ("model_type", "auto_map")})
    if kind == "arch":
        arch = (cfg.get("architectures") or [None])[0]
        klass = getattr(tr, arch, None) if isinstance(arch, str) else None
        if klass is None or not hasattr(klass, "config_class"):
            raise TraceError(f"transformers has no class {arch}")
        return klass.__name__, lambda: klass(config)
    if kind == "auto":
        kind = "seq2seq" if config.is_encoder_decoder else "causal" if type(config) in tr.MODEL_FOR_CAUSAL_LM_MAPPING else "base"
    auto = {"causal": tr.AutoModelForCausalLM, "base": tr.AutoModel, "seq2seq": tr.AutoModelForSeq2SeqLM}[kind]
    try:
        klass = auto._model_mapping[type(config)]
    except KeyError:
        raise TraceError(f"transformers has no {kind} class for {cfg.get('model_type')}") from None
    # the Auto factory, not klass(config): it picks the text sub-config of composite configs (Qwen3.5) itself
    return klass.__name__, lambda: auto.from_config(config)


def distinct_kinds(cfg: dict, kinds=("arch", "auto", "base")) -> list[str]:
    """kinds that build different classes (arch and auto are often both XForCausalLM): one trace each"""
    seen, out = set(), []
    for k in kinds:
        try:
            name = transformers_class(cfg, k)[0]
        except Exception:  # noqa: BLE001 - a kind with no class is tried, to report why
            name = f"?{k}"
        if name not in seen:
            seen.add(name)
            out.append(k)
    return out


def make_model(library: str, cls: str, cfg: dict, kind: str = "causal"):
    """The library class on the meta device. Remote code never runs: only classes the installed libraries ship."""
    import torch
    with torch.device("meta"):
        if library == "timm":
            import timm
            kw = {k: cfg[k] for k in ("num_classes", "in_chans") if k in cfg}
            m = timm.create_model(cls, pretrained=False, **kw)
        elif library == "diffusers":
            import diffusers
            klass = getattr(diffusers, cls, None)
            if klass is None:
                raise TraceError(f"diffusers has no class {cls}")
            m = klass.from_config({k: v for k, v in cfg.items() if not k.startswith("_")})
        elif library == "transformers":
            m = transformers_class(cfg, kind)[1]()
        else:
            raise TraceError(f"no library {library}")
    return m.eval()


def denoiser_inputs(cls: str, c: dict):
    """(args, kwargs) for a diffusers denoiser on meta: a small latent, 8 text tokens, one timestep; None for a class
    without a table entry (MiniMax-H3, Z-Image and the like need data-dependent index tensors or tensor lists)."""
    import torch
    m = {"device": "meta"}
    e = lambda *s: torch.empty(*s, **m)  # noqa: E731
    t, kw = torch.ones(1, **m), {"return_dict": False}
    if cls.startswith("UNet2D"):
        s = 8 if c.get("sample_size") is None else min(int(c["sample_size"] if not isinstance(c["sample_size"], (list, tuple)) else c["sample_size"][0]), 16)
        if c.get("cross_attention_dim"):
            dim = c["cross_attention_dim"]
            kw["encoder_hidden_states"] = e(1, 8, dim if isinstance(dim, int) else dim[0])
        if c.get("addition_embed_type") == "text_time":  # SDXL: pooled text embeds + 6 micro-conditioning ids
            kw["added_cond_kwargs"] = {"text_embeds": e(1, c["projection_class_embeddings_input_dim"] - 6 * c["addition_time_embed_dim"]),
                                       "time_ids": e(1, 6)}
        if c.get("class_embed_type") is not None or c.get("num_class_embeds"):
            kw["class_labels"] = e(1, c["projection_class_embeddings_input_dim"]) if c.get("class_embed_type") == "projection" \
                else torch.zeros(1, dtype=torch.long, **m)
        return (e(1, c["in_channels"], s, s), torch.tensor(1, **m)), kw
    if cls == "ControlNetModel":  # the UNet's down half + a conditioning image at 8x the latent size
        (x, ts), kw = denoiser_inputs("UNet2DConditionModel", c)
        return (x, ts), kw | {"controlnet_cond": e(1, c.get("conditioning_channels") or 3, 8 * x.shape[-2], 8 * x.shape[-1])}
    if cls == "FluxControlNetModel":  # Flux's inputs + a packed conditioning latent (+ a mode id for Union)
        (x,), kw = denoiser_inputs("FluxTransformer2DModel", c)
        kw["controlnet_cond"] = e(*x.shape)
        if c.get("num_mode"):
            kw["controlnet_mode"] = torch.zeros(1, 1, dtype=torch.long, **m)
        return (x,), kw
    if cls == "T2IAdapter":
        return (e(1, c.get("in_channels") or 3, 64, 64),), {}
    if cls in ("FluxTransformer2DModel", "Flux2Transformer2DModel"):  # packed latent [B, S, C] + per-token rope ids
        n_ax = len(c.get("axes_dims_rope") or [16, 56, 56])
        kw |= {"encoder_hidden_states": e(1, 8, c["joint_attention_dim"]), "timestep": t, "img_ids": e(16, n_ax), "txt_ids": e(8, n_ax)}
        if cls == "FluxTransformer2DModel":
            kw["pooled_projections"] = e(1, c["pooled_projection_dim"])
        if c.get("guidance_embeds"):
            kw["guidance"] = t
        return (e(1, 16, c["in_channels"]),), kw
    if cls == "SD3Transformer2DModel":
        s = 4 * c.get("patch_size", 2)
        kw |= {"encoder_hidden_states": e(1, 8, c["joint_attention_dim"]), "pooled_projections": e(1, c["pooled_projection_dim"]), "timestep": t}
        return (e(1, c["in_channels"], s, s),), kw
    if cls == "WanTransformer3DModel":
        pt, ph, pw = c.get("patch_size") or (1, 2, 2)
        kw |= {"timestep": t, "encoder_hidden_states": e(1, 8, c["text_dim"])}
        if c.get("image_dim"):  # I2V: CLIP image tokens
            kw["encoder_hidden_states_image"] = e(1, 8, c["image_dim"])
        return (e(1, c["in_channels"], 2 * pt, 4 * ph, 4 * pw),), kw
    if cls == "QwenImageTransformer2DModel":
        kw |= {"encoder_hidden_states": e(1, 8, c["joint_attention_dim"]), "encoder_hidden_states_mask": torch.ones(1, 8, **m),
               "timestep": t, "img_shapes": [[(1, 4, 4)]]}
        if c.get("guidance_embeds"):
            kw["guidance"] = t
        return (e(1, 16, c["in_channels"]),), kw
    if cls == "LTXVideoTransformer3DModel":
        kw |= {"encoder_hidden_states": e(1, 8, c["caption_channels"]), "timestep": t, "encoder_attention_mask": torch.ones(1, 8, **m),
               "num_frames": 2, "height": 4, "width": 2}
        return (e(1, 16, c["in_channels"]),), kw
    return None


def example_inputs(library: str, cls: str, cfg: dict, model, kind: str = "causal"):
    import torch
    meta = {"device": "meta"}
    if library == "timm":
        size = (cfg.get("pretrained_cfg") or {}).get("input_size") or [3, 224, 224]
        return (torch.empty(1, *size, **meta),), {}
    if library == "transformers":
        import inspect
        c = model.config
        if model.main_input_name == "pixel_values":  # ViT, ConvNeXt, Swin, ...: one image at the config's size
            size = c.image_size if isinstance(c.image_size, int) else c.image_size[0]
            x = torch.empty(1, getattr(c, "num_channels", 3), size, size, **meta)
        elif model.main_input_name == "input_ids":
            x = torch.zeros(1, 8, dtype=torch.long, **meta)
        else:
            raise TraceError(f"no example inputs for {type(model).__name__} ({model.main_input_name})")
        params = inspect.signature(model.forward).parameters
        kw = {"use_cache": False} if "use_cache" in params else {}
        if c.is_encoder_decoder and "decoder_input_ids" in params:
            kw["decoder_input_ids"] = torch.zeros(1, 8, dtype=torch.long, **meta)
        return (x,), kw
    if (got := denoiser_inputs(cls, dict(model.config))) is not None:
        return got
    if cls.startswith("Autoencoder"):
        return (torch.empty(1, model.config["in_channels"], 16, 16, **meta),), {"return_dict": False}
    raise TraceError(f"no example inputs for {library}:{cls}")


def _axes(p: Program, shape) -> list[str]:
    names = []
    for pos, n in enumerate(reversed(list(shape))):
        a = f"r{pos}_{int(n)}"
        p.dims[a] = int(n)
        names.insert(0, a)
    return names


def to_program(ep, name: str) -> tuple[Program, dict]:
    """(Program, {tensor name: checkpoint key}) from an ExportedProgram."""
    sig = ep.graph_signature
    params = {**sig.inputs_to_parameters, **sig.inputs_to_buffers}
    p = Program({}, symbols={})
    env, keys, uid = {}, {}, [0]

    def fresh(base):
        uid[0] += 1
        return f"{base}_{uid[0]}"

    views = {}

    def is_weight(arg) -> bool:
        """a parameter, or a cast of one (identity ops alias it in env)"""
        t = p.tensors.get(env.get(getattr(arg, "name", None), ""))
        return t is not None and t.role in ("parameter", "buffer")

    def weight_view(w, x_axis, out_in):
        """the weight as [fresh out axis, x's last axis] through Rename ops (a tied weight keeps its one declaration)"""
        if (w, x_axis) in views:
            return views[(w, x_axis)]
        a_out, a_in = p.axes(w) if out_in else p.axes(w)[::-1]
        o = fresh("o")
        p.dims[o] = p.dims[a_out]
        t = p.rename(fresh(f"{w}.out"), w, a_out, o)
        if a_in != x_axis:
            t = p.rename(fresh(f"{w}.in"), t, a_in, x_axis)
        views[(w, x_axis)] = t if out_in else p.permute(fresh(f"{w}.oi"), t, [o, x_axis])
        return views[(w, x_axis)]

    def val(n):
        v = n.meta.get("val")
        return v if hasattr(v, "shape") else None

    def tensor_of(arg):
        if hasattr(arg, "name") and arg.name in env:
            return env[arg.name]
        if isinstance(arg, (int, float)) and not isinstance(arg, bool):
            return p.constant(f"c_{str(arg).replace('.', 'p').replace('-', 'm')}", float(arg))
        return None

    def opaque(node, out_name, shape, dtype="float32", via=None):
        ins = [env[a.name] for a in (via or node).all_input_nodes if a.name in env]
        if not ins:
            ins = [p.constant("c_zero", 0.0)]
        axes = _axes(p, shape)
        p._emit("Opaque", str(node.target).replace("aten.", "").replace(".default", ""), out_name, ins, axes)
        return out_name

    for node in ep.graph.nodes:
        p.module = (node.meta.get("nn_module_stack") and list(node.meta["nn_module_stack"].values())[-1][0]) or name
        p.module_type = (node.meta.get("nn_module_stack") and str(list(node.meta["nn_module_stack"].values())[-1][1]).split(".")[-1].strip("'>")) or "Module"
        if node.op == "placeholder":
            v = val(node)
            if v is None:
                continue
            if node.name in params:
                key = params[node.name]
                t = p.declare(node.name, "parameter" if node.name in sig.inputs_to_parameters else "buffer",
                              _axes(p, v.shape), checkpoint_key=key, state_dict_key=key)
                keys[t] = key
            else:
                t = p.declare(node.name, "input", _axes(p, v.shape), "int64" if "int" in str(v.dtype) else "float32")
            env[node.name] = t
            continue
        if node.op == "output":
            outs = [a for a in node.args[0] if hasattr(a, "name") and a.name in env] if isinstance(node.args[0], (list, tuple)) else [node.args[0]]
            src = env[outs[0].name]
            p.reshape("L", src, list(p.axes(src)))  # the graph output is named L
            continue
        if node.op != "call_function":
            continue
        target = node.target
        if target is operator.getitem:  # one output of a multi-output op (chunk, split, ...): read that op's inputs
            v = val(node)
            src = node.args[0]
            if v is not None:
                env[node.name] = env[src.name] if src.name in env and val(src) is not None else opaque(node, node.name, v.shape, via=src)
            continue
        op = str(target).replace("aten.", "").split(".")[0].rstrip("_")
        v = val(node)
        if v is None:
            continue
        try:
            if op in IDENTITY:
                env[node.name] = env[node.args[0].name]
                continue
            if op in ("linear", "addmm", "mm") and is_weight(w := node.args[{"linear": 1, "addmm": 2, "mm": 1}[op]]):
                x = env[node.args[1 if op == "addmm" else 0].name]
                out_axis = _axes(p, v.shape)[-1]
                wv = weight_view(env[w.name], p.axes(x)[-1], op == "linear")
                y = p.contract(f"{node.name}.mm", x, wv, [p.axes(x)[-1]])
                y = p.rename(f"{node.name}.r", y, p.axes(wv)[0], out_axis)
                bias = node.args[2] if op == "linear" and len(node.args) > 2 else node.args[0] if op == "addmm" else None
                env[node.name] = p.elementwise("add", node.name, y, env[bias.name]) if bias is not None else p.reshape(node.name, y, list(p.axes(y)))
                continue
            if op == "matmul" and is_weight(node.args[0]) and len(p.axes(env[node.args[1].name])) >= 2:
                # W[o, i] @ X[..., i, t] (HF Mamba's dt_proj.weight @ x^T): contract W's input axis with X's axis -2
                w, x = env[node.args[0].name], env[node.args[1].name]
                xa, wa = p.axes(x), p.axes(w)
                o = fresh("o")
                p.dims[o] = p.dims[wa[0]]
                wv = p.rename(fresh(f"{w}.out"), w, wa[0], o)
                if wa[1] != xa[-2]:
                    wv = p.rename(fresh(f"{w}.in"), wv, wa[1], xa[-2])
                y = p.contract(f"{node.name}.mm", x, wv, [xa[-2]])  # [..., t, o]
                canon = _axes(p, v.shape)  # [..., o, t]
                y = p.rename(f"{node.name}.r", y, o, canon[-2]) if canon[-2] not in p.axes(y) else y
                env[node.name] = p.permute(node.name, y, canon)
                continue
            if op in BINARY and len(node.args) >= 2:
                a, b = tensor_of(node.args[0]), tensor_of(node.args[1])
                if a and b:
                    env[node.name] = p.elementwise(BINARY[op], node.name, a, b)
                    continue
            if op in UNARY:
                env[node.name] = p.elementwise(UNARY[op], node.name, env[node.args[0].name])
                continue
            if op in ("view", "reshape", "_unsafe_view", "flatten"):
                env[node.name] = p.reshape(node.name, env[node.args[0].name], _axes(p, v.shape))
                continue
            if op == "embedding":
                table, idx = env[node.args[0].name], env[node.args[1].name]
                env[node.name] = p.gather(node.name, table, idx, p.axes(table)[0])
                continue
        except (AssertionError, KeyError, ValueError):
            pass  # names do not line up: an opaque op keeps the graph honest
        env[node.name] = opaque(node, node.name, v.shape)
    return p, keys


def prune(doc: dict) -> dict:
    """Only what L depends on: masks, side outputs and unused heads are dropped (and their parameters stay stored, to
    be classed by binding)."""
    ops = doc["ops"]
    by_out = {o["out"]: o for o in ops}
    live, todo = set(), ["L"]
    while todo:
        n = todo.pop()
        if n in live:
            continue
        live.add(n)
        if n in by_out:
            todo += by_out[n]["inputs"]
    keep = [o for o in ops if o["out"] in live]
    index = {o["index"]: i for i, o in enumerate(keep)}
    for o in keep:
        o["index"] = index[o["index"]]
    tensors = []
    for t in doc["tensors"]:
        if t["name"] in live:
            t["produced_by"] = index.get(t["produced_by"]) if t["produced_by"] is not None else None
            tensors.append(t)
    return {**doc, "ops": keep, "tensors": tensors}


def trace_graph(library: str, cls: str, cfg: dict, name: str = "model", kind: str = "causal"):
    """(viz-graph, expected params) for a library class, path B."""
    g, expected, _ = trace(library, cls, cfg, name, kind)
    return g, expected


def trace(library: str, cls: str, cfg: dict, name: str = "model", kind: str = "causal"):
    """`_trace` in a worker process (see WORKERS)."""
    global _POOL
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    with _POOL_LOCK:
        if _POOL is None:
            import atexit
            _POOL = ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn"), max_tasks_per_child=25)
            atexit.register(_POOL.shutdown, wait=False, cancel_futures=True)
        pool = _POOL
    try:
        return pool.submit(_trace_in_worker, library, cls, cfg, name, kind).result()
    except BrokenProcessPool:  # a worker died (out of memory, a crash in native code): a fresh pool for the next trace
        with _POOL_LOCK:
            _POOL = None if _POOL is pool else _POOL
        raise TraceError("trace worker died") from None


def _trace_in_worker(*args):
    try:
        return _trace(*args)
    except TraceError:
        raise
    except Exception as e:  # noqa: BLE001 - torch exceptions may not pickle back to the parent
        raise TraceError(f"{type(e).__name__}: {str(e)[:300]}") from None


def _trace(library: str, cls: str, cfg: dict, name: str = "model", kind: str = "causal"):
    """(viz-graph, expected params, info): info["unused"] are state_dict keys the class declares that L never reads (a
    pooler beside a token classifier, a second head), classed `ignored` when stored; info["ties"] maps each key to every
    name of the same parameter object (lm_head.weight and embed_tokens.weight when tied)."""
    import json as _json

    import torch
    model = make_model(library, cls, cfg, kind)
    args, kw = example_inputs(library, cls, cfg, model, kind)
    try:
        ep = torch.export.export(model, args, kwargs=kw, strict=False)
    except Exception as e:  # noqa: BLE001 - data-dependent control flow and the like: report, never guess
        raise TraceError(f"export failed: {type(e).__name__}: {str(e)[:200]}") from None
    p, _ = to_program(ep, name)
    doc = prune(p.to_dict(name, {"B": 1, "T": 8}) | {"verification": []})
    g = viz_graph(_json.loads(_json.dumps(doc)), f"gpt2_pytorch.registry.trace {library}:{cls} (torch.export on meta)")
    transient = set(ep.graph_signature.non_persistent_buffers)  # never saved: HF position_ids and the like
    for t in g["tensors"]:
        if t["checkpoint_key"] in transient:
            t["checkpoint_key"], t["view"] = None, "non-persistent buffer"
    expected = [{k: t[k] for k in ("name", "role", "shape", "checkpoint_key", "view", "state_dict_key")} for t in g["tensors"]
                if t["role"] in ("parameter", "buffer") and t["checkpoint_key"]]
    sig = ep.graph_signature
    used = {e["checkpoint_key"] for e in expected}
    unused = sorted(set(sig.inputs_to_parameters.values()) | set(sig.inputs_to_buffers.values()) - transient - used)
    groups = {}
    for n, t in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(t), []).append(n)
    ties = {n: names for names in groups.values() if len(names) > 1 for n in names}
    # stored keys the class itself tells transformers to drop on load (GPT-2's old attn.bias masks, T5's spare bias)
    ignore_re = list(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
    return g, expected, {"unused": unused, "ties": ties, "ignore_re": ignore_re}
