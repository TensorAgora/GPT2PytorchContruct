import json
from collections import Counter
from datetime import datetime

from .program import Program

MODULE_TEXT = {
    "DistilGPT2LMHeadModel": "no own ops; the tied head is a Contraction of Xf with E over D",
    "GPT2Model": "no own ops; chains embeddings, 6 blocks, final LayerNorm",
    "GPT2Embeddings": "X_0 = token rows + position rows (1 add); dropout elided",
    "TokenEmbedding": "Gather: rows of E selected by ids (= one-hot(ids) · E)",
    "PositionEmbedding": "Gather: rows of Wp selected by pos",
    "TransformerBlock": "the two residual additions",
    "CausalSelfAttention": "head split (Slice, Reshape, Permute, Rename ×3), scores, mask, softmax, context, head merge",
    "GPT2MLP": "no own ops; fc, GELU, projection",
    "GPT2GELU": "9 ops: x·x, ·x, ·0.044715, +x, ·√(2/π), tanh, +1, x·0.5, product",
    "LayerNorm": "9 ops: mean, sub, square, mean, +eps, rsqrt, ·, ·gamma, +beta",
    "Linear": "Contraction x·W, then + bias (the lm_head has no bias: 1 op)",
    "Dropout": "identity in eval(): 0 ops",
}


def listing(p: Program, only=None) -> str:
    rows = []
    for op in p.ops:
        if only is None or only(op):
            label = op.primitive + (f".{op.fn}" if op.fn else "")
            rows.append(f"{op.index:03d}  {label:<18}{p.equation(op)}")
            rows.append(f"{'':23}[{op.module or 'model'}]")
    return "\n".join(rows)


def multiplications(p: Program) -> list:
    return [op for op in p.ops if p.is_multiplication(op)]


def _table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]


def render_markdown(p: Program, instances: Counter, verification: list, checkpoint: str, example: dict, tel_model_json=None) -> str:
    mults = multiplications(p)
    by_role = Counter(t.role for t in p.tensors.values())
    prims = Counter((op.primitive, op.fn) for op in p.ops)
    contractions = [op for op in mults if op.primitive == "Contraction"]
    macs = sum(p.cost(op) for op in contractions)
    out = [
        "# DistilGPT2 as tensor operations",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by `python -m tools.tensor_program` from `{checkpoint}`. "
        f"Example shape: B={example['B']}, T={example['T']} (the program itself is symbolic in B and T).",
        "",
        "Every operand is a tensor: weights, biases, inputs, masks, constants (rank-0 tensors) and every intermediate. "
        "`y = x·W + b` is three tensors `x`, `W`, `b` and two operations, the product `x·W` and the sum with `b`.",
        "",
        "## Summary",
        "",
        *_table(
            ["", "count"],
            [
                ["tensors", f"{len(p.tensors)}"],
                ["parameters (checkpoint tensors, tied head counted once)", f"{by_role['parameter']}"],
                ["buffers (causal masks, from the checkpoint's `attn.bias`)", f"{by_role['buffer']}"],
                ["inputs / constants", f"{by_role['input']} / {by_role['constant']}"],
                ["intermediates", f"{by_role['activation']}"],
                ["tensor operations", f"{len(p.ops)}"],
                ["**multiplications** (Contraction + elementwise ×)", f"**{len(mults)}** = {len(contractions)} contractions + {len(mults) - len(contractions)} elementwise"],
                ["contraction multiply-accumulates (B=2, T=8)", f"{macs:,}"],
                ["modules with no ops (dropout, eval mode)", f"{len(p.elided)}"],
            ],
        ),
        "",
        "## 1. Conventions",
        "",
        "- Axes are named: `B` batch, `T` tokens, `q`/`k` query/key copies of `T`, `D` hidden (768), `Dc` the head-merged channel (768), "
        "`H` heads (12), `Dh` head dim (64), `F` MLP width (3072), `QKV` (2304), `V` vocabulary (50257), `P` positions (1024).",
        "- `a · b` is a contraction: the axes `a` and `b` share and that are listed as `Σ` are summed; the others stay. `a × b`, `+`, `-`, `/` are "
        "elementwise and align operands by axis name (a bias `b[QKV]` meets `y[B,T,QKV]` by the axis `QKV`). Named axes make transposes unnecessary: "
        "`K.T` and the tied head's `E.T` are just contractions over the right axis.",
        "- Parameters are bound to the checkpoint **in its original Hugging Face layout**: Conv1D weights are `[in,out]` and need no transposition here.",
        "- `Mb_l` is the checkpoint's `attn.bias[0,0,:T,:T]` (1 = visible): `log(1) = 0`, `log(0) = -inf`, so adding `log(Mb_l)` is the causal mask.",
        "- Dropout is the identity in `eval()` and emits no operation; the primitive names follow TEL's TensorIR.",
        "",
        "## 2. What each module became",
        "",
    ]
    rows = []
    for type_name, count in instances.items():
        own = sum(1 for op in p.ops if op.module_type == type_name)
        rows.append([f"`{type_name}`", str(count), f"{own}" + (f" ({own / count:g} each)" if count > 1 and own else ""), MODULE_TEXT.get(type_name, "")])
    out += _table(["module type", "instances", "ops emitted", "lowering"], rows)

    out += ["", f"## 3. All tensor multiplications ({len(mults)})", "",
            "`·` contraction (matrix/tensor product), `×` elementwise product. `cost` is multiply-accumulates for a contraction and output elements for `×` (B=2, T=8).", ""]
    rows = [[str(i), f"{op.index:03d}", "contract" if op.primitive == "Contraction" else "mul", f"`{p.equation(op)}`", f"{p.cost(op):,}", f"`{op.module or 'model'}`"] for i, op in enumerate(mults, 1)]
    out += _table(["#", "op", "kind", "equation", "cost", "module"], rows)

    per_block = []
    for layer in range(instances["TransformerBlock"]):
        ops = [op for op in mults if f"blocks.{layer}." in op.module + "."]
        per_block.append([str(layer), str(sum(op.primitive == "Contraction" for op in ops)), str(sum(op.primitive != "Contraction" for op in ops)), f"{sum(p.cost(op) for op in ops if op.primitive == 'Contraction'):,}"])
    out += ["", "Per block (all six are identical in structure):", "", *_table(["block", "contractions", "elementwise ×", "contraction MACs"], per_block)]

    others = Counter({k: v for k, v in prims.items() if not (k[0] == "Contraction" or k == ("Elementwise", "mul"))})
    out += ["", "## 4. The other tensor operations", "",
            "Not multiplications: sums and differences (bias, residual, eps, mask, softmax shift, GELU terms), division (score scale, softmax normalisation), "
            "the nonlinearities `exp`, `tanh`, `rsqrt`, `log`, reductions, and data movement. They are in the full listing "
            "(`artifacts/tensor_program.txt`).", ""]
    out += _table(["primitive", "fn", "count"], [[k[0], k[1] or "", str(v)] for k, v in sorted(others.items(), key=lambda kv: (kv[0][0], -kv[1]))])
    out += ["", "The embedding lookup is a Gather; as a multiplication it is `one_hot(ids)[B,T,V] · E[V,D]` (617,611,328 MACs at B=2, T=8), which selects the same rows exactly."]

    out += ["", "## 5. Tensors", "", "### Parameters and buffers (checkpoint keys, original layout)", ""]
    seen, rows = set(), []
    for t in p.tensors.values():
        if t.role in ("parameter", "buffer"):
            pattern = t.name.rsplit("_", 1)[0] if t.name.rsplit("_", 1)[-1].isdigit() else t.name
            if pattern in seen:
                continue
            seen.add(pattern)
            key = t.checkpoint_key.replace(".0.", ".{l}.") if t.name.endswith("_0") else t.checkpoint_key
            rows.append([f"`{pattern}{'_{l}' if t.name.endswith('_0') else ''}`", t.role, f"`{key}`", f"[{','.join(t.axes)}]", (f"1 × 1 × P × P, cut {t.view}" if t.view else " × ".join(map(str, t.shape))), "6 blocks" if t.name.endswith("_0") else "1"])
    out += _table(["symbol", "role", "checkpoint key", "axes", "shape", "count"], rows)
    out += ["", f"Tied weight: `{', '.join(f'{k} = {v}' for k, v in p.aliases.items())}` (the same tensor, used as the embedding table and as the output matrix).",
            "", "### Inputs and constants", ""]
    out += _table(["tensor", "role", "axes / value"], [[f"`{t.name}`", t.role, f"[{','.join(t.axes)}] {t.dtype}" if t.role == "input" else f"{t.value!r}"] for t in p.tensors.values() if t.role in ("input", "constant")])
    out += ["", "### Intermediates and their PyTorch counterparts", "",
            "Every symbol below is a tensor in this program **and** a tensor in `model(input_ids, return_debug=True).debug`, compared on every run.", ""]
    mapped = {}
    for t in p.tensors.values():
        if t.pytorch_name:
            pattern = t.name.rsplit("_", 1)[0] + "_{l}" if t.name.rsplit("_", 1)[-1].isdigit() and t.name != "X_0" else t.name
            if pattern == "X_{l}":
                pattern = "X_{l+1}"  # the output of block l
            mapped.setdefault(pattern, (t.pytorch_name.replace(".0.", ".{l}.") if "block." in t.pytorch_name else t.pytorch_name, t.axes, t.shape))
    out += _table(["symbol", "PyTorch debug name", "axes", "shape (B=2, T=8)"], [[f"`{k}`", f"`{v[0]}`", f"[{','.join(v[1])}]", " × ".join(map(str, v[2]))] for k, v in mapped.items()])
    out += ["", f"{sum(1 for t in p.tensors.values() if t.pytorch_name)} tensors are compared in total; the rest are finer intermediates (`<symbol>.<part>`) such as `N1_0.centered` or `A_0.exp`."]

    block0_end = max(o.index for o in p.ops if o.out == "X_1")
    final_start = min(o.index for o in p.ops if o.out == "Xf.mean")
    out += ["", "## 6. The program: embeddings, block 0, head", "",
            f"Blocks 1 to 5 repeat block 0 with the index changed (`_1` … `_5`). All {len(p.ops)} operations: `artifacts/tensor_program.txt`.", "",
            "```text", listing(p, lambda op: op.index <= block0_end or op.index >= final_start), "```"]

    out += ["", "## 7. Verification", "",
            "The program is executed from the checkpoint's own tensors (original HF layout, no transposition and no use of the pure model's weights) "
            "and every tensor that has a PyTorch counterpart is compared, with error relative to that tensor's max magnitude (tolerance 1e-5).", ""]
    out += _table(["sample", "input shape", "tensors compared", "worst rel. error", "worst tensor", "logits max abs err", "argmax equal", "result"],
                  [[v["sample"], str(v["shape"]), str(v["compared"]), f"{v['worst_relative_error']:.2e}", f"`{v['worst_tensor']}`", f"{v['logits_max_abs_error']:.2e}", str(v["argmax_identical"]), "PASS" if v["ok"] else "FAIL"] for v in verification])

    if tel_model_json:
        tel = Counter()
        for op in json.load(open(tel_model_json))["graph"]["operations"]:
            tel[(op["primitive"], op["attributes"].get("op"))] += 1
        ours = Counter((op.primitive, op.fn) for op in p.ops)
        ours[("Constant", None)] = by_role["constant"]
        notes = {
            ("Elementwise", "mul"): "GELU's x³ is two multiplications here, one `pow` in TEL",
            ("Elementwise", "pow"): "see mul",
            ("Elementwise", "sqrt"): "√Dh is the constant 8.0 here; TEL computes sqrt(D_h) at run time",
            ("Permute", None): "`K.T` and `E.T` need no transpose with named axes (7 in TEL)",
            ("Broadcast", None): "named-axis alignment is implicit in Elementwise here; TTIR makes it an explicit op",
            ("Constant", None): "constants are declared tensors here, not ops (shared, so fewer)",
            ("Convert", None): "TEL converts the integer `D_h` to fp32",
            ("ExternalRef", None): "TEL reads `D_h` as a runtime dimension tensor",
        }
        keys = sorted(set(ours) | set(tel), key=lambda k: (k[0], str(k[1])))
        out += ["", "## 8. Cross-check with TEL's emitted program", "",
                f"TEL's importer (`tensorparser`, `packages/model-import`) compiled the same checkpoint into {sum(tel.values())} primitive ops "
                f"(`{tel_model_json}`, produced by `node dist/cli.js artifacts/model.safetensors artifacts/tel --model distilbert/distilgpt2 "
                "--config test/fixtures/distilgpt2.config.json`). Same-named primitives, side by side:", ""]
        out += _table(["primitive", "fn", "this program", "TEL", "note"], [[k[0], k[1] or "", str(ours.get(k, 0)), str(tel.get(k, 0)), notes.get(k, "equal" if ours.get(k, 0) == tel.get(k, 0) else "")] for k in keys])
    return "\n".join(out) + "\n"
