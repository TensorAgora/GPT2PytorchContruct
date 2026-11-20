"""S5 quantization sidecars (master R-07, design doc §7.4). Before binding, packed principals are replaced by their
logical shape and every sidecar that satisfies its scheme's shape relation becomes `aux` of its principal. A companion
that fits no rule is left alone, so the binding verifier reports it. Recorded, never decoded."""
import math
from dataclasses import dataclass, field

BNB_SUFFIXES = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")


@dataclass
class Sidecars:
    stored: dict  # key -> tensor (logical principals in place of packed or missing ones)
    aux: dict = field(default_factory=dict)  # sidecar key -> principal key
    views: dict = field(default_factory=dict)  # principal key -> view text
    records: list = field(default_factory=list)  # [{tensor, principal, scheme}]
    unchecked: set = field(default_factory=set)  # principals whose logical shape is not in the header (bitsandbytes)

    def attach(self, sidecar, principal, scheme):
        self.aux[sidecar] = principal
        self.records.append({"tensor": sidecar, "principal": principal, "scheme": scheme})


def _quant(cfg: dict) -> dict:
    q = dict(cfg.get("quantization_config") or {})
    if not q and isinstance(cfg.get("quantization"), dict):  # MLX
        q = {"quant_method": "mlx", **cfg["quantization"]}
    return q


def apply(manifest: dict, cfg: dict | None = None) -> Sidecars:
    q = _quant(cfg or {})
    method, bits, group = q.get("quant_method"), int(q.get("bits") or 4), int(q.get("group_size") or 128)
    block = q.get("weight_block_size") or [128, 128]
    s = Sidecars({t["key"]: dict(t) for t in manifest["tensors"]})
    st = s.stored
    per = 32 // bits

    for key in sorted(st):
        t = st.get(key)
        if t is None:
            continue
        # block FP8: X.weight_scale_inv beside an F8 X.weight
        if key.endswith("_scale_inv") and (p := st.get(key.removesuffix("_scale_inv"))) and p["dtype"].startswith("F8"):
            want = [math.ceil(n / b) for n, b in zip(p["shape"], block)] + p["shape"][len(block):]
            if list(t["shape"]) == want:
                s.attach(key, p["key"], "fp8_block")
                s.views[p["key"]] = f"dequantized: {p['dtype']} x {key} {t['shape']} (block-wise)"
        # Mistral FP8: X.qscale_weight / X.qscale_act, one scale per tensor
        elif key.endswith((".qscale_weight", ".qscale_act")) and (p := st.get(key.rsplit(".", 1)[0] + ".weight")):
            if math.prod(t["shape"]) <= 1:
                s.attach(key, p["key"], "fp8_mistral")
                s.views[p["key"]] = f"dequantized: {p['dtype']} x {key.rsplit('.', 1)[0]}.qscale_weight (per tensor)"
        # MXFP4: X_blocks + X_scales -> logical X
        elif key.endswith("_blocks") and (sc := st.get(key.removesuffix("_blocks") + "_scales")):
            b, c = t["shape"], sc["shape"]
            if b[:-1] == c and b[-1] == 16:
                logical = key.removesuffix("_blocks")
                st[logical] = {"key": logical, "dtype": "MXFP4", "shape": b[:-2] + [b[-2] * 32], "file": t["file"], "virtual": True}
                s.attach(key, logical, "mxfp4")
                s.attach(sc["key"], logical, "mxfp4")
                s.views[logical] = f"mxfp4: {key} + {sc['key']} (E2M1, block 32)"
        # GPTQ / AWQ: P.qweight + qzeros + scales (+ g_idx) -> logical P.weight [out, in]
        elif key.endswith(".qweight") and (sc := st.get(key.removesuffix(".qweight") + ".scales")):
            base, qw, scs = key.removesuffix(".qweight"), t["shape"], sc["shape"]
            gptq = method == "gptq" or (method != "awq" and qw[1] == scs[1])
            if gptq and qw[1] == scs[1]:
                shape, scheme = [qw[1], qw[0] * per], "gptq"
            elif not gptq and scs[1] == qw[1] * per:
                shape, scheme = [qw[1] * per, qw[0]], "awq"
            else:
                continue
            st[f"{base}.weight"] = {"key": f"{base}.weight", "dtype": f"INT{bits}", "shape": shape, "file": t["file"], "virtual": True}
            parts = [p for p in (f"{base}.qweight", f"{base}.qzeros", f"{base}.scales", f"{base}.g_idx") if p in st]
            for p in parts:
                s.attach(p, f"{base}.weight", scheme)
            s.views[f"{base}.weight"] = f"{scheme} int{bits}: {' / '.join('.' + p.rsplit('.', 1)[1] for p in parts)} (group {group})"
        # MLX: P.weight U32 packed + P.scales + P.biases
        elif key.endswith(".scales") and (bi := st.get(key.removesuffix(".scales") + ".biases")) \
                and (w := st.get(key.removesuffix(".scales") + ".weight")) and w["dtype"] == "U32":
            if t["shape"] == bi["shape"] and t["shape"][0] == w["shape"][0]:
                w.update(shape=w["shape"][:-1] + [w["shape"][-1] * per], dtype=f"INT{bits}", packed=True)
                s.attach(key, w["key"], "mlx")
                s.attach(bi["key"], w["key"], "mlx")
                s.views[w["key"]] = f"mlx int{bits}: {w['key']} + .scales / .biases (group {group})"
        # bitsandbytes: P.weight.absmax / .quant_map / .quant_state.* -> P.weight (shape only in the payload)
        elif (key.endswith(BNB_SUFFIXES) or ".quant_state." in key) and (p := st.get(key.split(".weight.", 1)[0] + ".weight")):
            s.attach(key, p["key"], "bitsandbytes")
            s.unchecked.add(p["key"])
            s.views[p["key"]] = "bitsandbytes 4-bit: logical shape in quant_state (not read)"
    return s
