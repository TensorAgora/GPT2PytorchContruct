"""HF ↔ GGUF names (master §6, design doc §7.6): GGUF naming schemes generated from the pinned gguf-py tables
(namemaps/gguf.json, tools/gen_namemaps.py) instead of hand-listed, plus an overlay of storage transforms
(namemaps/overlay.json): llama.cpp's Q/K RoPE permutation, Gemma's norm +1, and fused tensors split into the
parameters a descriptor names."""
import json
import re
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path

DIR = Path(__file__).resolve().parent / "namemaps"
# descriptor scheme role -> gguf MODEL_TENSOR
ROLE_TENSOR = {
    "embed": "TOKEN_EMBD", "attn_norm": "ATTN_NORM", "q": "ATTN_Q", "k": "ATTN_K", "v": "ATTN_V", "o": "ATTN_OUT",
    "ffn_norm": ("FFN_NORM", "FFN_PRE_NORM"),  # Gemma's pre_feedforward_layernorm; both are blk.{i}.ffn_norm
    "gate": "FFN_GATE", "up": "FFN_UP", "down": "FFN_DOWN", "norm": "OUTPUT_NORM", "head": "OUTPUT",
    "router": "FFN_GATE_INP", "experts_g": "FFN_GATE_EXP", "experts_u": "FFN_UP_EXP", "experts_d": "FFN_DOWN_EXP",
    "q_a": "ATTN_Q_A", "q_a_norm": "ATTN_Q_A_NORM", "q_b": "ATTN_Q_B", "kv_a": "ATTN_KV_A_MQA", "kv_a_norm": "ATTN_KV_A_NORM",
    "kv_b": "ATTN_KV_B", "shared_gate": "FFN_GATE_SHEXP", "shared_up": "FFN_UP_SHEXP", "shared_down": "FFN_DOWN_SHEXP",
    "q_norm": "ATTN_Q_NORM", "k_norm": "ATTN_K_NORM", "post_attn_norm": "ATTN_POST_NORM", "post_ffn_norm": "FFN_POST_NORM",
}


@cache
def table() -> dict:
    return json.loads((DIR / "gguf.json").read_text())


@cache
def overlay() -> dict:
    return json.loads((DIR / "overlay.json").read_text())


def gguf_scheme(hf_scheme: dict) -> dict:
    """The GGUF scheme for a descriptor's roles. Every mapped role gets a name, even when an arch lacks that tensor,
    so an unmapped parameter shows up as missing instead of passing silently."""
    names = table()["tensor_names"]
    return {role: names[_types(role)[0]] + ".{what}" for role in hf_scheme if role in ROLE_TENSOR}


def _types(role: str) -> tuple:
    t = ROLE_TENSOR[role]
    return t if isinstance(t, tuple) else (t,)


def hf_tensor(hf_key: str, arch: str | None = None) -> list[str]:
    """MODEL_TENSOR names gguf-py maps an HF name template to (suffix .weight/.bias/.{what} stripped)."""
    k = re.sub(r"\.(weight|bias|\{what\})$", "", hf_key)
    t = table()
    return (t["arch_hf_to_tensor"].get(arch or "", {}).get(k) or []) + t["hf_to_tensor"].get(k, [])


def split_roles(descriptor_id: str) -> set:
    """Roles the hf overlay binds as slices of a fused tensor (their scheme keys name roles, not stored tensors)."""
    rules = overlay().get("hf", {}).get(descriptor_id, {}).get("split", {})
    return {role for r in rules.values() for role, _ in r["parts"]}


def c3(hf_scheme: dict, arch: str | None = None, skip=()) -> list[str]:
    """Conformance C3: every HF scheme name maps (through gguf-py) to the MODEL_TENSOR its role names in GGUF."""
    bad = []
    for role, key in hf_scheme.items():
        if role in skip:
            continue
        if isinstance(key, str) and role in ROLE_TENSOR and not set(_types(role)) & set(hf_tensor(key, arch)):
            bad.append(f"{role}: {key} -> {hf_tensor(key, arch) or 'unmapped'}, GGUF says {'/'.join(_types(role))}")
    return bad


def _pattern(p: str) -> re.Pattern:
    return re.compile("^" + re.escape(p).replace(r"\{i\}", r"\d+").replace(r"\*", ".*") + "$")


def apply_overlay(expected: list[dict], manifest: dict, d: dict, scheme_name: str, scheme: dict, fmt: dict,
                  arch: str | None = None):
    """(expected with split parameters rebound, overrides {param: {checkpoint_key, view}}, transforms reported)."""
    rules = overlay().get(scheme_name, {}).get(arch if scheme_name == "gguf" else d["id"], {})
    stored = {t["key"]: t for t in manifest["tensors"]}
    expected = [dict(e) for e in expected]
    by_key = {e["checkpoint_key"]: e for e in expected if e["checkpoint_key"]}
    overrides, transforms = {}, []
    for pattern, rule in rules.get("split", {}).items():
        n_split = 0
        for i in range(int(fmt.get("L", 0))):
            fused = stored.get(pattern.format(i=i))
            sizes = [int(fmt[s]) for _, s in rule["parts"]]
            if not fused or sum(sizes) != fused["shape"][rule["axis"]]:
                continue
            off = 0
            for (role, _), n in zip(rule["parts"], sizes):
                e = by_key.get(scheme.get(role, "").format(**fmt, i=i, what="weight")) if role in scheme else None
                if e and list(e["shape"][1:]) == list(fused["shape"][1:]):
                    e["checkpoint_key"], e["view"] = fused["key"], f"[{off}:{off + n},:]"
                    overrides[e["name"]] = {"checkpoint_key": e["checkpoint_key"], "view": e["view"]}
                off += n
            n_split += 1
        if n_split:
            transforms.append({"transform": "split", "rule": pattern, "tensors": n_split, "parts": rule["parts"]})
    for kind in ("rope_permute", "offset"):
        for pattern, arg in rules.get(kind, {}).items():
            rx = _pattern(pattern)
            n = sum(1 for k in stored if rx.match(k))
            if n:
                transforms.append({"transform": kind, "rule": pattern, "tensors": n, "arg": arg})
    return expected, overrides, transforms


def transform_of(key: str, scheme_name: str, arch_or_id: str) -> list[str]:
    """Which overlay transforms apply to one stored tensor name."""
    rules = overlay().get(scheme_name, {}).get(arch_or_id, {})
    return [kind for kind in ("rope_permute", "offset", "split") for p in rules.get(kind, {}) if _pattern(p).match(key)
            or fnmatchcase(key, p)]
