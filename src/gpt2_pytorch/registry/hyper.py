"""S7 hyperparameters (master R-09): every value resolved from an ordered list of sources and recorded with the one
that gave it — config key, GGUF key, header shape, catalog column or default. The prose of
csv/categories/<category>/metadata_spec.json is parsed for `a | b` alternatives and `(else X)` fallbacks only."""
import json
import re
from functools import cache
from pathlib import Path

from .descriptor import DescriptorError, config_value, evaluate

SPEC_ROOT = Path(__file__).resolve().parents[3] / "csv/categories"
NESTED = ("text_config", "llm_config", "language_config", "thinker_config")
SPEC_ALIASES = {"K": "topk"}
CATALOG = {"L": "num_layers", "D": "hidden_size", "Hq": "num_attention_heads", "Hkv": "num_kv_heads",
           "E": "routed_experts", "K": "experts_per_token", "V": "vocab_size"}
NAME = re.compile(r"(<arch>\.)?[A-Za-z_][\w.]*|len\((<arch>\.)?[\w.]+\)")


class HyperError(DescriptorError):
    pass


def parse_alternatives(text: str) -> list[str]:
    extra = [m.group(1).strip() for m in re.finditer(r"\(else ([^)]*)\)", text)]
    main = re.sub(r"\s\([^)]*\)", "", text)  # comments follow a space; len(x) does not
    return [p for p in (s.strip() for s in main.split("|") + extra) if NAME.fullmatch(p)]


@cache
def spec(category: str) -> dict:
    """{symbol: (config alternatives, gguf alternatives)} from the category's metadata_spec.json dims and attrs."""
    path = SPEC_ROOT / category / "metadata_spec.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text())
    out = {}
    for section in ("dims", "dims_mla", "attrs"):
        for sym, pair in (s.get(section) or {}).items():
            if isinstance(pair, list) and len(pair) == 2 and sym not in out:
                out[sym] = (parse_alternatives(pair[0]), parse_alternatives(pair[1]))
    return out


def sources_for(name: str, d: dict, category: str) -> list[str]:
    explicit = d["hyperparameters"][name]
    defaults = [s for s in explicit if s.startswith("=")]
    out = [s for s in explicit if not s.startswith("=")]
    cfg_alts, gguf_alts = spec(category).get(SPEC_ALIASES.get(name, name), ([], []))
    out += [a for a in cfg_alts if a not in out] + [f"gguf:{a}" for a in gguf_alts if f"gguf:{a}" not in out]
    if name in CATALOG:
        out.append(f"catalog:{CATALOG[name]}")
    return out + defaults


def _number(v):
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return {"true": True, "false": False}.get(v.lower(), v)
    return v


def _lookup(src: str, cfg: dict, kv: dict, tensors: dict, fmt: str, row: dict):
    """(value, source label) or (None, None)."""
    if src.startswith("gguf:"):
        key = src[5:].replace("<arch>", str(kv.get("general.architecture", "")))
        m = re.fullmatch(r"len\((.+)\)", key)
        v = kv.get(m.group(1) if m else key)
        if m and v is not None:
            v = v["count"] if isinstance(v, dict) else len(v)
        return (v, f"gguf:{key}") if v is not None else (None, None)
    if src.startswith("shape:"):
        m = re.fullmatch(r"shape:(.+)\[(-?\d+)\]", src)
        t = tensors.get(m.group(1)) if m else None
        return (t["shape"][int(m.group(2))], src) if t else (None, None)
    if src.startswith("catalog:"):
        v = _number(row.get(src[8:], ""))
        return (v, src) if v is not None else (None, None)
    if src.startswith("missing:"):
        want_fmt, key = src[8:].split(":", 1)
        return (key not in tensors, src) if fmt == want_fmt else (None, None)
    for c in [cfg] + [cfg[n] for n in NESTED if isinstance(cfg.get(n), dict)]:
        v = config_value(c, src)
        if v is not None:
            return v, f"config:{src}"
    return None, None


def resolve_hyper(d: dict, manifest: dict | None = None, row: dict | None = None, category: str | None = None,
                  cfg: dict | None = None) -> dict:
    """{name: (value, source)} in descriptor order. Raises HyperError naming the hyperparameter and what was tried."""
    manifest = manifest or {}
    cfg = cfg if cfg is not None else (manifest.get("config") or {}).get("config.json") or {}
    kv = (manifest.get("gguf") or {}).get("kv") or {}
    tensors = {t["key"]: t for t in manifest.get("tensors", [])}
    fmt = (manifest.get("model") or {}).get("format", "safetensors")
    category = category or (manifest.get("model") or {}).get("category", "")
    out, pending = {}, list(d["hyperparameters"])
    while pending:  # a default may name a hyperparameter declared after it: retry until no progress
        left = []
        for name in pending:
            tried = sources_for(name, d, category)
            for src in tried:
                if src.startswith("="):
                    try:
                        out[name] = (evaluate(src[1:], {k: v for k, (v, _) in out.items()}), f"default:{src[1:]}")
                        break
                    except KeyError:
                        continue
                v, label = _lookup(src, cfg, kv, tensors, fmt, row or {})
                if v is not None:
                    out[name] = (_number(v), label)
                    break
            else:
                left.append(name)
        if len(left) == len(pending):
            raise HyperError(f"{d['id']}: no source for {left[0]} (tried {', '.join(sources_for(left[0], d, category))})")
        pending = left
    out = {k: out[k] for k in d["hyperparameters"]}  # descriptor order
    return out


def dim_sources(d: dict, hyper: dict, graph: dict) -> dict:
    """{dim: source} for every dim of the graph (R-09)."""
    exprs = d["recipe"].get("dims", {})
    out = {}
    for dim in graph["dims"]:
        e = str(exprs.get(dim, dim))
        if dim in ("B", "T") or e in ("B", "T"):
            out[dim] = "example"
        elif e in hyper:
            out[dim] = hyper[e][1]
        elif dim in exprs:
            out[dim] = f"derived:{e}"
        elif dim in hyper:
            out[dim] = hyper[dim][1]
        else:
            out[dim] = f"lowering:{d['recipe']['template']}"
    return out
