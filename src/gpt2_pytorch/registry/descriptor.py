"""Architecture descriptors (master §4 → design doc §7.2): JSON documents under registry/descriptors/ with match,
hyperparameters, schemes, recipe, ties, ignored and provenance, plus the layer selectors a recipe uses to pick a block
per layer. Descriptors are data; blocks.py holds the code."""
import ast
import json
import operator
import re
from functools import cache
from pathlib import Path

DIR = Path(__file__).resolve().parent / "descriptors"
KEYS = ("id", "version", "match", "hyperparameters", "schemes", "recipe", "ties", "ignored", "provenance")
SELECTORS = ("uniform", "list", "first_k", "period", "pattern", "index_sets", "per_layer_config", "by_presence")


class DescriptorError(ValueError):
    pass


@cache
def registry_version() -> str:
    """Content hash of everything a graph depends on besides the repo: descriptors, registry code, the lowerings.
    Determinism (master §3) is per (repo, sha, registry version)."""
    import hashlib
    root = DIR.parent
    files = sorted(DIR.glob("*.json")) + sorted(root.glob("*.py")) + sorted((root.parent / "tensorprog").glob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:12]


def ids() -> list[str]:
    return sorted(p.stem for p in DIR.glob("*.json"))


@cache
def _raw(id_: str) -> dict:
    path = DIR / f"{id_}.json"
    if not path.exists():
        raise DescriptorError(f"no descriptor {id_!r} (have: {', '.join(ids())})")
    return json.loads(path.read_text())


def load(id_: str) -> dict:
    """The descriptor with `extends` resolved: the child's top-level keys win, dicts merge one level deep."""
    d = dict(_raw(id_))
    if parent := d.pop("extends", None):
        base = load(parent)
        for k, v in base.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict) and isinstance(d[k], dict) and k not in ("provenance",):
                d[k] = {**v, **d[k]}
        d["id"], d["version"] = _raw(id_)["id"], _raw(id_)["version"]
    validate(d)
    return d


def validate(d: dict) -> None:
    missing = [k for k in KEYS if k not in d]
    if missing:
        raise DescriptorError(f"{d.get('id')}: missing {missing}")
    if "template" not in d["recipe"]:
        raise DescriptorError(f"{d['id']}: recipe has no template")
    for slot, sel in d["recipe"].get("layers", {}).items():
        if not isinstance(sel, dict) or not any(s in sel for s in SELECTORS):
            raise DescriptorError(f"{d['id']}: layers.{slot} has no selector ({', '.join(SELECTORS)})")


# ---- default expressions: arithmetic over resolved names, parsed, never eval'd ----------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def evaluate(expr: str, env: dict):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise KeyError(n.id)
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise DescriptorError(f"unsupported expression {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


def config_value(cfg: dict, key: str):
    """`a.b` reads nested dicts (text_config.hidden_size); None when absent."""
    v = cfg
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            return None
        v = v[part]
    return v


def hyper_from_config(d: dict, cfg: dict) -> dict:
    """{name: (value, source)} from the descriptor's ordered sources: config keys, then `=expr` defaults."""
    out = {}
    for name, sources in d["hyperparameters"].items():
        for s in sources:
            if s.startswith("="):
                try:
                    out[name] = (evaluate(s[1:], {k: v for k, (v, _) in out.items()}), f"default:{s[1:]}")
                    break
                except KeyError:
                    continue
            if (v := config_value(cfg, s)) is not None:
                out[name] = (v, f"config:{s}")
                break
        else:
            raise DescriptorError(f"{d['id']}: no source for hyperparameter {name} (tried {sources})")
    return out


# ---- layer selectors: one block name per layer ------------------------------------------------------------------
def _num(v, cfg):
    return v if isinstance(v, int) else config_value(cfg, v)


def layer_blocks(sel: dict, cfg: dict, n_layers: int, names=()) -> list[str]:
    if "uniform" in sel:
        return [sel["uniform"]] * n_layers
    if "list" in sel:
        values = config_value(cfg, sel["list"])
        if not isinstance(values, list) or len(values) < n_layers:
            raise DescriptorError(f"selector list: {sel['list']} is not a list of {n_layers}")
        m = sel.get("map", {})
        return [m.get(v, v) for v in values[:n_layers]]
    if "first_k" in sel:
        k = _num(sel["first_k"], cfg) or 0
        return [sel["first"] if i < k else sel["then"] for i in range(n_layers)]
    if "period" in sel:
        n, phase = _num(sel["period"], cfg), _num(sel.get("phase", 0), cfg) or 0
        return [sel["on"] if i % n == phase else sel["off"] for i in range(n_layers)]
    if "pattern" in sel:
        pattern = config_value(cfg, sel["pattern"])
        if not isinstance(pattern, str) or len(pattern) < n_layers:
            raise DescriptorError(f"selector pattern: {sel['pattern']} is not a string of {n_layers}")
        return [sel["map"][c] for c in pattern[:n_layers]]
    if "index_sets" in sel:
        out = [sel["default"]] * n_layers
        for block, members in sel["index_sets"].items():
            for i in (members if isinstance(members, list) else config_value(cfg, members) or []):
                out[i] = block
        return out
    if "per_layer_config" in sel:
        layers = config_value(cfg, sel["per_layer_config"])
        return [sel["map"][layer[sel["field"]]] for layer in layers[:n_layers]]
    if "by_presence" in sel:
        stored = set(names)
        out = []
        for i in range(n_layers):
            out.append(next((b for b, pat in sel["by_presence"].items()
                             if re.sub(r"\{i\}", str(i), pat) in stored), sel["default"]))
        return out
    raise DescriptorError(f"no selector in {sel}")
