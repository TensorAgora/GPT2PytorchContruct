"""S8 binding verification (master R-10, design doc §7.5): every stored tensor falls in exactly one class — bound,
tied, ignored, aux or component — or the candidate fails with a near-miss report naming what is missing, misshaped or
unclaimed. The header proposes; this decides."""
import re
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

CLASSES = ("bound", "tied", "ignored", "aux", "component")
STACK = re.compile(r"stack of (?P<pre>.*?)\{(?P<a>\d+)…(?P<b>\d+)\}(?P<post>\S*)")


@dataclass
class Binding:
    ok: bool
    partition: dict = field(default_factory=dict)  # class -> sorted keys
    missing: list = field(default_factory=list)  # [param, key]
    misshaped: list = field(default_factory=list)  # [param, key, graph shape, header shape]
    unclaimed: list = field(default_factory=list)  # keys
    views: dict = field(default_factory=dict)  # param name -> view text the binding adds (quantized storage)
    sidecars: list = field(default_factory=list)  # [{tensor, principal, scheme}] attached (06e)

    def near_miss(self, limit=20) -> dict:
        return {"missing": self.missing[:limit], "misshaped": self.misshaped[:limit], "unclaimed": self.unclaimed[:limit],
                "counts": {"missing": len(self.missing), "misshaped": len(self.misshaped), "unclaimed": len(self.unclaimed)}}

    def checkpoint(self, manifest: dict) -> dict:
        """The `checkpoint` field tools/model_graph.py writes: files, tensors, bound, unbound by top-level prefix."""
        bound = set(self.partition.get("bound", [])) | set(self.partition.get("aux", [])) | set(self.partition.get("tied", []))
        keys = [t["key"] for t in manifest["tensors"]]
        unbound = [k for k in keys if k not in bound]
        return {"files": sorted({t["file"] for t in manifest["tensors"]}), "tensors": len(keys), "bound": len(bound),
                "unbound": dict(Counter(re.match(r"(model\.)?[^.]+", k).group() for k in unbound))}


def stack_members(view: str) -> list[str]:
    m = STACK.fullmatch(view.split(" (")[0])
    return [f"{m['pre']}{j}{m['post']}" for j in range(int(m["a"]), int(m["b"]) + 1)] if m else []


def conv_layout(want, got):
    """A depthwise conv weight [C, 1, K] stored squeezed (GGUF: [C, K]) or kernel-major (MLX: [C, K, 1]): the same
    numbers in another singleton layout, bound with the layout named."""
    want, got = list(want), list(got)
    if len(want) == 3 and want[1] == 1:
        if got == [want[0], want[2]]:
            return f"squeezed {got} (GGUF depthwise conv)"
        if got == [want[0], want[2], 1]:
            return f"{got} [C, K, 1] (MLX conv layout)"
    return None


def verify(expected: list[dict], manifest: dict, d: dict, aliases: dict | None = None, fmt: dict | None = None) -> Binding:
    """`fmt` (resolved hyperparameter values) fills templated component prefixes such as `model.layers.{L}.`."""
    from .sidecars import apply as sidecar_rules

    sc = sidecar_rules(manifest, (manifest.get("config") or {}).get("config.json") or {})
    stored = sc.stored  # logical principals in place of packed or missing ones
    claim, owner, b = {}, {}, Binding(False)
    for e in expected:
        key, view = e["checkpoint_key"], e["view"] or ""
        if key:
            t = stored.get(key)
            if t is None:
                b.missing.append([e["name"], key])
                continue
            if view.startswith("["):  # a cut of the stored tensor: its index list must match the stored rank
                if len(view.split("]")[0].lstrip("[").split(",")) != len(t["shape"]):
                    b.misshaped.append([e["name"], key, view, t["shape"]])
            elif key not in sc.unchecked and list(t["shape"]) != list(e["shape"]):
                if layout := conv_layout(e["shape"], t["shape"]):
                    b.views[e["name"]] = layout
                else:
                    b.misshaped.append([e["name"], key, e["shape"], t["shape"]])
            claim[key], owner[key] = "bound", e["name"]
        elif members := stack_members(view):
            for k in members:
                t = stored.get(k)
                if t is None:
                    b.missing.append([e["name"], k])
                elif list(t["shape"]) != list(e["shape"][1:]):
                    b.misshaped.append([e["name"], k, e["shape"][1:], t["shape"]])
                else:
                    claim[k] = "bound"
    shapes = {e["name"]: e["shape"] for e in expected}
    for key, target in (aliases or {}).items():
        if key in stored and key not in claim:
            if target in shapes and list(stored[key]["shape"]) != list(shapes[target]):  # a tied copy must match what it ties to
                b.misshaped.append([target, key, shapes[target], stored[key]["shape"]])
            claim[key] = "tied"
    prefixes = [c["prefix"].format(**(fmt or {})) for c in d.get("components", [])]
    for key, t in stored.items():
        if key in claim:
            continue
        if t.get("virtual"):  # a logical principal synthesized from sidecars; its parts are what is stored
            continue
        if key in sc.aux and claim.get(sc.aux[key]) == "bound":
            claim[key] = "aux"
            continue
        if any(fnmatchcase(key, pat) for pat in d.get("ignored", [])):
            claim[key] = "ignored"
        elif any(fnmatchcase(key, pat) for pat in d.get("aux", [])):  # descriptor-declared aux (e.g. router biases)
            claim[key] = "aux"
        elif any(key.startswith(pre) for pre in prefixes):
            claim[key] = "component"
        else:
            b.unclaimed.append(key)
    for principal, text in sc.views.items():
        if principal in owner and claim.get(principal) == "bound":
            b.views[owner[principal]] = text
    b.sidecars = [r for r in sc.records if claim.get(r["tensor"]) == "aux"]
    claim = {k: v for k, v in claim.items() if not stored.get(k, {}).get("virtual")}
    b.partition = {c: sorted(k for k, v in claim.items() if v == c) for c in CLASSES}
    b.ok = not (b.missing or b.misshaped or b.unclaimed)
    return b
