from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import GPT2Config
from ..model import DistilGPT2LMHeadModel
from .infer_config import infer_config
from .mapping import map_key
from .pytorch_loader import load_torch_state_dict
from .safetensors_loader import load_safetensors_state_dict

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "weights" / "pytorch_model.bin"


class CheckpointError(RuntimeError):
    pass


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)  # checkpoint keys copied into the model (incl. transposed)
    transposed: list[str] = field(default_factory=list)  # subset of loaded: Conv1D weights
    tied: list[str] = field(default_factory=list)  # model aliases that share storage with another parameter
    ignored: list[str] = field(default_factory=list)  # checkpoint keys deliberately not loaded (verified)
    missing: list[str] = field(default_factory=list)  # model keys no checkpoint tensor fills
    unexpected: list[str] = field(default_factory=list)  # checkpoint keys with no mapping

    def summary(self) -> str:
        rows = [(n, getattr(self, n)) for n in ("loaded", "transposed", "tied", "ignored", "missing", "unexpected")]
        lines = [f"{name:<11}{len(items)}" for name, items in rows]
        for name in ("missing", "unexpected"):
            lines += [f"  {name}: {k}" for k in getattr(self, name)]
        return "\n".join(lines)


def load_state_dict_file(path) -> dict[str, torch.Tensor]:
    """Format from extension: .bin/.pt/.pth -> torch (weights_only), .safetensors -> safetensors."""
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return load_safetensors_state_dict(path)
    if suffix in {".bin", ".pt", ".pth"}:
        return load_torch_state_dict(path)
    raise ValueError(f"unsupported checkpoint extension {suffix!r}: {path}")


def load_checkpoint(model: DistilGPT2LMHeadModel, path, strict: bool = True) -> LoadReport:
    return load_state_dict_into(model, load_state_dict_file(path), strict=strict)


def load_state_dict_into(model: DistilGPT2LMHeadModel, state: dict[str, torch.Tensor], strict: bool = True) -> LoadReport:
    """Validate the whole mapping first, then copy. Raises CheckpointError on any gap (strict)."""
    report = LoadReport()
    targets = model.state_dict(keep_vars=True)  # includes both tied names
    buffers = dict(model.named_buffers())
    plan = []  # (source key, mapping)
    covered = set()
    problems = []

    for key in state:
        mapping = map_key(key)
        if mapping is None:
            report.unexpected.append(key)
            continue
        plan.append((key, mapping))
        if mapping.action in ("load", "transpose"):
            source = state[key]
            expected = targets.get(mapping.target)
            if expected is None:
                problems.append(f"{key}: target {mapping.target} not in model")
                continue
            got = tuple(source.t().shape) if mapping.action == "transpose" else tuple(source.shape)
            if got != tuple(expected.shape):
                problems.append(f"{key}: shape {got} (after mapping) != model {mapping.target} {tuple(expected.shape)}")
            covered.add(mapping.target)

    # tying: the model must alias, and an explicit checkpoint copy must equal the embedding
    canonical = {}
    for name, param in targets.items():
        canonical.setdefault(id(param), name)
    for key, mapping in plan:
        if mapping.action == "tie":
            alias, owner = mapping.target, canonical[id(targets[mapping.target])]
            if alias == owner:
                problems.append(f"{key}: model does not tie {alias} to another parameter")
            elif "transformer.wte.weight" in state and not torch.equal(state[key], state["transformer.wte.weight"]):
                problems.append(f"{key}: checkpoint lm_head differs from wte, cannot load into a tied model")
            covered.add(alias)
            report.tied.append(f"{alias} -> {owner}")
        elif mapping.action == "ignore" and mapping.target is not None:
            expected = buffers[mapping.target]
            n = expected.size(0)
            if tuple(state[key].shape) != (1, 1, n, n) or not torch.equal(state[key][0, 0].bool().to(expected.device), expected):
                problems.append(f"{key}: not equal to the model's causal mask, refusing to ignore it")
    # aliases the checkpoint does not carry (e.g. safetensors drops lm_head.weight): reconstruct by tying
    for name, param in targets.items():
        if name not in covered and canonical[id(param)] != name:
            report.tied.append(f"{name} -> {canonical[id(param)]} (reconstructed)")
            covered.add(name)
    report.missing = [name for name in targets if name not in covered]
    report.ignored = [key for key, mapping in plan if mapping.action == "ignore"]

    if strict and (problems or report.missing or report.unexpected):
        raise CheckpointError("\n".join(["checkpoint does not match model:", *problems, report.summary()]))

    with torch.no_grad():
        for key, mapping in plan:
            if mapping.action in ("load", "transpose"):
                source = state[key].t() if mapping.action == "transpose" else state[key]
                targets[mapping.target].copy_(source)
                report.loaded.append(key)
                if mapping.action == "transpose":
                    report.transposed.append(key)
    return report


def load_pretrained(path=DEFAULT_CHECKPOINT, device="cpu", strict: bool = True):
    """Infer config from the checkpoint, build the model, load it, eval(). Returns (model, config, report)."""
    state = load_state_dict_file(path)
    config: GPT2Config = infer_config(state)
    model = DistilGPT2LMHeadModel(config)
    report = load_state_dict_into(model, state, strict=strict)
    model.to(device).eval()
    return model, config, report
