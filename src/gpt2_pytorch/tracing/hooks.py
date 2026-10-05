import dataclasses
import re

import torch

ROOT_INPUT = "input.input_ids"
OUTPUT_ALIASES = {"": "model.logits", "transformer": "model.final_hidden"}


def flatten_tensors(obj) -> list[torch.Tensor]:
    """All tensors inside tensors / tuples / lists / dicts / dataclasses, in order."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        return [t for item in obj for t in flatten_tensors(item)]
    return []


def tensor_stem(module_name: str) -> str:
    """transformer.blocks.0.ln_1 -> block.0.ln_1 ; transformer.embeddings.token_embedding -> embedding.token"""
    name = module_name.removeprefix("transformer.")
    name = re.sub(r"^blocks\.(\d+)", r"block.\1", name)
    name = re.sub(r"^embeddings\.(token|position)_embedding", r"embedding.\1", name)
    return name or "model"


def tensor_names(module_name: str, direction: str, count: int) -> list[str]:
    """Stable hierarchical names for a module's `count` input or output tensors."""
    if direction == "output" and module_name in OUTPUT_ALIASES and count == 1:
        return [OUTPUT_ALIASES[module_name]]
    if direction == "input" and module_name == "" and count == 1:
        return [ROOT_INPUT]
    stem = tensor_stem(module_name)
    return [f"{stem}.{direction}"] if count == 1 else [f"{stem}.{direction}.{i}" for i in range(count)]
