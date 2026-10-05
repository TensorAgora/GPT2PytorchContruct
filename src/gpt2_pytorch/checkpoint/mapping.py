"""HF GPT-2 checkpoint key -> pure-PyTorch model key. The only place that knows HF naming or Conv1D.

HF GPT-2 uses `Conv1D`, a Linear whose weight is stored [in, out]; nn.Linear stores [out, in].
So c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj *weights* are transposed on load. Biases, embeddings
and LayerNorm parameters are never transposed.
"""
import re
from dataclasses import dataclass

# HF submodule (inside transformer.h.N.) -> (our submodule, is Conv1D)
BLOCK_MODULES = {
    "ln_1": ("ln_1", False),
    "attn.c_attn": ("attention.qkv_projection", True),
    "attn.c_proj": ("attention.output_projection", True),
    "ln_2": ("ln_2", False),
    "mlp.c_fc": ("mlp.fc", True),
    "mlp.c_proj": ("mlp.projection", True),
}

TOP_LEVEL = {
    "transformer.wte.weight": "transformer.embeddings.token_embedding.weight",
    "transformer.wpe.weight": "transformer.embeddings.position_embedding.weight",
    "transformer.ln_f.weight": "transformer.final_layer_norm.weight",
    "transformer.ln_f.bias": "transformer.final_layer_norm.bias",
}

TIED_KEYS = {"lm_head.weight"}  # must alias transformer.wte.weight; never copied

_BLOCK_KEY = re.compile(r"^transformer\.h\.(\d+)\.(.+)\.(weight|bias)$")
_BLOCK_BUFFER = re.compile(r"^transformer\.h\.(\d+)\.attn\.(bias|masked_bias)$")


@dataclass(frozen=True)
class KeyMapping:
    action: str  # "load" | "transpose" | "tie" | "ignore"
    target: str | None  # model state_dict key (for "ignore": the buffer the tensor must equal, if any)


def map_key(key: str) -> KeyMapping | None:
    """None means the key is unknown (reported as unexpected)."""
    if key in TOP_LEVEL:
        return KeyMapping("load", TOP_LEVEL[key])
    if key in TIED_KEYS:
        return KeyMapping("tie", "lm_head.weight")
    if m := _BLOCK_BUFFER.match(key):
        # HF registers the causal mask (and a -1e4 scalar in older versions) as buffers in the checkpoint.
        index, name = m.groups()
        target = f"transformer.blocks.{index}.attention.causal_mask" if name == "bias" else None
        return KeyMapping("ignore", target)
    if m := _BLOCK_KEY.match(key):
        index, hf_module, kind = m.groups()
        if hf_module in BLOCK_MODULES:
            ours, conv1d = BLOCK_MODULES[hf_module]
            action = "transpose" if conv1d and kind == "weight" else "load"
            return KeyMapping(action, f"transformer.blocks.{index}.{ours}.{kind}")
    return None


def checkpoint_keys(num_layers: int) -> list[str]:
    """Every key an HF GPT-2 checkpoint of this depth contains (lm_head.weight included)."""
    keys = ["transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight"]
    for i in range(num_layers):
        keys.append(f"transformer.h.{i}.attn.bias")
        keys += [f"transformer.h.{i}.{module}.{kind}" for module in BLOCK_MODULES for kind in ("weight", "bias")]
    return keys


def checkpoint_key_for(target: str, num_layers: int) -> str | None:
    """Inverse of map_key: the checkpoint key that fills model key `target` (None if no key does, e.g. a tied alias)."""
    for key in checkpoint_keys(num_layers):
        mapping = map_key(key)
        if mapping is not None and mapping.action != "tie" and mapping.target == target:
            return key
    return None
