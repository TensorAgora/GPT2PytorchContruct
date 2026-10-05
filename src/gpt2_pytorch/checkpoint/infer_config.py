import re

import torch

from ..config import GPT2Config

HEAD_DIM = 64  # not recoverable from tensor shapes; GPT-2 family convention (verified by HF parity)


def infer_config(state: dict[str, torch.Tensor], head_dim: int = HEAD_DIM) -> GPT2Config:
    """Derive the architecture from tensor shapes; every cross-check is asserted, not assumed."""
    vocab_size, hidden = state["transformer.wte.weight"].shape
    positions, hidden_wpe = state["transformer.wpe.weight"].shape
    layers = {int(m.group(1)) for k in state if (m := re.match(r"transformer\.h\.(\d+)\.", k))}
    num_layers = len(layers)
    assert layers == set(range(num_layers)), f"non-contiguous block indices: {sorted(layers)}"
    assert hidden == hidden_wpe, "wte/wpe hidden size mismatch"

    c_attn = state["transformer.h.0.attn.c_attn.weight"].shape
    c_fc = state["transformer.h.0.mlp.c_fc.weight"].shape
    assert tuple(c_attn) == (hidden, 3 * hidden), f"c_attn {tuple(c_attn)} != (C, 3C)"
    assert tuple(state["transformer.h.0.attn.c_proj.weight"].shape) == (hidden, hidden)
    intermediate = c_fc[1]
    assert tuple(state["transformer.h.0.mlp.c_proj.weight"].shape) == (intermediate, hidden)
    assert tuple(state["transformer.ln_f.weight"].shape) == (hidden,)
    assert hidden % head_dim == 0

    return GPT2Config(
        vocab_size=vocab_size,
        max_position_embeddings=positions,
        hidden_size=hidden,
        num_layers=num_layers,
        num_heads=hidden // head_dim,
        intermediate_size=intermediate,
    )
