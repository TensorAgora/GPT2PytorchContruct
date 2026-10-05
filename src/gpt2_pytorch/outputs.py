from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CausalLMOutput:
    logits: torch.Tensor  # [B, T, vocab]
    loss: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None  # embeddings, block 0..N-1, final (debug mode only)
    debug: dict[str, Any] | None = None  # flat {tensor name: tensor} (debug mode only)
