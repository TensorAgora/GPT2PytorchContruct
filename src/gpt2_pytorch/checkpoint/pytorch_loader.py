from pathlib import Path

import torch


def load_torch_state_dict(path) -> dict[str, torch.Tensor]:
    """Safe (weights_only) load of a pickled state_dict, always onto CPU."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path} is not a flat {{name: tensor}} state_dict")
    return dict(state)
