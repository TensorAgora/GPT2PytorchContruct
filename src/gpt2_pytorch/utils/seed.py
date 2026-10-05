import torch


def set_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str = "cpu") -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but not available")
    return torch.device(name)
