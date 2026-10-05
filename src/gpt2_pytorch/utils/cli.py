import argparse

from ..checkpoint import DEFAULT_CHECKPOINT, load_pretrained
from .seed import resolve_device, set_deterministic


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="pytorch_model.bin or model.safetensors")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return parser


def load_model(args):
    """Seeded, eval-mode model on args.device. Returns (model, config, report)."""
    set_deterministic(0)
    model, config, report = load_pretrained(args.checkpoint, resolve_device(args.device))
    return model, config, report
