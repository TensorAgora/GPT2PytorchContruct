"""Shared by every exporter: the source model, its reference logits, HF-named weights, and manifest entries.

Every exporter runs as its own process (`python -m tools.export_formats.<name> --model M --out DIR`), so TensorFlow,
Core ML and OpenVINO never share an interpreter. Each writes DIR/manifest.d/<name>.json; run_all merges them.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from gpt2_pytorch import DistilGPT2LMHeadModel, GPT2Config
from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_pretrained, load_state_dict_file
from gpt2_pytorch.checkpoint.mapping import checkpoint_key_for, map_key
from gpt2_pytorch.data import FIXED_INPUT_IDS

# small enough to commit as test fixtures, same architecture as DistilGPT2
TINY = GPT2Config(vocab_size=512, max_position_embeddings=64, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=256)
MODELS = ("distilgpt2", "tiny")


def parse_args(doc: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--model", default="distilgpt2", choices=MODELS)
    parser.add_argument("--out", help="output directory (default exports/<model>)")
    args = parser.parse_args()
    args.out = Path(args.out or f"exports/{args.model}")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def load_source(name: str) -> tuple[DistilGPT2LMHeadModel, torch.Tensor]:
    """(eval-mode model, example ids [2, 8]). distilgpt2: the real weights; tiny: seeded random weights."""
    if name == "distilgpt2":
        return load_pretrained(DEFAULT_CHECKPOINT)[0], FIXED_INPUT_IDS
    torch.manual_seed(0)
    return DistilGPT2LMHeadModel(TINY).eval(), FIXED_INPUT_IDS % TINY.vocab_size


def reference_logits(model, ids) -> np.ndarray:
    with torch.no_grad():
        return model(ids).numpy()


def hf_state_dict(name: str, model: DistilGPT2LMHeadModel) -> dict[str, torch.Tensor]:
    """Hugging Face GPT-2 naming and layout (Conv1D weights [in, out], lm_head tied to wte, attn.bias masks)."""
    if name == "distilgpt2":
        return load_state_dict_file(DEFAULT_CHECKPOINT)  # the original file, bit for bit
    c = model.config
    state = {}
    for key, tensor in model.state_dict().items():
        hf = checkpoint_key_for(key, c.num_layers)
        if hf is not None:
            state[hf] = tensor.t().contiguous() if map_key(hf).action == "transpose" else tensor.clone()
    state["lm_head.weight"] = state["transformer.wte.weight"]  # same storage, as HF saves it
    mask = torch.tril(torch.ones(c.max_position_embeddings, c.max_position_embeddings)).view(1, 1, *([c.max_position_embeddings] * 2))
    for i in range(c.num_layers):
        state[f"transformer.h.{i}.attn.bias"] = mask.clone()  # one buffer per layer, as HF stores them
    return state


def compare(reference: np.ndarray, got) -> dict:
    got = np.asarray(got, dtype=np.float64).reshape(reference.shape)
    diff = np.abs(got - reference)
    return {"max_abs_diff": float(diff.max()), "max_rel_diff": float(diff.max() / np.abs(reference).max())}


def _describe(path: Path) -> dict:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return {"path": str(path), "kind": "directory", "size": sum(p.stat().st_size for p in files),
                "files": [str(p.relative_to(path)) for p in files]}
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return {"path": str(path), "kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}


class Manifest:
    """Collects one entry per exported variant; a failing variant is recorded as failed, never hidden."""

    def __init__(self, args, exporter: str):
        self.args, self.exporter, self.entries = args, exporter, []

    def variant(self, fmt: str, variant: str, graph: str, fn, notes: str = "", tol: float = 1e-4):
        """fn() -> (list of written paths, verification dict). graph: what the file holds besides weights.
        Status is "ok" only if the reloaded file's logits are finite and within `tol` (relative to max |reference|)."""
        entry = {"exporter": self.exporter, "format": fmt, "variant": variant, "graph": graph, "notes": notes}
        try:
            paths, verify = fn()
            rel = verify["max_rel_diff"]
            status = "ok" if np.isfinite(rel) and rel <= tol else "mismatch"
            entry |= {"status": status, "files": [_describe(Path(p)) for p in paths], "verify": verify | {"tolerance": tol}}
            print(f"  {status:<7} {fmt:<16} {variant:<34} {verify}")
        except Exception as e:  # recorded, then the next variant still runs
            entry |= {"status": "failed", "error": f"{type(e).__name__}: {e}"[:2000], "files": []}
            print(f"  FAILED  {fmt:<16} {variant:<34} {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
        self.entries.append(entry)

    def skip(self, fmt: str, variant: str, reason: str):
        self.entries.append({"exporter": self.exporter, "format": fmt, "variant": variant, "status": "skipped", "notes": reason, "files": []})
        print(f"  skipped {fmt:<16} {variant:<34} {reason}")

    def write(self):
        d = self.args.out / "manifest.d"
        d.mkdir(exist_ok=True)
        (d / f"{self.exporter}.json").write_text(json.dumps(self.entries, indent=1))
