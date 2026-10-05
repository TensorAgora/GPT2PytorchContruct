"""OPTIONAL parity check against Hugging Face Transformers (needs `pip install -e .[reference]`).

Builds the HF model from a locally constructed config (no network), loads the same checkpoint,
and compares logits on the same fixed ids in eval mode. Nothing under src/ imports transformers.

    python -m reference.compare_huggingface [--check-gelu-variants]
"""
import argparse

import torch
from torch import nn

from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_pretrained, load_state_dict_file
from gpt2_pytorch.data import FIXED_INPUT_IDS


def build_hf_model(state):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=50257, n_positions=1024, n_embd=768, n_layer=6, n_head=12,
        activation_function="gelu_new", layer_norm_epsilon=1e-5,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel._from_config(config, attn_implementation="eager")
    result = model.load_state_dict(state, strict=False)
    print(f"HF load_state_dict: missing={list(result.missing_keys)} unexpected={list(result.unexpected_keys)}")
    model.tie_weights()
    return model.eval()


def compare(ours: torch.Tensor, ref: torch.Tensor, label: str) -> bool:
    diff = (ours - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-12)
    ok = torch.allclose(ours, ref, rtol=1e-4, atol=1e-3)
    print(f"{label:<22} max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} max_rel={rel.max():.3e} allclose(rtol=1e-4,atol=1e-3)={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--check-gelu-variants", action="store_true", help="show that erf GELU breaks parity")
    args = parser.parse_args()

    ours, _, _ = load_pretrained(args.checkpoint)
    hf = build_hf_model(load_state_dict_file(args.checkpoint))
    with torch.no_grad():
        ours_logits = ours(FIXED_INPUT_IDS)
        hf_logits = hf(FIXED_INPUT_IDS).logits
    ok = compare(ours_logits, hf_logits, "tanh GELU (ours)")
    print("argmax identical:", torch.equal(ours_logits.argmax(-1), hf_logits.argmax(-1)))

    if args.check_gelu_variants:
        for block in ours.transformer.blocks:
            block.mlp.activation = nn.GELU()  # exact erf
        with torch.no_grad():
            compare(ours(FIXED_INPUT_IDS), hf_logits, "erf GELU (wrong)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
