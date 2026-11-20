"""The Qwen3.5 lowering against transformers' own Qwen3.5 at tiny dims (random weights), both layer types."""
import pytest
import torch

qwen3_5 = pytest.importorskip("transformers.models.qwen3_5")

from gpt2_pytorch.tensorprog.interpret import run_program  # noqa: E402
from gpt2_pytorch.tensorprog.qwen3_5 import lower_qwen3_5  # noqa: E402

TINY = dict(
    vocab_size=61, hidden_size=32, intermediate_size=40, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=6,
    linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    rms_norm_eps=1e-6, tie_word_embeddings=False, attn_output_gate=True,
    rope_parameters={"rope_type": "default", "rope_theta": 1000.0, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0],
                     "mrope_interleaved": True},
)


def test_qwen3_5_program_matches_transformers():
    torch.manual_seed(0)
    config = qwen3_5.Qwen3_5TextConfig(**TINY)
    config._attn_implementation = "eager"
    model = qwen3_5.Qwen3_5ForCausalLM(config).eval().float()
    with torch.no_grad():  # zero-init norms and dt_bias = 1 would hide a missing (1 + w) or a wrong decay path
        for prm in model.parameters():
            prm.normal_(0, 0.3)
    ids = torch.tensor([[3, 17, 5, 60, 8, 2, 41]])
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    program = lower_qwen3_5(config.to_dict(), batch=1, seq=ids.shape[1], prefix="model.")
    state = model.state_dict()
    used = {t.checkpoint_key for t in program.tensors.values() if t.checkpoint_key}
    assert used == set(state), f"unbound {set(state) - used}, unknown {used - set(state)}"
    got = run_program(program, state, ids)["L"]
    assert (got - ref).abs().max().item() < 1e-4 * ref.abs().max().item()
