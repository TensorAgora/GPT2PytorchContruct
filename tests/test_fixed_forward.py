import pytest
import torch

from conftest import needs_checkpoint
from gpt2_pytorch.data import FIXED_INPUT_IDS

pytestmark = needs_checkpoint

# Golden values from the pure model after Hugging Face parity was established (max abs diff 7.6e-5).
GOLDEN_LAST_LOGITS = [-54.2574462890625, -47.81869888305664, -50.43892288208008, -51.153560638427734, -53.93793487548828]
GOLDEN_ARGMAX = [[383, 290, 1438, 318, 449, 88, 198, 198], [383, 257, 845, 286, 13, 766, 262, 198]]


def test_fixed_input_forward(real_model):
    assert FIXED_INPUT_IDS.shape == (2, 8)
    with torch.no_grad():
        logits = real_model(FIXED_INPUT_IDS)
    assert logits.shape == (2, 8, 50257) and torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, -1, :5], torch.tensor(GOLDEN_LAST_LOGITS), rtol=0, atol=1e-3)
    assert logits.argmax(-1).tolist() == GOLDEN_ARGMAX


def test_forward_is_bit_reproducible(real_model):
    with torch.no_grad():
        assert torch.equal(real_model(FIXED_INPUT_IDS), real_model(FIXED_INPUT_IDS))


def test_debug_path_gives_the_same_logits(real_model):
    with torch.no_grad():
        assert torch.equal(real_model(FIXED_INPUT_IDS), real_model(FIXED_INPUT_IDS, return_debug=True).logits)


def test_hugging_face_parity_when_available(real_model):
    pytest.importorskip("transformers")
    from reference.compare_huggingface import build_hf_model
    from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_state_dict_file

    hf = build_hf_model(load_state_dict_file(DEFAULT_CHECKPOINT))
    with torch.no_grad():
        torch.testing.assert_close(real_model(FIXED_INPUT_IDS), hf(FIXED_INPUT_IDS).logits, rtol=1e-4, atol=1e-3)
