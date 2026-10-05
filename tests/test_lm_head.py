import pytest
import torch

from gpt2_pytorch.model import causal_lm_loss


def test_logits_are_final_hidden_times_embedding_transpose(tiny_model, tiny_ids):
    with torch.no_grad():
        out = tiny_model(tiny_ids, return_debug=True)
    wte = tiny_model.transformer.token_embedding.weight
    assert tiny_model.lm_head.bias is None
    torch.testing.assert_close(out.logits, out.debug["model.final_hidden"] @ wte.T, rtol=1e-5, atol=1e-5)


def test_loss_shift_and_values(tiny_model, tiny_ids):
    with torch.no_grad():
        out = tiny_model(tiny_ids, labels=tiny_ids, return_debug=True)
    shift_logits, shift_labels = out.debug["loss.shift_logits"], out.debug["loss.shift_labels"]
    assert shift_logits.shape == (2, 5, 100) and torch.equal(shift_labels, tiny_ids[:, 1:])
    assert torch.equal(shift_logits, out.logits[:, :-1])
    manual = -torch.log_softmax(out.logits[:, :-1], -1).gather(-1, tiny_ids[:, 1:, None]).mean()
    torch.testing.assert_close(out.loss, manual)


def test_labels_without_debug_returns_output_with_loss(tiny_model, tiny_ids):
    out = tiny_model(tiny_ids, labels=tiny_ids)
    assert out.loss is not None and out.debug is None and out.hidden_states is None


def test_ignore_index_is_respected():
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3]])
    loss, _, shift_labels = causal_lm_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[0, [0, 2]], labels[0, [1, 3]])
    torch.testing.assert_close(loss, expected)
