import torch

from conftest import needs_checkpoint


def test_lm_head_is_the_embedding_parameter(tiny_model):
    emb = tiny_model.transformer.token_embedding.weight
    assert tiny_model.lm_head.weight is emb
    assert tiny_model.lm_head.weight.data_ptr() == emb.data_ptr()
    state = tiny_model.state_dict()
    assert "lm_head.weight" in state and state["lm_head.weight"].data_ptr() == state["transformer.embeddings.token_embedding.weight"].data_ptr()


def test_unique_vs_logical_parameter_references(tiny_model):
    unique = sum(p.numel() for p in tiny_model.parameters())
    logical = sum(p.numel() for _, p in tiny_model.named_parameters(remove_duplicate=False))
    assert logical - unique == 100 * 32


def test_gradient_accumulates_from_both_uses(tiny_model, tiny_ids):
    out = tiny_model(tiny_ids, labels=tiny_ids)
    out.loss.backward()
    grad = tiny_model.transformer.token_embedding.weight.grad
    assert grad is tiny_model.lm_head.weight.grad
    assert (grad.abs().sum(1) > 0).all()  # the head's softmax gives every vocabulary row a gradient


def test_in_place_update_is_visible_through_both_names(tiny_model):
    with torch.no_grad():
        tiny_model.lm_head.weight[0, 0] = 123.0
    assert tiny_model.transformer.token_embedding.weight[0, 0] == 123.0


@needs_checkpoint
def test_tying_survives_loading(real_model):
    assert real_model.lm_head.weight is real_model.transformer.token_embedding.weight
