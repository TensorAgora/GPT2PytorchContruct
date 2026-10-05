import re
from pathlib import Path

import torch

import gpt2_pytorch
from conftest import needs_checkpoint
from gpt2_pytorch import CausalLMOutput, DistilGPT2LMHeadModel, GPT2Config


def test_plain_call_returns_logits_and_debug_returns_output(tiny_model, tiny_ids):
    with torch.no_grad():
        logits = tiny_model(tiny_ids)
        out = tiny_model(tiny_ids, return_debug=True)
    assert torch.is_tensor(logits) and logits.shape == (2, 6, 100)
    assert isinstance(out, CausalLMOutput) and torch.equal(out.logits, logits)
    assert len(out.hidden_states) == 2 + 2 and out.loss is None  # embeddings + 2 blocks + final
    assert out.debug["model.logits"] is out.logits and out.debug["input.input_ids"] is tiny_ids
    assert torch.equal(out.hidden_states[-1], out.debug["model.final_hidden"])


def test_eval_forward_is_deterministic_and_dropout_is_off(tiny_model, tiny_ids):
    assert not tiny_model.training
    with torch.no_grad():
        assert torch.equal(tiny_model(tiny_ids), tiny_model(tiny_ids))


def test_train_mode_dropout_is_active(tiny_model, tiny_ids):
    tiny_model.train()
    torch.manual_seed(0)
    a = tiny_model(tiny_ids)
    b = tiny_model(tiny_ids)
    assert not torch.equal(a, b)


def test_default_parameter_count():
    model = DistilGPT2LMHeadModel(GPT2Config())
    assert sum(p.numel() for p in model.parameters()) == 81_912_576  # tied head counted once
    assert sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False)) == 120_509_952


def test_submodule_layout(tiny_model):
    t = tiny_model.transformer
    assert [type(b).__name__ for b in t.blocks] == ["TransformerBlock"] * 2
    assert t.token_embedding is t.embeddings.token_embedding and t.final_layer_norm.eps == 1e-5


def test_src_never_imports_transformers():
    src = Path(gpt2_pytorch.__file__).parent
    for path in src.rglob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+transformers", path.read_text(), re.M), path


@needs_checkpoint
def test_real_model_output_shape(real_model):
    with torch.no_grad():
        assert real_model(torch.zeros(2, 8, dtype=torch.long)).shape == (2, 8, 50257)
