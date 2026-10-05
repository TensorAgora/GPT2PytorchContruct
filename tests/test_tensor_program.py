import json

import pytest
import torch
from torch import nn

from conftest import needs_checkpoint
from gpt2_pytorch import DistilGPT2LMHeadModel
from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_state_dict_file
from gpt2_pytorch.checkpoint.loader import load_state_dict_into
from gpt2_pytorch.checkpoint.mapping import checkpoint_key_for, checkpoint_keys
from gpt2_pytorch.data import SAMPLES
from gpt2_pytorch.tensorprog import lower, pytorch_name, run_program, verify
from gpt2_pytorch.tensorprog.lower import Lowering
from test_checkpoint_mapping import hf_state_for


@pytest.fixture
def tiny(tiny_config):
    """Tiny model loaded from a random HF-layout state dict, plus that state dict."""
    state = hf_state_for(tiny_config, seed=3)
    model = DistilGPT2LMHeadModel(tiny_config).eval()
    load_state_dict_into(model, state)
    return model, state


def test_checkpoint_key_inverse(tiny_config):
    assert checkpoint_key_for("transformer.blocks.1.attention.qkv_projection.weight", 2) == "transformer.h.1.attn.c_attn.weight"
    assert checkpoint_key_for("transformer.blocks.0.attention.causal_mask", 2) == "transformer.h.0.attn.bias"
    assert checkpoint_key_for("lm_head.weight", 2) is None  # tied: no key of its own


def test_tiny_program_runs_from_hf_layout_and_matches_the_model(tiny):
    model, state = tiny
    program, _ = lower(model, 2, 6)
    for ids in (torch.tensor([[1, 5, 9, 2, 7, 3], [4, 4, 8, 0, 6, 99]]), torch.tensor([[7, 8, 9]]), torch.randint(0, 100, (3, 5))):
        result = verify(program, model, state, ids)  # same program, other (B, T): axes are symbolic
        assert result["ok"] and result["compared"] == 1 + 2 * 15 + 2  # X_0, 15 per block (14 TEL symbols + Ga), Xf, L
        assert result["argmax_identical"]


def test_structure_counts_for_tiny_model(tiny):
    model, _ = tiny
    p, instances = lower(model, 2, 6)
    n = 2
    mults = [op for op in p.ops if p.is_multiplication(op)]
    assert sum(op.primitive == "Contraction" for op in p.ops) == 6 * n + 1
    assert len(mults) == (6 + 12) * n + 3 + 1  # per block 6 contractions + 12 elementwise ×; final LayerNorm 3; lm_head 1
    assert p.aliases == {"lm_head.weight": "E"} and len(p.elided) == 3 * n + 1
    assert instances["TransformerBlock"] == n and instances["Linear"] == 4 * n + 1


def test_every_operand_is_a_tensor_and_all_checkpoint_keys_are_used(tiny):
    model, state = tiny
    p, _ = lower(model, 2, 6)
    assert all(name in p.tensors for op in p.ops for name in op.inputs)
    assert all(len(op.inputs) == 2 for op in p.ops if p.is_multiplication(op))
    bound = {t.checkpoint_key for t in p.tensors.values() if t.checkpoint_key}
    assert bound | {"lm_head.weight"} == set(state) == set(checkpoint_keys(2))
    assert not any(t.role == "constant" and t.axes for t in p.tensors.values())  # constants are rank 0


def test_unknown_module_fails_loudly(tiny):
    model, _ = tiny
    with pytest.raises(NotImplementedError, match="ReLU"):
        Lowering(model, 1, 1).call("transformer.x", nn.ReLU(), "ids")


def test_symbol_names_match_the_debug_dict(tiny):
    model, _ = tiny
    p, _ = lower(model, 2, 6)
    assert [pytorch_name(s) for s in ("X_0", "X_1", "X_2", "N1_0", "Q_1", "A_0", "Xf", "L", "N1_0.centered")] == [
        "embedding.output", "block.0.output", "block.1.output", "block.0.ln_1.output", "block.1.attention.q",
        "block.0.attention.probs", "model.final_hidden", "model.logits", None,
    ]
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    debug = model(ids, return_debug=True).debug
    assert {t.pytorch_name for t in p.tensors.values() if t.pytorch_name} <= set(debug)


def test_program_json_is_serializable_and_complete(tiny):
    model, _ = tiny
    p, _ = lower(model, 2, 6)
    data = json.loads(json.dumps(p.to_dict("tiny", {"B": 2, "T": 6})))
    assert data["format"] == "tensormorph-tensor-program" and len(data["ops"]) == len(p.ops)
    assert all(op["equation"] and "multiplication" in op for op in data["ops"])
    assert next(t for t in data["tensors"] if t["name"] == "Mb_0")["view"] == "[0,0,:T,:T]"


@needs_checkpoint
def test_real_model_program_matches_pytorch_on_every_sample(real_model):
    state = load_state_dict_file(DEFAULT_CHECKPOINT)
    program, _ = lower(real_model, 2, 8)
    for name, ids in SAMPLES.items():
        result = verify(program, real_model, state, ids)
        assert result["ok"] and result["compared"] == 93 and result["argmax_identical"], (name, result)


@needs_checkpoint
def test_real_model_counts_and_checkpoint_coverage(real_model):
    state = load_state_dict_file(DEFAULT_CHECKPOINT)
    p, _ = lower(real_model, 2, 8)
    assert (len(p.tensors), len(p.ops)) == (475, 385)
    mults = [op for op in p.ops if p.is_multiplication(op)]
    assert (sum(op.primitive == "Contraction" for op in mults), len(mults)) == (37, 112)
    bound = {t.checkpoint_key for t in p.tensors.values() if t.checkpoint_key}
    assert bound | {"lm_head.weight"} == set(state)  # all 83 checkpoint tensors are operands (82 + the tied alias)
    macs = sum(p.cost(op) for op in mults if op.primitive == "Contraction")
    assert macs == 1_298_214_912


@needs_checkpoint
def test_program_runs_on_the_original_layout_not_on_transposed_weights(real_model):
    state = load_state_dict_file(DEFAULT_CHECKPOINT)
    p, _ = lower(real_model, 2, 8)
    qkv = p.tensors["W_qkv_0"]
    assert qkv.axes == ("D", "QKV") and tuple(state[qkv.checkpoint_key].shape) == (768, 2304)  # Conv1D [in, out], bound as is
    env = run_program(p, state, SAMPLES["sample_batch"])
    assert env["W_qkv_0"] is state[qkv.checkpoint_key]
