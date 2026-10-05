import torch
from torch.fx import symbolic_trace

CONCRETE = {"labels": None, "position_ids": None, "return_debug": False}


def test_symbolic_trace_matches_eager(tiny_model, tiny_ids):
    gm = symbolic_trace(tiny_model, concrete_args=CONCRETE)
    targets = {n.target for n in gm.graph.nodes if n.op == "call_module"}
    assert "transformer.blocks.0.attention.qkv_projection" in targets and "lm_head" in targets
    with torch.no_grad():
        assert torch.equal(gm(tiny_ids), tiny_model(tiny_ids))


def test_traced_graph_works_for_other_sequence_lengths(tiny_model):
    gm = symbolic_trace(tiny_model, concrete_args=CONCRETE)
    ids = torch.tensor([[3, 1, 4]])
    with torch.no_grad():
        assert torch.equal(gm(ids), tiny_model(ids))


def test_torch_export_matches_eager(tiny_model, tiny_ids):
    with torch.no_grad():
        exported = torch.export.export(tiny_model, (tiny_ids,))
        assert torch.allclose(exported.module()(tiny_ids), tiny_model(tiny_ids), atol=1e-6)
    kinds = {s.kind.name for s in exported.graph_signature.input_specs}
    assert {"PARAMETER", "BUFFER", "USER_INPUT"} <= kinds
