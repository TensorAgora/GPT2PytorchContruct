import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gpt2_pytorch.tracing import load_state_dict_checked, matmul_graph


class NotGPT(nn.Module):
    """Conv, einsum, SDPA and a tied head: none of these has a rule in tensorprog/lower.py."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.qkv = nn.Linear(8, 24)
        self.mix = nn.Parameter(torch.randn(8, 8))
        self.embed = nn.Embedding(10, 8)
        self.head = nn.Linear(8, 10, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, img):
        x = self.conv(img).flatten(2).transpose(1, 2)  # [B, 36, 8]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.head(torch.einsum("btd,de->bte", y, self.mix) + x)


def test_state_dict_round_trip_and_graph():
    torch.manual_seed(0)
    state = NotGPT().state_dict()
    del state["head.weight"]  # tied copy, like Safetensors drops it
    model = load_state_dict_checked(NotGPT(), state)
    g = matmul_graph(model, (torch.randn(2, 3, 8, 8),))

    # assert modules, sources and wiring by position, not ATen ids or op choices (those are the decomposer's, per torch version)
    nodes = g["nodes"]
    pos = {v["id"]: v["index"] for v in nodes} | {"input": "input"}
    links = [{role: [pos[d] for d in o["from"]] for role, o in v["operands"].items() if o["from"]} for v in nodes]
    weights = [next((o["source"] for o in v["operands"].values() if o["kind"] == "parameter"), None) for v in nodes]
    assert [v["module"] for v in nodes] == ["conv", "qkv", "", "", "", "head"]
    assert weights == ["conv.weight", "qkv.weight", None, None, "mix", "head.weight"]
    assert links == [
        {"left": ["input"]},                # conv(img)
        {"left": [0]},                      # qkv(x)
        {"left": [1], "right": [1]},        # Q·Kᵀ inside SDPA
        {"left": [2], "right": [1]},        # probs·V
        {"left": [3]},                      # einsum with mix
        {"left": [0, 4]},                   # head(einsum + x): residual skips qkv
    ]
    assert g["outputs"] == [nodes[5]["id"]] and g["tied_parameters"] == [["embed.weight", "head.weight"]]
    assert nodes[0]["flops"] == 2 * (2 * 8 * 6 * 6) * (3 * 3 * 3)

def test_rejects_mismatched_state_dict():
    state = NotGPT().state_dict()
    del state["qkv.bias"]
    with pytest.raises(ValueError, match="qkv.bias"):
        load_state_dict_checked(NotGPT(), state)


def test_distilgpt2_one_node_per_contraction(tiny_model, tiny_ids):
    g = matmul_graph(tiny_model, (tiny_ids,))
    layers = tiny_model.config.num_layers
    assert len(g["nodes"]) == 6 * layers + 1  # qkv, QK^T, AV, out, fc, proj per block + lm_head
    assert g["nodes"][0]["module"] == "transformer.blocks.0.attention.qkv_projection" and g["nodes"][0]["deps"] == ["input"]
    assert g["nodes"][-1]["operands"]["right"]["source"] == "lm_head.weight"
    assert all(g["nodes"][[v["id"] for v in g["nodes"]].index(d)]["index"] < v["index"] for v in g["nodes"] for d in v["deps"] if d != "input")
