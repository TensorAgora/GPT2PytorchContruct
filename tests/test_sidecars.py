"""One synthetic header per sidecar scheme (design doc §7.4): the right shape relation binds, a wrong one stays unclaimed."""
import copy

import pytest

from gpt2_pytorch.registry.bind import verify

D = {"ignored": [], "components": []}


def T(key, shape, dtype="F16"):
    return {"key": key, "shape": shape, "dtype": dtype, "file": "m.safetensors"}


def param(key, shape):
    return [{"name": "W", "role": "parameter", "shape": shape, "checkpoint_key": key, "view": None}]


CASES = {  # scheme: (config, stored tensors, principal key, logical shape, the sidecar to break)
    "fp8_block": ({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}},
                  [T("l.weight", [256, 300], "F8_E4M3"), T("l.weight_scale_inv", [2, 3], "F32")], "l.weight", [256, 300], "l.weight_scale_inv"),
    "fp8_mistral": ({}, [T("l.weight", [64, 64], "F8_E4M3"), T("l.qscale_weight", [], "F32"), T("l.qscale_act", [], "F32")],
                    "l.weight", [64, 64], "l.qscale_weight"),
    "mxfp4": ({"quantization_config": {"quant_method": "mxfp4"}},
              [T("e.gate_up_proj_blocks", [4, 64, 3, 16], "U8"), T("e.gate_up_proj_scales", [4, 64, 3], "U8")], "e.gate_up_proj", [4, 64, 96],
              "e.gate_up_proj_scales"),
    "gptq": ({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}},
             [T("l.qweight", [64, 256], "I32"), T("l.qzeros", [4, 32], "I32"), T("l.scales", [4, 256]), T("l.g_idx", [512], "I32")],
             "l.weight", [256, 512], "l.scales"),
    "awq": ({"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}},
            [T("l.qweight", [512, 32], "I32"), T("l.qzeros", [4, 32], "I32"), T("l.scales", [4, 256])], "l.weight", [256, 512], "l.scales"),
    "mlx": ({"quantization": {"bits": 4, "group_size": 64}},
            [T("l.weight", [256, 64], "U32"), T("l.scales", [256, 8]), T("l.biases", [256, 8])], "l.weight", [256, 512], "l.biases"),
    "bitsandbytes": ({"quantization_config": {"quant_method": "bitsandbytes"}},
                     [T("l.weight", [65536, 1], "U8"), T("l.weight.absmax", [1024], "F32"), T("l.weight.quant_map", [16], "F32")],
                     "l.weight", [256, 512], None),
}


@pytest.mark.parametrize("scheme", CASES)
def test_relation_binds(scheme):
    cfg, tensors, principal, shape, _ = CASES[scheme]
    b = verify(param(principal, shape), {"tensors": tensors, "config": {"config.json": cfg}}, D)
    assert b.ok, b.near_miss()
    assert {r["scheme"] for r in b.sidecars} == {scheme} and "W" in b.views
    assert principal not in b.partition["bound"] or any(t["key"] == principal for t in tensors)


@pytest.mark.parametrize("scheme", [s for s in CASES if CASES[s][4]])
def test_wrong_shape_stays_unbound(scheme):
    cfg, tensors, principal, shape, bad = CASES[scheme]
    tensors = copy.deepcopy(tensors)
    next(t for t in tensors if t["key"] == bad)["shape"] = [7, 7, 7]
    b = verify(param(principal, shape), {"tensors": tensors, "config": {"config.json": cfg}}, D)
    assert not b.ok and (bad in b.unclaimed or b.missing or b.misshaped)
