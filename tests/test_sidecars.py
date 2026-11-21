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


def _m(*ts):
    return {"tensors": [{"key": k, "dtype": dt, "shape": sh, "file": "m.safetensors", "numel": 1} for k, dt, sh in ts]}


def test_modern_quant_forms_give_logical_principals():
    from gpt2_pytorch.registry.sidecars import apply
    p = "model.layers.0.mlp.down_proj"
    cases = {  # form -> (manifest, config, logical shape of p.weight, scheme)
        "fp8 tensor": (_m((f"{p}.weight", "F8_E4M3", [64, 256]), (f"{p}.weight_scale", "BF16", [1]), (f"{p}.input_scale", "BF16", [1])), {}, [64, 256], "fp8_tensor"),
        "int8 channel": (_m((f"{p}.weight", "I8", [64, 256]), (f"{p}.weight_scale", "F32", [64])), {}, [64, 256], "int8_channel"),
        "nvfp4 ct": (_m((f"{p}.weight_packed", "U8", [64, 128]), (f"{p}.weight_scale", "F8_E4M3", [64, 16]),
                        (f"{p}.weight_global_scale", "F16", [1]), (f"{p}.input_global_scale", "F16", [1])), {}, [64, 256], "nvfp4"),
        "nvfp4 modelopt": (_m((f"{p}.weight", "U8", [64, 128]), (f"{p}.weight_scale", "F8_E4M3", [64, 16]),
                              (f"{p}.weight_scale_2", "F32", []), (f"{p}.input_scale", "F32", [])), {}, [64, 256], "nvfp4"),
        "int4 packed": (_m((f"{p}.weight_packed", "I32", [64, 32]), (f"{p}.weight_scale", "BF16", [64, 4]), (f"{p}.weight_shape", "I64", [2]),
                           (f"{p}.weight_zero_point", "I32", [8, 4])),
                        {"quantization_config": {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}}}, [64, 256], "pack_quantized_int4"),
    }
    for form, (m, cfg, shape, scheme) in cases.items():
        s = apply(m, cfg)
        assert s.stored[f"{p}.weight"]["shape"] == shape, form
        others = {t["key"] for t in m["tensors"]} - {f"{p}.weight"}
        assert others <= set(s.aux) and all(r["scheme"] == scheme for r in s.records), form


def test_gptq_bias_is_aux_unless_bound():
    from gpt2_pytorch.registry.sidecars import apply
    p = "model.layers.0.mlp.down_proj"
    s = apply(_m((f"{p}.qweight", "I32", [32, 64]), (f"{p}.qzeros", "I32", [2, 8]), (f"{p}.scales", "F16", [2, 64]), (f"{p}.bias", "F16", [64])),
              {"quantization_config": {"quant_method": "gptq", "bits": 4}})
    assert s.aux[f"{p}.bias"] == f"{p}.weight" and s.stored[f"{p}.weight"]["shape"] == [64, 256]


def test_kv_cache_scales_and_block_int8_are_aux():
    from gpt2_pytorch.registry.sidecars import apply
    a = "model.layers.0.self_attn"
    s = apply(_m((f"{a}.k_proj.weight", "BF16", [8, 16]), (f"{a}.k_proj.k_scale", "F32", []),
                 (f"{a}.kv_b_proj.weight", "BF16", [8, 16]), (f"{a}.v_proj.v_scale", "F32", []),  # MLA: no v_proj
                 ("model.layers.0.mlp.down_proj.weight", "I8", [256, 512]), ("model.layers.0.mlp.down_proj.weight_scale_inv", "F32", [2, 4])), {})
    assert s.aux == {f"{a}.k_proj.k_scale": f"{a}.k_proj.weight", f"{a}.v_proj.v_scale": f"{a}.kv_b_proj.weight",
                     "model.layers.0.mlp.down_proj.weight_scale_inv": "model.layers.0.mlp.down_proj.weight"}
