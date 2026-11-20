"""The expert storage forms of design doc §15 bind to one graph's [E, Fe, D] / [E, D, Fe] expert parameters."""
from gpt2_pytorch.registry.bind import verify
from gpt2_pytorch.registry.namemaps import apply_overlay

E, Fe, D = 4, 6, 8
SCHEME = {"experts_g": "l.{i}.experts.gate.weight", "experts_u": "l.{i}.experts.up.weight", "experts_d": "l.{i}.experts.down.weight"}
FMT = {"L": 1, "E": E, "Fe": Fe, "D": D}
DESC = {"id": "t", "ignored": [], "components": []}


def T(key, shape, dtype="BF16"):
    return {"key": key, "shape": shape, "dtype": dtype, "file": "m.safetensors"}


def params(keys=None, views=None):
    keys, views = keys or {}, views or {}
    shapes = {"g": [E, Fe, D], "u": [E, Fe, D], "d": [E, D, Fe]}
    return [{"name": f"W{r}e_0", "role": "parameter", "shape": shapes[r], "checkpoint_key": keys.get(r), "view": views.get(r)} for r in "gud"]


def bind(expected, tensors, rules=None, cfg=None):
    m = {"tensors": tensors, "config": {"config.json": cfg or {}}}
    expected, overrides, _ = apply_overlay(expected, m, DESC, "hf", SCHEME, FMT, rules=rules or {})
    return verify(expected, m, DESC), overrides


def test_one_tensor_per_expert():
    views = {r: f"stack of l.0.experts.{{0…{E - 1}}}.{n}.weight" for r, n in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj"))}
    tensors = [T(f"l.0.experts.{e}.{n}.weight", [Fe, D] if n != "down_proj" else [D, Fe]) for e in range(E) for n in ("gate_proj", "up_proj", "down_proj")]
    b, _ = bind(params(views=views), tensors)
    assert b.ok and len(b.partition["bound"]) == 3 * E


def test_stacked_per_role():
    keys = {"g": "blk.0.ffn_gate_exps.weight", "u": "blk.0.ffn_up_exps.weight", "d": "blk.0.ffn_down_exps.weight"}
    b, _ = bind(params(keys), [T(keys["g"], [E, Fe, D], "Q4_K"), T(keys["u"], [E, Fe, D], "Q4_K"), T(keys["d"], [E, D, Fe], "Q6_K")])
    assert b.ok


def test_stacked_fused_axis1():
    keys = {r: SCHEME[f"experts_{r}"].format(i=0) for r in "gud"}
    rules = {"split": {"l.{i}.experts.gate_up_proj": {"axis": 1, "parts": [["experts_g", "Fe"], ["experts_u", "Fe"]]}}}
    b, o = bind(params(keys), [T("l.0.experts.gate_up_proj", [E, 2 * Fe, D]), T(keys["d"], [E, D, Fe])], rules)
    assert b.ok and o["Wue_0"]["view"] == f"[:,{Fe}:{2 * Fe},:]"


def test_stacked_fused_in_out():
    keys = {r: SCHEME[f"experts_{r}"].format(i=0) for r in "gud"}
    rules = {"split": {"l.{i}.experts.gate_up_proj": {"axis": 2, "layout": "in_out", "parts": [["experts_g", "Fe"], ["experts_u", "Fe"]]}}}
    b, o = bind(params(keys), [T("l.0.experts.gate_up_proj", [E, D, 2 * Fe]), T(keys["d"], [E, D, Fe])], rules)
    assert b.ok and o["Wge_0"]["view"] == f"[:,:,0:{Fe}] (in_out)"


def test_stacked_fused_packed_mxfp4():
    keys = {r: SCHEME[f"experts_{r}"].format(i=0) for r in "gud"}
    rules = {"split": {"l.{i}.experts.gate_up_proj": {"axis": 1, "interleave": True, "parts": [["experts_g", "Fe"], ["experts_u", "Fe"]]}}}
    blocks = [E, 2 * Fe, D // 32 if D >= 32 else 1, 16]
    tensors = [T("l.0.experts.gate_up_proj_blocks", [E, 2 * Fe, 1, 16], "U8"), T("l.0.experts.gate_up_proj_scales", [E, 2 * Fe, 1], "U8"),
               T(keys["d"], [E, D, Fe])]
    exp = params(keys)
    for e in exp[:2]:
        e["shape"] = [E, Fe, 32]  # one MXFP4 block of 32 values per row
    b, o = bind(exp, tensors, rules, {"quantization_config": {"quant_method": "mxfp4"}})
    assert blocks and b.ok and o["Wue_0"]["view"] == "[:,1::2,:]" and {r["scheme"] for r in b.sidecars} == {"mxfp4"}


def test_wrong_fused_size_does_not_bind():
    keys = {r: SCHEME[f"experts_{r}"].format(i=0) for r in "gud"}
    rules = {"split": {"l.{i}.experts.gate_up_proj": {"axis": 1, "parts": [["experts_g", "Fe"], ["experts_u", "Fe"]]}}}
    b, _ = bind(params(keys), [T("l.0.experts.gate_up_proj", [E, 2 * Fe + 1, D]), T(keys["d"], [E, D, Fe])], rules)
    assert not b.ok


def test_active_flops_equal_the_k_of_e_hand_counts():
    """07b DESIGN: mixtral 204,346,490,880 and deepseek-v3 586,829,070,336, the goldens otherwise unchanged."""
    import json
    from pathlib import Path

    from gpt2_pytorch.registry.build import build
    from gpt2_pytorch.registry.descriptor import hyper_from_config, load
    from gpt2_pytorch.registry.namemaps import gguf_scheme
    from tools.graph_check import compare

    for desc, golden, scheme, want in (("mixtral", "mixtral-8x7b-instruct-q4km", "gguf", 204_346_490_880),
                                       ("deepseek_v3", "deepseek-v3", "hf", 586_829_070_336)):
        d = load(desc)
        if scheme == "gguf":
            d = {**d, "schemes": {**d["schemes"], "gguf": gguf_scheme(d["schemes"]["hf"])}}
        cfg = json.loads(Path(f"tests/fixtures/spelled/{golden}.config.json").read_text())
        g, _ = build(d, hyper_from_config(d, cfg), scheme, cfg=cfg)
        assert g["totals"]["active_flops"] == want
        for m in g["multiplications"]:  # expert contractions are routed; the router itself is dense
            assert (m.get("execution", {}).get("kind") == "routed") == m["module"].endswith(".experts")
        assert compare(g, json.loads(Path(f"json/{golden}.json").read_text())) == []
