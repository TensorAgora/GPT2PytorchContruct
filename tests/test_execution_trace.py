import json

import torch

from gpt2_pytorch.tracing import build_trace, write_trace


def test_trace_structure_and_names(tiny_model, tiny_ids):
    trace = build_trace(tiny_model, tiny_ids)
    assert (trace["format"], trace["version"]) == ("tensormorph-pytorch-trace", 1)
    assert set(trace) == {"format", "version", "model", "inputs", "parameters", "buffers", "events", "outputs"}
    assert trace["inputs"] == [{"name": "input.input_ids", "shape": [2, 6], "dtype": "int64", "device": "cpu", "numel": 12}]
    assert [o["name"] for o in trace["outputs"]] == ["model.final_hidden", "model.logits"]
    assert trace["outputs"][1]["shape"] == [2, 6, 100]
    assert [e["index"] for e in trace["events"]] == list(range(len(trace["events"])))

    qkv = next(e for e in trace["events"] if e["module"] == "transformer.blocks.0.attention.qkv_projection")
    assert qkv["module_type"] == "Linear" and qkv["input_shapes"] == [[2, 6, 32]] and qkv["output_shapes"] == [[2, 6, 96]]
    assert (qkv["dtype"], qkv["device"]) == ("float32", "cpu")
    assert qkv["inputs"][0]["name"] == "block.0.attention.qkv_projection.input"
    assert qkv["parameters"] == ["transformer.blocks.0.attention.qkv_projection.weight", "transformer.blocks.0.attention.qkv_projection.bias"]

    attention = next(e for e in trace["events"] if e["module"] == "transformer.blocks.0.attention")
    assert [t["name"] for t in attention["internals"]][:5] == [f"block.0.attention.{n}" for n in ("qkv", "q", "k", "v", "scores")]
    assert [t["shape"] for t in attention["internals"]][4] == [2, 4, 6, 6]


def test_parameters_and_buffers_are_separate_and_tying_is_visible(tiny_model, tiny_ids):
    trace = build_trace(tiny_model, tiny_ids)
    params = {p["name"]: p for p in trace["parameters"]}
    wte, head = params["transformer.embeddings.token_embedding.weight"], params["lm_head.weight"]
    assert wte["storage_id"] == head["storage_id"] and head["tied_with"] == [wte["name"]] and wte["tied_with"] == ["lm_head.weight"]
    assert all(p["kind"] == "parameter" and p["requires_grad"] for p in params.values())
    assert {b["name"] for b in trace["buffers"]} >= {"transformer.blocks.0.attention.causal_mask", "transformer.embeddings.position_ids"}
    assert all(b["kind"] == "buffer" and not b["requires_grad"] for b in trace["buffers"])
    assert all(p["storage_ptr"] is None for p in trace["parameters"] + trace["buffers"])  # no memory addresses in the file


def test_values_are_excluded_by_default_and_capped_when_requested(tiny_model, tiny_ids):
    assert "values" not in json.dumps(build_trace(tiny_model, tiny_ids))
    trace = build_trace(tiny_model, tiny_ids, max_values=12)
    assert trace["inputs"][0]["values"] == tiny_ids.flatten().tolist()
    sized = [t for e in trace["events"] for t in e["inputs"] + e["outputs"] + e["internals"]]
    assert all(("values" in t) == (t["numel"] <= 12) for t in sized)


def test_trace_is_json_serializable_and_restores_training_mode(tiny_model, tiny_ids, tmp_path):
    tiny_model.train()
    trace = build_trace(tiny_model, tiny_ids)
    assert tiny_model.training
    path = write_trace(trace, tmp_path / "sub" / "t.json")
    assert json.loads(path.read_text()) == trace


def test_trace_is_deterministic(tiny_model, tiny_ids):
    assert build_trace(tiny_model, tiny_ids) == build_trace(tiny_model, tiny_ids)
