import torch

from gpt2_pytorch.model import CausalSelfAttention
from gpt2_pytorch.tracing import ActivationRecorder


def record(model, ids, **kw):
    with torch.no_grad(), ActivationRecorder(model, **kw) as recorder:
        logits = model(ids)
    return recorder, logits


def test_records_every_module_call_in_execution_order(tiny_model, tiny_ids):
    recorder, _ = record(tiny_model, tiny_ids)
    types = [r.module_type for r in recorder.records[:9]]
    assert types == ["DistilGPT2LMHeadModel", "GPT2Model", "GPT2Embeddings", "TokenEmbedding", "PositionEmbedding", "Dropout",
                     "TransformerBlock", "LayerNorm", "CausalSelfAttention"]
    assert [r.index for r in recorder.records] == list(range(len(recorder.records)))
    assert sorted(r.exit_index for r in recorder.records) == list(range(len(recorder.records)))
    assert recorder.records[0].exit_index == len(recorder.records) - 1  # root finishes last
    names = [r.module for r in recorder.records]
    assert names.index("transformer.blocks.0.attention.qkv_projection") < names.index("transformer.blocks.0.attention.output_projection")


def test_stats_match_the_tensor(tiny_model, tiny_ids):
    recorder, logits = record(tiny_model, tiny_ids)
    out = recorder.records[0].outputs[0]
    assert out["name"] == "model.logits" and out["shape"] == [2, 6, 100] and out["dtype"] == "float32" and out["device"] == "cpu"
    assert out["numel"] == logits.numel() and out["bytes"] == logits.numel() * 4
    assert out["min"] == logits.min().item() and out["max"] == logits.max().item()
    assert abs(out["mean"] - logits.mean().item()) < 1e-5 and abs(out["std"] - logits.std(correction=0).item()) < 1e-5
    assert recorder.records[0].inputs[0]["name"] == "input.input_ids" and recorder.records[0].inputs[0]["dtype"] == "int64"


def test_attention_internals_are_captured_without_model_changes(tiny_model, tiny_ids):
    recorder, _ = record(tiny_model, tiny_ids)
    attn = next(r for r in recorder.records if r.module == "transformer.blocks.1.attention")
    assert list(attn.internals) == [f"block.1.attention.{n}" for n in ("qkv", "q", "k", "v", "scores", "mask", "probs", "context")]
    assert attn.internals["block.1.attention.q"]["shape"] == [2, 4, 6, 8]
    assert attn.internals["block.1.attention.probs"]["shape"] == [2, 4, 6, 6]


def test_hooks_and_sinks_are_removed_on_exit(tiny_model, tiny_ids):
    record(tiny_model, tiny_ids)
    for module in tiny_model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
        if isinstance(module, CausalSelfAttention):
            assert module.debug_sink is None


def test_recording_does_not_change_outputs_and_keep_tensors_is_opt_in(tiny_model, tiny_ids):
    with torch.no_grad():
        plain = tiny_model(tiny_ids)
    recorder, logits = record(tiny_model, tiny_ids)
    assert torch.equal(plain, logits) and all(not r.tensors for r in recorder.records)
    kept, _ = record(tiny_model, tiny_ids, keep_tensors=True)
    assert torch.equal(kept.records[0].tensors["model.logits"], plain)


def test_str_is_one_line(tiny_model, tiny_ids):
    recorder, _ = record(tiny_model, tiny_ids)
    assert "\n" not in str(recorder.records[3]) and "TokenEmbedding" in str(recorder.records[3])
