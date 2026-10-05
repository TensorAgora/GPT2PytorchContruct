import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from conftest import needs_checkpoint
from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint, load_pretrained, load_state_dict_file
from gpt2_pytorch.checkpoint.safetensors_loader import convert_to_safetensors, read_safetensors_header
from gpt2_pytorch.data import FIXED_INPUT_IDS


def test_tied_tensors_are_written_once(tmp_path):
    shared = torch.randn(4, 3)
    state = {"a": shared, "b": torch.randn(2), "alias": shared}
    dropped = convert_to_safetensors(state, tmp_path / "t.safetensors")
    assert dropped == {"alias": "a"}
    assert set(load_file(tmp_path / "t.safetensors")) == {"a", "b"}
    assert json.loads(read_safetensors_header(tmp_path / "t.safetensors")["__metadata__"]["tied"]) == dropped


def test_header_parsing_reads_no_tensor_data(tmp_path):
    convert_to_safetensors({"x": torch.zeros(3, 5), "y": torch.ones(7, dtype=torch.long)}, tmp_path / "t.safetensors")
    header = read_safetensors_header(tmp_path / "t.safetensors")
    assert header["x"]["shape"] == [3, 5] and header["x"]["dtype"] == "F32"
    assert header["y"]["dtype"] == "I64"
    spans = sorted(v["data_offsets"] for k, v in header.items() if k in "xy")  # safetensors orders by dtype size, not name
    assert spans[0][0] == 0 and spans[0][1] == spans[1][0] and spans[1][1] == 60 + 56
    assert header["x"]["data_offsets"][1] - header["x"]["data_offsets"][0] == 60


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    path = tmp_path_factory.mktemp("st") / "model.safetensors"
    dropped = convert_to_safetensors(load_state_dict_file(DEFAULT_CHECKPOINT), path, source="pytorch_model.bin")
    return path, dropped


@needs_checkpoint
def test_conversion_keeps_every_value(converted):
    path, dropped = converted
    original, reloaded = load_state_dict_file(DEFAULT_CHECKPOINT), load_file(path)
    assert dropped == {"lm_head.weight": "transformer.wte.weight"}
    assert reloaded.keys() == original.keys() - {"lm_head.weight"} and len(reloaded) == 82
    assert all(torch.equal(reloaded[k], original[k]) for k in reloaded)


@needs_checkpoint
def test_bin_and_safetensors_logits_are_identical(converted, real_model):
    model_b, _, report = load_pretrained(converted[0])  # lm_head.weight absent -> reconstructed by tying
    assert report.tied == ["lm_head.weight -> transformer.embeddings.token_embedding.weight (reconstructed)"]
    assert model_b.lm_head.weight is model_b.transformer.token_embedding.weight
    with torch.no_grad():
        assert torch.equal(real_model(FIXED_INPUT_IDS), model_b(FIXED_INPUT_IDS))


@needs_checkpoint
def test_unified_load_checkpoint_accepts_both_formats(converted, real_model):
    from gpt2_pytorch import DistilGPT2LMHeadModel

    for path in (DEFAULT_CHECKPOINT, converted[0]):
        model = DistilGPT2LMHeadModel(real_model.config).eval()
        report = load_checkpoint(model, path)
        assert report.missing == [] and report.unexpected == []
        assert torch.equal(model.lm_head.weight, real_model.lm_head.weight)


@needs_checkpoint
def test_matches_huggingface_safetensors_if_present(converted):
    hf = DEFAULT_CHECKPOINT.parent / "model.safetensors"
    if not hf.exists():
        pytest.skip("weights/model.safetensors not present")
    ours, theirs = load_file(converted[0]), load_file(hf)
    assert ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)


@needs_checkpoint
def test_inspect_cli_prints_offsets(converted):
    out = subprocess.run([sys.executable, "-m", "tools.inspect_safetensors", str(converted[0])], capture_output=True, text=True, check=True).stdout
    rows = [line.split()[0] for line in out.splitlines() if line.startswith(("transformer.", "lm_head"))]
    assert "transformer.wte.weight" in rows and len(rows) == 82 and "82 tensors" in out and "lm_head.weight" not in rows
