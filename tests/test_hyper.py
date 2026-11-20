from functools import cache

import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.hub.fixtures import FixtureClient, find  # noqa: E402
from gpt2_pytorch.hub.manifest import read_manifest  # noqa: E402
from gpt2_pytorch.registry.descriptor import load  # noqa: E402
from gpt2_pytorch.registry.hyper import parse_alternatives, resolve_hyper  # noqa: E402


@cache
def fixture(repo):
    c = FixtureClient(find(repo))
    return read_manifest(c.job(), c, fetched_at="t")[1]


def test_spec_prose_parses_to_names_only():
    assert parse_alternatives("hidden_size | n_embd (GPT-2)") == ["hidden_size", "n_embd"]
    assert parse_alternatives("<arch>.expert_feed_forward_length (else <arch>.feed_forward_length)") == \
        ["<arch>.expert_feed_forward_length", "<arch>.feed_forward_length"]
    assert parse_alternatives("<arch>.vocab_size | len(tokenizer.ggml.tokens)") == ["<arch>.vocab_size", "len(tokenizer.ggml.tokens)"]
    assert parse_alternatives("Hq * Dh") == []


@pytest.mark.skipif(find("Qwen/Qwen2.5-7B-Instruct") is None, reason="fixture missing")
def test_qwen25_config_sources():
    h = resolve_hyper(load("qwen2"), fixture("Qwen/Qwen2.5-7B-Instruct"))
    assert h["D"] == (3584, "config:hidden_size") and h["Hkv"] == (4, "config:num_key_value_heads")
    assert h["Dh"] == (128, "default:D // Hq")


@pytest.mark.skipif(find("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF") is None, reason="fixture missing")
def test_mixtral_gguf_sources():
    h = resolve_hyper(load("mixtral"), fixture("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF"))
    assert h["D"] == (4096, "gguf:llama.embedding_length") and h["E"] == (8, "gguf:llama.expert_count")
    assert h["K"] == (2, "gguf:llama.expert_used_count") and h["tie"] == (False, "missing:gguf:output.weight")
