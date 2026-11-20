import logging

import pytest

from gpt2_pytorch.hub.http import Client, HttpError, NotAllowed, Response, allowed


class Fake:
    """Transport that answers from a list of Responses and records what it was sent."""

    def __init__(self, *responses):
        self.responses, self.sent = list(responses), []

    def __call__(self, method, url, headers):
        self.sent.append((url, headers))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def client(tmp_path, fake, **kw):
    slept = []
    return Client(cache_dir=tmp_path, transport=fake, sleep=slept.append, **kw), slept


def test_429_retry_after_then_ok(tmp_path):
    fake = Fake(Response(429, {"retry-after": "2"}, b""), Response(206, {}, b"12345678"))
    c, slept = client(tmp_path, fake)
    assert c.get_range("https://x/f", 0, 7) == b"12345678"
    assert slept == [2.0] and c.requests == 2 and c.bytes_read == 8
    assert fake.sent[-1][1]["range"] == "bytes=0-7"


def test_503_exponential_backoff_then_gives_up(tmp_path):
    c, slept = client(tmp_path, Fake(Response(503, {}, b"busy")), max_retries=3)
    with pytest.raises(HttpError) as e:
        c.get("https://x/f")
    assert e.value.status == 503 and slept == [1.0, 2.0, 4.0]


def test_range_overrun_is_an_error(tmp_path):
    c, _ = client(tmp_path, Fake(Response(200, {}, b"x" * 100)))
    with pytest.raises(HttpError):
        c.get_range("https://x/f", 0, 7)  # a server ignoring Range would send payload bytes


def test_small_file_cached(tmp_path):
    fake = Fake(Response(200, {}, b'{"a": 1}'))
    c, _ = client(tmp_path, fake)
    assert c.get_small("o/r", "abc", "config.json") == b'{"a": 1}'
    before = c.requests
    assert c.get_small("o/r", "abc", "config.json") == b'{"a": 1}'
    assert c.requests == before == 1


def test_allowlist(tmp_path):
    c, _ = client(tmp_path, Fake(Response(200, {}, b"")))
    for bad in ("model.safetensors", "modeling_x.py", "a/b/config.json", "unet/model_index.json", "../config.json"):
        assert not allowed(bad)
        with pytest.raises(NotAllowed):
            c.get_small("o/r", "abc", bad)
    for ok in ("config.json", "unet/config.json", "model.safetensors.index.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json", "model_index.json", "params.json"):
        assert allowed(ok)
    assert c.requests == 0


def test_token_sent_but_never_logged(tmp_path, monkeypatch, caplog):
    secret = "hf_SECRETtoken123"
    monkeypatch.setenv("HF_TOKEN", secret)
    fake = Fake(Response(500, {}, f"echo {secret}".encode()))
    c, _ = client(tmp_path, fake, max_retries=2)
    with caplog.at_level(logging.DEBUG, logger="gpt2_pytorch.hub"), pytest.raises(HttpError) as e:
        c.get(f"https://x/f?token={secret}")
    assert fake.sent[0][1]["authorization"] == f"Bearer {secret}"
    assert secret not in str(e.value) and secret not in caplog.text
    assert "***" in str(e.value)
