"""One rate-limited, token-safe HTTP client for the Hub (master §3, §5.1): bounded concurrency, exponential backoff on
429/5xx honouring Retry-After, byte-counted range reads, and an on-disk cache of the few small files §3 allows.

HF_TOKEN is read from the environment only (never from a stored login), sent as the Authorization header and replaced by
*** in every message this module produces. Do not set HF_DEBUG: huggingface_hub then logs full requests.
"""
import copy
import email.utils
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

ENDPOINT = "https://huggingface.co"
USER_AGENT = "tensormorph-hf-graph/0.1"
log = logging.getLogger("gpt2_pytorch.hub")


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def redact(text: str) -> str:
    t = token()
    return text.replace(t, "***") if t and text else text


class _Redact(logging.Filter):
    def filter(self, record):
        record.msg, record.args = redact(record.getMessage()), ()
        return True


log.addFilter(_Redact())


class HttpError(Exception):
    def __init__(self, status: int, url: str, detail: str = ""):
        self.status, self.url = status, redact(url)
        super().__init__(redact(f"HTTP {status} {url} {detail}".strip()))


class NotAllowed(ValueError):
    """A file outside the §3 allowlist was requested."""


def allowed(path: str) -> bool:
    """§3: config.json (root or one component folder), *.safetensors.index.json, model_index.json, params.json."""
    parts = path.split("/")
    name, depth = parts[-1], len(parts) - 1
    if depth > 1 or ".." in parts:
        return False
    return (name == "config.json" or name.endswith(".safetensors.index.json")
            or (depth == 0 and name in ("model_index.json", "params.json")))


@dataclass
class Response:
    status: int
    headers: dict  # lower-case keys
    body: bytes


def _default_transport(method: str, url: str, headers: dict) -> Response:
    from huggingface_hub import get_session
    r = get_session().request(method, url, headers=headers, follow_redirects=True, timeout=60)
    return Response(r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content)


def _retry_after(headers: dict) -> float | None:
    v = headers.get("retry-after")
    if not v:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", v.strip()):
        return float(v)
    try:
        return max(0.0, email.utils.parsedate_to_datetime(v).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


class Client:
    def __init__(self, concurrency=4, cache_dir=None, transport=None, max_retries=6, sleep=time.sleep):
        self.cache_dir = Path(cache_dir or os.environ.get("TM_HF_CACHE") or Path.home() / ".cache/tensormorph/hf")
        self.transport = transport or _default_transport
        self.max_retries, self.sleep = max_retries, sleep
        self._slots = threading.BoundedSemaphore(concurrency)
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes_read = 0

    def scoped(self) -> "Client":
        """A view with its own request/byte counters that shares the semaphore, cache and transport (one per job)."""
        view = copy.copy(self)
        view.requests = view.bytes_read = 0
        return view

    def _headers(self, extra=None) -> dict:
        h = {"user-agent": USER_AGENT}
        if token():
            h["authorization"] = f"Bearer {token()}"
        return {**h, **(extra or {})}

    def get(self, url: str, headers=None) -> Response:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            with self._slots:
                try:
                    r = self.transport("GET", url, self._headers(headers))
                except Exception as e:  # network error: retry like a 5xx
                    r = Response(0, {}, redact(f"{type(e).__name__}: {e}").encode())
            with self._lock:
                self.requests += 1
            if r.status not in (0, 429) and r.status < 500:
                return r
            if attempt == self.max_retries:
                raise HttpError(r.status, url, r.body[:200].decode(errors="replace"))
            wait = min(_retry_after(r.headers) or delay, 60.0)
            log.warning("HTTP %s on %s, retry %d in %.1fs", r.status, redact(url), attempt + 1, wait)
            self.sleep(wait)
            delay = min(delay * 2, 60.0)

    def get_range(self, url: str, start: int, end: int) -> bytes:
        r = self.get(url, {"range": f"bytes={start}-{end}"})
        want = end - start + 1
        if r.status not in (200, 206) or len(r.body) > want:
            raise HttpError(r.status, url, f"range {start}-{end} answered with {len(r.body)} bytes")
        with self._lock:
            self.bytes_read += len(r.body)
        return r.body

    def file_url(self, repo: str, sha: str, path: str) -> str:
        return f"{ENDPOINT}/{repo}/resolve/{sha}/{quote(path)}"

    def get_small(self, repo: str, sha: str, path: str) -> bytes:
        if not allowed(path):
            raise NotAllowed(f"{path!r} is not a §3 small file")
        cached = self.cache_dir / repo / sha / path
        if cached.exists():
            return cached.read_bytes()
        r = self.get(self.file_url(repo, sha, path))
        if r.status != 200:
            raise HttpError(r.status, self.file_url(repo, sha, path))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(r.body)
        return r.body

    def model_info(self, repo: str, revision: str | None = None, refresh: bool = False) -> dict:
        """The JSON HfApi.model_info(files_metadata=True) reads (/api/models/<repo>[/revision/<rev>]?blobs=true), cached."""
        cached = self.cache_dir / repo / "_info" / f"{quote(revision or 'main', safe='')}.json"
        if cached.exists() and not refresh:
            return json.loads(cached.read_text())
        url = f"{ENDPOINT}/api/models/{repo}" + (f"/revision/{quote(revision, safe='')}" if revision else "") + "?blobs=true"
        r = self.get(url)
        if r.status != 200:
            raise HttpError(r.status, url)
        info = json.loads(r.body)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(info, sort_keys=True))
        return info
