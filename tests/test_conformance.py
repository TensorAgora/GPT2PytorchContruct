"""Every registered descriptor passes C1-C6 offline (tools/conformance.py)."""
import pytest

pytest.importorskip("gguf")
from gpt2_pytorch.registry.descriptor import ids  # noqa: E402
from tools.conformance import check  # noqa: E402


@pytest.mark.parametrize("descriptor", ids())
def test_conformance(descriptor):
    r = check(descriptor)
    assert not [f"{k} {v}" for k, v in r.items() if v.startswith("FAIL")]
