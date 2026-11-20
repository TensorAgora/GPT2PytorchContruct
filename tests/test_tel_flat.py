"""T-2: the Python T♭ printer/decoder against TensorNodes nodes/tel.js on the 8 goldens (master §9.6)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gpt2_pytorch.tensorprog.tel_flat import build, print_flat

NODES = Path(__file__).resolve().parents[2] / "TensormorphDesignUI/tensormorph/ui/activity_bar/nodes/tel.js"
GOLDENS = sorted(Path("json").glob("*.json"))
needs_tel = pytest.mark.skipif(not (shutil.which("node") and NODES.exists()), reason="node or TensormorphDesignUI missing")


@pytest.fixture(scope="module")
def teljs(tmp_path_factory):
    out = tmp_path_factory.mktemp("flat")
    r = subprocess.run(["node", "tools/tel_roundtrip.cjs", *map(str, GOLDENS), "--const-as-buffer", "--fix-contract",
                        "--emit-flat", str(out)], capture_output=True, text=True, timeout=900)
    return out, json.loads(r.stdout.strip().splitlines()[-1])["summary"]


@needs_tel
@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_print_equals_teljs(path, teljs):
    assert print_flat(json.loads(path.read_text()))[0] == (teljs[0] / f"{path.stem}.tel").read_text()


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_roundtrip(path):
    g = json.loads(path.read_text())
    h = build(*print_flat(g))
    T, O = {t["name"]: t for t in h["tensors"]}, {o["out"]: o for o in h["ops"]}
    assert len(h["ops"]) == len(g["ops"])
    assert all(O[o["out"]][k] == o[k] for o in g["ops"] for k in ("primitive", "fn", "inputs", "attrs"))
    assert all(T[t["name"]]["dtype"] == t["dtype"] and T[t["name"]]["axes"] == t["axes"] for t in g["tensors"])
    assert h["totals"] == g["totals"]


@needs_tel
def test_compile_and_d3(teljs):
    """The Python text is the tel.js text (test above), which TEL compiles cleanly with D3 equal on all 8."""
    assert teljs[1]["compile_ok"] == teljs[1]["d3_pass"] == teljs[1]["zero_diff"] == len(GOLDENS)


def test_multi_axis_contract_decodes():
    g = json.loads(Path("json/distilgpt2.json").read_text())
    text, sigma = print_flat(g)
    op = next(o for o in g["ops"] if o["primitive"] == "Contraction")
    line = next(ln for ln in text.splitlines() if ln.startswith(op["out"].replace(".", "_") + " = "))
    from gpt2_pytorch.tensorprog.tel_flat import _decode
    d = _decode("contract(a, b, over=X, Y)", lambda n: n, lambda n: ["X", "Y"], "o")
    assert d["attrs"]["over"] == ["X", "Y"] and line
