import copy
import json
from functools import cache
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
from tools.graph_check import check, compare  # noqa: E402

GOLDENS = sorted(Path("json").glob("*.json"))


@cache
def load(name: str) -> dict:
    return json.loads(Path("json", name).read_text())


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_goldens_pass(path):
    assert check(load(path.name)) == []


def test_dangling_input_fails_naming_the_op():
    g = copy.deepcopy(load("distilgpt2.json"))
    op = g["ops"][5]
    op["inputs"][0] = g["ops"][7]["out"]  # produced later: misorder / cycle
    errs = check(g)
    assert any(e.startswith("op 5: input") for e in errs), errs


def test_totals_recompute():
    g = copy.deepcopy(load("distilgpt2.json"))
    g["totals"]["contraction_flops"] += 1
    assert any("totals.contraction_flops" in e for e in check(g))


def test_manifest_shape_mismatch():
    g = load("distilgpt2.json")
    p = next(t for t in g["tensors"] if t["role"] == "parameter" and t["checkpoint_key"])
    manifest = {"tensors": [{"key": t["checkpoint_key"], "shape": t["shape"]} for t in g["tensors"] if t["checkpoint_key"]]}
    assert check(g, manifest) == []
    next(t for t in manifest["tensors"] if t["key"] == p["checkpoint_key"])["shape"] = [1]
    assert any(p["checkpoint_key"] in e for e in check(g, manifest))


def test_against():
    m = load("mistral-7b-instruct-v0.3.json")
    assert compare(m, m) == []
    diffs = compare(m, load("qwen2.5-7b-instruct.json"))
    assert any(d.startswith("dims.D:") for d in diffs), diffs


def test_against_ignores_additive_and_listed_fields():
    g = copy.deepcopy(load("distilgpt2.json"))
    g["totals"]["active_flops"] = 1
    p = next(t for t in g["tensors"] if t["role"] == "parameter")
    p["view"] = "changed"
    assert compare(g, load("distilgpt2.json")) == [f"parameter {p['name']}.view: 'changed' != golden None"]
    assert compare(g, load("distilgpt2.json"), ignore=("view",)) == []
