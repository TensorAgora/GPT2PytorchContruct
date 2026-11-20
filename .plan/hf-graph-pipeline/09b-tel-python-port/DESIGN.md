# 09b DESIGN

Port of TensorNodes `nodes/tel.js` (TensormorphDesignUI 8755b2c) to `src/gpt2_pytorch/tensorprog/tel_flat.py`:

| Python | tel.js | notes |
| --- | --- | --- |
| `tel_id(name)` | `doc.js` 93 `telOf` | `.` → `_`, `_` suffix on TEL vocabulary (`tel_vocabulary.json`, exported by `tools/tel_roundtrip.cjs --emit-flat`) |
| `_rhs(o, id, T, fix_contract)` | `tel.js` 18-38 `rhs`, plus tel_roundtrip's `--fix-contract` | contract respelling built in |
| `print_flat(graph, const_as_buffer=True, fix_contract=True) -> (source, sigma)` | 48-73 `headerLines` + `printFlat` | TEL's `print(parse(…))` normalization is the identity on this subset, except that it ends without a newline; numbers use JSON.stringify spelling |
| `build(source, sigma) -> graph` | 152-200 `decodeExpr` + `build` | a line parser for the T♭ subset (no TEL parser in Python); `contract(…, over=X, Y)` decodes several axes; edges, groups, multiplications and totals re-derived by `viz_graph()` |

Σ = `{graph: source/model/example/aliases/regions/…/additive fields, dims, tensors: {tel id: name, role, keys, view,
value}, ops: {tel id: module, module_type, region, equation, cost, multiplication, execution}}`, joined by output symbol
as `doc.sigma` does.

Checked: Python text byte-identical to `tel_roundtrip.cjs --emit-flat` on the 8 goldens, round trip 0 differences with
equal totals, and tel.js's compile/D3 on that text 8/8.
