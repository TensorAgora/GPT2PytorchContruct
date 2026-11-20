// node tools/tel_roundtrip.cjs <viz-graph.json | dir>... [--const-as-buffer] [--fix-contract]
// One JSON line per graph (a directory means every *.json in it), then a {summary} line.
// viz-graph -> T♭ (flat TEL, TensorNodes nodes/tel.js printFlat) -> graph (build/decodeExpr), every field TEL rebuilds
// compared with the original and grouped by kind; then TEL compiles T♭ and D3 compares tm.core counts with the graph's.
//   --const-as-buffer  tensor-valued constants (masks, rope tables) become declared buffers, as if bound by tensor:// URI
//   --fix-contract     a contraction printed with `@` whose right operand has a batch axis (all but its last two) that is not
//                      a batch axis of the left (e.g. N2[B,T,D] @ Wge[E,Fe,D].T, Hh[B,T,E,Fe] @ Wde[E,D,Fe].T) is respelled
//                      contract(a, b, over=X)
//   --emit-flat DIR    also write each graph's T♭ (after the fixes) to DIR/<name>.tel, and the TEL vocabulary to DIR/vocabulary.json
// TN_NODES overrides the path to TensormorphDesignUI/tensormorph/ui/activity_bar/nodes.
const NODES = process.env.TN_NODES ?? require('node:path').resolve(__dirname, '../../TensormorphDesignUI/tensormorph/ui/activity_bar/nodes');
const args = process.argv.slice(2), flag = (f) => args.includes(f);
const fs = require('node:fs'), np = require('node:path');
globalThis.window = globalThis;
require(NODES + '/core.js'); require(NODES + '/registry.js'); require(NODES + '/doc.js');
require(NODES + '/../vendor/tm-interface.js'); require(NODES + '/tel.js');
const N = globalThis.TM.nodes, tel = globalThis.TMI.tel;
function one(path) {
const J = JSON.parse(fs.readFileSync(path, 'utf8'));
const listConsts = J.tensors.filter((t) => t.role === 'constant' && Array.isArray(t.value)).map((t) => t.name);
if (flag('--const-as-buffer')) for (const t of J.tensors) if (listConsts.includes(t.name)) { t.role = 'buffer'; t.value = null; }
const doc = N.doc.raise(J, { glyphs: (ids) => new Map(ids.map((x) => [x, x])) });
const T = new Map(doc.json.tensors.map((t) => [t.name, t])), S = N.doc.sigma(doc);
const isT = (o) => { if (o.primitive !== 'Contraction') return false; const A = T.get(o.inputs[0]).axes, B = T.get(o.inputs[1]).axes, ov = o.attrs.over || []; return ov.length === 1 && A[A.length - 1] === ov[0] && B[B.length - 1] === ov[0] && !(B.length === 2 && B[0] === ov[0]); };
const printedWithAt = (o) => { const A = T.get(o.inputs[0]).axes, ov = o.attrs.over || []; return ov.length === 1 && A[A.length - 1] === ov[0]; };
let flat = N.tel.printFlat(doc);
const emit = () => { if (emitDir) fs.writeFileSync(np.join(emitDir, path.split('/').pop().replace(/\.json(\.gz)?$/, '') + '.tel'), flat); };
const respelled = new Set();
if (flag('--fix-contract')) {
  for (const o of doc.json.ops) {
    if (o.primitive !== 'Contraction' || !printedWithAt(o)) continue;
    const A = T.get(o.inputs[0]).axes, B = T.get(o.inputs[1]).axes;
    if (!B.slice(0, -2).some((x) => !A.slice(0, -2).includes(x))) continue; // `@` batches over all but the last two axes: B's must be A's
    const out = S.get(o.out).tel, re = new RegExp('^' + out.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ' = .*$', 'm');
    flat = flat.replace(re, out + ' = contract(' + S.get(o.inputs[0]).tel + ', ' + S.get(o.inputs[1]).tel + ', over=' + o.attrs.over.join(', ') + ')');
    respelled.add(o.out);
  }
}
emit();
// graph from text, compared field by field
const b = N.tel.build(flat, doc), byOut = new Map(b.ops.map((o) => [o.out, o]));
const diffs = {}, ex = {};
const note = (k, e) => { diffs[k] = (diffs[k] || 0) + 1; if (!ex[k]) ex[k] = JSON.stringify(e).slice(0, 220); };
const same = (x, y) => JSON.stringify(x ?? null) === JSON.stringify(y ?? null);
for (const o of doc.json.ops) {
  const d = byOut.get(o.out);
  if (!d) { note('op missing from TEL', o.out); continue; }
  if (d.primitive !== o.primitive) note('primitive', { out: o.out, want: o.primitive, got: d.primitive });
  if ((d.fn ?? null) !== (o.fn ?? null)) note('fn', { out: o.out, want: o.fn, got: d.fn });
  if (!same(d.inputs, o.inputs)) note('inputs', { out: o.out, want: o.inputs, got: d.inputs });
  for (const k of new Set([...Object.keys(o.attrs || {}), ...Object.keys(d.attrs || {})]))
    if (!same((o.attrs || {})[k], (d.attrs || {})[k])) note('attrs.' + o.primitive + '.' + k, { out: o.out, want: (o.attrs || {})[k], got: (d.attrs || {})[k] });
}
for (const t of doc.json.tensors) {
  const d = b.decls.get(t.name);
  if (!d) { note('tensor not declared', t.name); continue; }
  if (d.dtype !== t.dtype) note('dtype', { t: t.name, want: t.dtype, got: d.dtype });
  if (!same(d.axes, t.axes)) note('axes', { t: t.name, want: t.axes, got: d.axes });
  if (t.role === 'constant') {
    const v = b.consts.get(t.name);
    if (Array.isArray(t.value)) note('tensor-valued constant (no TEL literal)', { t: t.name, shape: t.shape, got: v });
    else if (Number(v) !== Number(t.value)) note('scalar constant', { t: t.name, want: t.value, got: v });
  }
}
const extra = b.ops.filter((o) => !doc.json.ops.some((x) => x.out === o.out)).length;
if (extra) note('extra ops in TEL', extra);
// compile + D3 (tel.js verify): tm.core counts equal the graph's; each `.T` adds a Permute; Broadcast is ignored
const c = tel.compile(flat), errs = c.diagnostics.errors || [], codes = {};
for (const e of errs) codes[e.code] = (codes[e.code] || 0) + 1;
const got = {}, want = {};
for (const o of c.ttir.ops) got[o.name] = (got[o.name] || 0) + 1;
for (const o of doc.json.ops) want[o.primitive] = (want[o.primitive] || 0) + 1;
want.Permute = (want.Permute || 0) + doc.json.ops.filter((o) => isT(o) && !respelled.has(o.out)).length;
want.Constant = doc.json.tensors.filter((t) => t.role === 'constant').length;
const keys = ['Elementwise', 'Reduce', 'Contraction', 'Gather', 'Slice', 'Reshape', 'Rename', 'Permute', 'Constant'];
const d3bad = keys.filter((k) => (got[k] || 0) !== (want[k] || 0)).map((k) => k + ' got ' + (got[k] || 0) + ' want ' + (want[k] || 0));
const carried = { module: 0, module_type: 0, region: 0, equation: 0, checkpoint_key: 0, view: 0, pytorch_name: 0 };
for (const o of J.ops) for (const k of ['module', 'module_type', 'region', 'equation']) if (o[k] != null) carried[k]++;
for (const t of J.tensors) for (const k of ['checkpoint_key', 'view', 'pytorch_name']) if (t[k] != null) carried[k]++;
return ({ name: path.split('/').pop(), mode: args.filter((a) => a.startsWith('--')).join(' ') || 'as-is', ops: J.ops.length,
  tensors: J.tensors.length, list_constants: listConsts.length, respelled: respelled.size, flat_lines: flat.split('\n').length,
  parse_diagnostics: b.diagnostics.length, diff_total: Object.values(diffs).reduce((a, x) => a + x, 0), diffs, examples: ex,
  compile: { ok: c.ok, errors: errs.length, codes, first: errs[0] ? errs[0].code + ' ' + String(errs[0].message).slice(0, 140) : null, at: errs[0] && errs[0].span ? flat.split('\n')[flat.slice(0, errs[0].span.start).split('\n').length - 1] : null },
  d3: { pass: c.ok && !errs.length && !d3bad.length, mismatches: d3bad }, carried_by_sigma_not_tel: carried });
}

const emitAt = args.indexOf('--emit-flat'), emitDir = emitAt >= 0 ? args[emitAt + 1] : null;
const paths = args.filter((a, i) => !a.startsWith('--') && !(emitAt >= 0 && i === emitAt + 1)).flatMap((p) => fs.statSync(p).isDirectory()
  ? fs.readdirSync(p).filter((f) => f.endsWith('.json')).sort().map((f) => np.join(p, f)) : [p]);
if (emitDir) { fs.mkdirSync(emitDir, { recursive: true }); fs.writeFileSync(np.join(emitDir, 'vocabulary.json'), JSON.stringify([...new Set([].concat(tel.TEL_VOCABULARY || [], tel.KEYWORDS || ['axis', 'indices', 'where', 'new']))].sort())); }
const rows = [];
for (const p of paths) {
  let r;
  try { r = one(p); } catch (e) { r = { name: p.split('/').pop(), error: String(e.message || e).slice(0, 200) }; }
  rows.push(r);
  console.log(JSON.stringify(r));
}
const ok = (r) => !r.error;
console.log(JSON.stringify({ summary: { graphs: rows.length, errors: rows.filter((r) => !ok(r)).length,
  zero_diff: rows.filter((r) => ok(r) && r.diff_total === 0).length, compile_ok: rows.filter((r) => ok(r) && r.compile.ok && !r.compile.errors).length,
  d3_pass: rows.filter((r) => ok(r) && r.d3.pass).length, mode: args.filter((a) => a.startsWith('--')).join(' ') || 'as-is' } }));
