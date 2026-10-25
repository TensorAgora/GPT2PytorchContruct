// node tools/tel_compile.mjs <generated.tel> <reference.tel>
// Compiles both files with TEL's own compiler (TEL_DIST, default: the sibling tensormorph-source checkout) and fails
// unless the generated one compiles cleanly and has the same structural ops as the reference.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const dist = process.env.TEL_DIST ?? fileURLToPath(new URL(
  '../../tensormorph-source/tensorexpressionlanguage/tensorexpressionlanguage/packages/tel/dist', import.meta.url));
const { compile } = await import(`${dist}/index.js`);
const STRUCTURAL = ['tm.core.Contraction', 'tm.core.Gather', 'tm.core.Slice', 'tm.core.Reshape', 'tm.core.Permute',
  'tm.core.Rename', 'tm.math.layernorm', 'tm.math.softmax'];

function run(path) {
  const c = compile(readFileSync(path, 'utf8'));
  const counts = {};
  for (const op of c.ttir.ops) counts[`${op.level}.${op.name}`] = (counts[`${op.level}.${op.name}`] ?? 0) + 1;
  return { ok: c.ok, errors: c.diagnostics.errors.map(d => `${d.code} ${d.message}`), ops: c.ttir.ops.length, counts };
}

const [generated, reference] = process.argv.slice(2).map(run);
console.log(JSON.stringify({ generated, reference }, null, 1));
const diff = STRUCTURAL.filter(k => generated.counts[k] !== reference.counts[k]);
if (!generated.ok || diff.length) {
  console.error(`FAIL: ok=${generated.ok}, structural differences: ${diff.join(', ') || 'none'}`);
  process.exit(1);
}
console.log(`PASS: compiles, ${STRUCTURAL.map(k => `${k.split('.').pop()} ${generated.counts[k]}`).join(', ')} equal the reference`);
