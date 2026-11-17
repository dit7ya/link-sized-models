const {checkPrefix} = require('../tools/host.cjs');
// Runs the exact URL payload in Node the way the loader page does: decode the base64url program
// before the first '.', eval it with d = the rest. Then (1) bits/char on the first N held-out
// chars via the program's own R()/S() and (2) a few sampled stories via its real onclick path.
// usage: node stories/verify_stories.js out/NAME_url.txt [N=20000] [samples=3] [seed=0]
const fs = require('fs');
const [file, N = '20000', NS = '3', SEED = '0'] = process.argv.slice(2);
let randomState = +SEED >>> 0;
Math.random = () => ((randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0) / 4294967296);
const url = fs.readFileSync(file, 'utf8'), frag = url.split('#')[1], dot = frag.indexOf('.');
checkPrefix(url);
const src = Buffer.from(frag.slice(0, dot).replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('latin1');
globalThis.d = frag.slice(dot + 1); globalThis.document = { body: { style: {} } };
const timer = globalThis.setTimeout;
if (+NS === 0) globalThis.setTimeout = () => {}; // suppress the auto-sample's queued cache writes
let t0 = Date.now(); (0, eval)(src); globalThis.setTimeout = timer;
console.log(`url ${url.length} chars, program ${src.length} chars, load+first steps ${Date.now() - t0} ms`);
const held = fs.readFileSync('data/heldout.u8').subarray(0, +N);
const run = () => {
  t0 = Date.now(); R(); let t = 0, bits = 0;
  for (const y of held) { const l = S(t); const mx = Math.max(...l); let z = 0; for (const v of l) z += Math.exp(v - mx); bits += (Math.log(z) + mx - l[y]) / Math.LN2; t = y; }
  console.log(`JS bits/char on first ${held.length} held-out chars: ${(bits / held.length).toFixed(5)} (${((Date.now() - t0) / held.length).toFixed(2)} ms/char)`);
  const metaFile = /_url\.txt$/.test(file) ? file.replace(/_url\.txt$/, '_meta.json') : null;
  if (held.length === 20000 && metaFile && fs.existsSync(metaFile)) {
    const expected = JSON.parse(fs.readFileSync(metaFile, 'utf8')).heldout_first20k_bpc;
    if (Math.abs(bits / held.length - expected) > 0.00002) throw Error(`Python/JS mismatch: expected ${expected}`);
    console.log('Python/JS bpc match within 0.00002');
  }
};
// let the auto-started story finish first so it does not share the cache with the measurement
let ns = 0; const samples = [];
const wait = () => { const s = document.body.innerText || ''; if (s.endsWith('_') || !s) return setTimeout(wait, 5); samples.push(s); if (samples.length < +NS) { document.body.innerText = ''; onclick(); setTimeout(wait, 5) } else { samples.forEach((s, i) => console.log(`--- sample ${i + 1} (${s.length} chars)\n${s}`)); run(); } };
if (+NS === 0) run(); else setTimeout(wait, 5);
