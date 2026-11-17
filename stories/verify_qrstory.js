// Run the exact QR story URL as site/index.html does, check weights, forward parity, print samples.
// Usage: node stories/verify_qrstory.js out/qrstory_url.txt [fixture.json]
const fs = require('fs');
const {checkPrefix} = require('../tools/host.cjs');
const url = fs.readFileSync(process.argv[2], 'utf8').trim();
checkPrefix(url);
const frag = decodeURIComponent(url.slice(url.indexOf('#') + 1));
const n = +frag.slice(0, 3), src = frag.slice(3, n + 3);
if (src.length !== n || !/^\d+$/.test(frag.slice(n + 3))) throw Error('Bad raw payload');
window = globalThis; window.d = frag.slice(n + 3);
(0, eval)(src);
if (typeof g !== 'function') throw Error('No g()');
const out = {};
if (process.argv[3]) {
  const fx = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  let integer = BigInt(d), radix = BigInt(fx.levels), M = (fx.levels - 1) / 2;
  for (let i = fx.weights.length; i--;) {
    const q = integer / radix;
    if (Number(integer - q * radix) - M !== fx.weights[i] || W[i] !== fx.weights[i]) throw Error('Weight mismatch ' + i);
    integer = q;
  }
  if (integer !== 1n) throw Error('Missing radix marker');
  let maxError = 0, comparisons = 0, bits = 0, count = 0;
  const realRandom = Math.random;
  for (let s = 0; s < fx.seqs.length; s++) {
    let cur = 0; const seq = fx.seqs[s];
    Math.random = () => {
      const lp = globalThis.p.map(v => Math.log(v / globalThis.t));
      for (let j = 0; j < lp.length; j++) {
        if (!Number.isFinite(lp[j])) throw Error('Invalid probability');
        maxError = Math.max(maxError, Math.abs(lp[j] - fx.logprobs[s][cur][j])); comparisons++;
      }
      const T = fx.temperature, l1 = lp.map(v => v * T), m = Math.max(...l1);
      const z = Math.log(l1.reduce((a, v) => a + Math.exp(v - m), 0)) + m;
      const y = seq[cur]; bits -= (l1[y] - z) / Math.LN2; count++;
      const before = globalThis.p.slice(0, y).reduce((a, b) => a + b, 0);
      cur++;
      return (before + globalThis.p[y] / 2) / globalThis.t;
    };
    const got = g();
    const want = fx.seqs[s].map(j => fx.alphabet[j]).join('');
    if (got.toLowerCase() !== want || cur !== seq.length) throw Error('Forced-sequence mismatch ' + s);
  }
  Math.random = realRandom;
  Object.assign(out, {max_logprob_error: maxError, probability_comparisons: comparisons,
    forced_sequences: fx.seqs.length, node_bpc_T1_on_fixture: bits / count});
}
let seed = 7;
Math.random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
out.sample_seed = 7;
out.samples = Array.from({length: 5}, () => g());
console.log(JSON.stringify(out));
