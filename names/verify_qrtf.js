// Execute the exact raw URL with the loader's indirect non-strict eval.
// All 2,521 validation positions, all vocabulary probabilities; no surrogate
// implementation of the transformer's forward pass is used in this verifier.
// Usage: node verify_qrtf.js URL.txt fixture.json [sample_count=2000] [readable.js]
const fs = require('fs');
const {checkPrefix} = require('../tools/host.cjs');
const url = fs.readFileSync(process.argv[2], 'utf8').trim();
checkPrefix(url);
const meta = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const frag = decodeURIComponent(url.slice(url.indexOf('#') + 1));
const n = +frag.slice(0, 3); let src = frag.slice(3,n+3);
if (src.length !== n || !/^\d+$/.test(frag.slice(n + 3))) throw Error('Bad raw payload');
globalThis.d = frag.slice(n + 3);
if(process.argv[5])src=fs.readFileSync(process.argv[5],'utf8');
globalThis.document = {body:{}};
(0, eval)(src);
if (typeof g !== 'function' || (!meta.runtime_legacy && typeof f !== 'function')) throw Error('No g()/f()');
let integer = BigInt(d), radix = BigInt(meta.config.levels);
for (let i = meta.weights.length; i--;) {
  const quotient = integer / radix;
  if (Number(integer - quotient * radix) - (meta.config.levels - 1) / 2 !== meta.weights[i]) throw Error('Integer mismatch');
  integer = quotient;
}
if (integer !== 1n) throw Error('Missing radix marker');
let maxError = 0, total = 0, count = 0, comparisons = 0;
if (meta.runtime_legacy) {
  for (let ni=0;ni<meta.names.length;ni++) {
    let cursor=0; const text=meta.names[ni]+'\n';
    Math.random=()=>{
      const raw=globalThis.p, logits=raw.map(v=>Math.log(v)/1.25), m=Math.max(...logits);
      const sum=logits.reduce((a,v)=>a+Math.exp(v-m),0);
      for(let j=0;j<logits.length;j++) {
        maxError=Math.max(maxError,Math.abs(logits[j]-m-Math.log(sum)-meta.logprobs[ni][cursor][j])); comparisons++;
      }
      const target=meta.chars.indexOf(text[cursor]);
      total-=(logits[target]-m-Math.log(sum))/Math.LN2;count++;cursor++;
      const denom=raw.reduce((a,b)=>a+b,0);
      return (raw.slice(0,target).reduce((a,b)=>a+b,0)+raw[target]/2)/denom;
    };
    if(g()!==meta.names[ni] || cursor!==text.length)throw Error('Forced-prefix mismatch');
  }
} else {
for (let ni = 0; ni < meta.names.length; ni++) {
  globalThis.K = []; globalThis.U = [];
  globalThis.KK = Array.from({length:meta.config.layers || 1},()=>[]);
  globalThis.UU = Array.from({length:meta.config.layers || 1},()=>[]);
  let input = 0, ctx = Array(meta.config.k || 3).fill(0);
  const name = meta.names[ni] + '\n';
  for (let p = 0; p < name.length; p++) {
    if (meta.config.window > 1 && p === 0) globalThis.R = [];
    const raw = f((meta.kind === 'tf' || meta.kind === 'tap') ? input : ctx, p);
    const untempered=raw.map(v=>v**(meta.runtime_temperature || 1));
    const sum=untempered.reduce((a,b)=>a+b,0),probs=untempered.map(v=>v/sum);
    for (let j = 0; j < probs.length; j++) {
      if (!(probs[j] > 0) || !Number.isFinite(probs[j])) throw Error('Invalid probability');
      maxError = Math.max(maxError, Math.abs(Math.log(probs[j])-meta.logprobs[ni][p][j]));
      comparisons++;
    }
    input = meta.chars.indexOf(name[p]);
    total -= Math.log2(probs[input]); count++;
    ctx = ctx.slice(1).concat(input);
  }
}
}
if (maxError > 1e-9) throw Error('Forward mismatch '+maxError);
let seed = 1, draws = 0;
Math.random = () => {
  if (++draws > 10000000) throw Error('Sampling did not terminate');
  seed = (Math.imul(seed,1664525)+1013904223) >>> 0;
  return seed / 4294967296;
};
const real = new Set(meta.all_names), outputs = [];
for (let i = 0; i < +(process.argv[4] || 2000); i++) outputs.push(g());
console.log(JSON.stringify({node_bpc:total/count,validation_chars:count,probability_comparisons:comparisons,
  max_logprob_error:maxError,novel_pct:100*outputs.filter(x=>!real.has(x)).length/outputs.length,
  over24_pct:100*outputs.filter(x=>x.length>24).length/outputs.length,
  samples:outputs.slice(0,10),sample_count:outputs.length,sample_seed:1,sample_temperature:.8,
  longest_sample:Math.max(...outputs.map(s=>s.length)),rng_draws:draws}));
