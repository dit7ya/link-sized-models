// No packages: test the exact loader's fallback against Node zlib.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict'),zlib=require('zlib');
const html=fs.readFileSync('site/index.html','utf8');
const shim=html.split('// DEFLATE-FALLBACK:START')[1].split('// DEFLATE-FALLBACK:END')[0];
const ctx=vm.createContext({Uint8Array,TransformStream,DecompressionStream:undefined});
vm.runInContext(shim,ctx);
const fixtures=[Buffer.alloc(0),Buffer.from('hello '.repeat(30000)),Buffer.from(Array.from({length:8192},(_,i)=>(i*i+7*i)&255))];
let seed=1;fixtures.push(Buffer.from(Array.from({length:100000},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed>>>24})));
(async()=>{
let count=0;
for(const raw of fixtures)for(const options of [{level:0},{strategy:zlib.constants.Z_FIXED},{level:9}]){
  const compressed=zlib.deflateSync(raw,options);
  assert.deepEqual(Buffer.from(ctx.inflateZlib(compressed)),raw);
  const stream=new ctx.DecompressionStream('deflate');
  const output=new Response(new Blob([compressed.subarray(0,3),compressed.subarray(3)]).stream().pipeThrough(stream)).arrayBuffer();
  assert.deepEqual(Buffer.from(await output),raw);count++;
  const bad=Buffer.from(compressed);bad[bad.length-1]^=1;assert.throws(()=>ctx.inflateZlib(bad));
  assert.throws(()=>ctx.inflateZlib(compressed.subarray(0,compressed.length-2)));
}
assert.throws(()=>ctx.inflateZlib(zlib.deflateSync(Buffer.alloc(16*1024*1024+1))));
console.log(`${count} DEFLATE round trips passed (stored/fixed/dynamic blocks, split streams); corrupt/truncated/oversized input rejected`);
})().catch(e=>{console.error(e);process.exitCode=1});
