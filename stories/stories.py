"""Character-level story models for the 64 KB and 1 MB URL tiers (TinyStories).

Model: small transformer, pre-norm (RMSNorm without gain), ALiBi attention limited to the last
CTX tokens (so a rolling key/value cache in JS gives exactly the same numbers), GELU(tanh) MLP,
no biases, output tied to the embedding. Tensor or compact row scales, with QAT.

  uv run python stories/stories.py train  NAME d=64 L=2 H=4 ctx=256 bits=4 steps=4000 bs=32 lr=3e-3
  uv run python stories/stories.py export NAME          # -> out/NAME_url.txt, out/NAME_meta.json, eval
"""
import sys, os, json, math, time, base64, signal
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
os.makedirs('out', exist_ok=True)
VOCAB = json.loads(Path('data/vocab.json').read_text())['vocab']; V = len(VOCAB)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for tools/
from tools.host import PREFIX
CLIP = {3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5}   # clip range in std units per bit width


def quantize(w, bits, mode='tensor', clip=0.0):
    """Return integer weights and exactly exportable scales (no learned scale parameters).

    Row scales use one 8-bit log multiplier per output row, relative to a four-significant-
    figure tensor scale. This adds only 1,147 bytes for d=60,L=2, rather than JS float literals.
    The fixed log grid and rounded reference are used during QAT as well as export.
    """
    if mode not in ('tensor', 'row'): raise ValueError(f'unknown scale mode: {mode}')
    M = 2 ** (bits - 1) - 1
    c = clip or CLIP[bits]
    base = float('%.4g' % max(w.detach().std().item() * c / M, 1e-8))
    codes = None
    if mode == 'row':
        raw = w.detach().std(dim=1, keepdim=True).clamp(min=1e-8) * c / M
        codes = (128 + 32 * torch.log2(raw / base)).round().clamp(0, 255)
        scales = base * 2 ** ((codes - 128) / 32)
    else:
        scales = base
    q = (w.detach() / scales).round().clamp(-M, M)
    return q, scales, base, codes


def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


class GPT(nn.Module):
    def __init__(s, d, L, H, ctx, bits, scale='tensor', qclip=0.0, ste='clipped'):
        super().__init__(); s.d, s.L, s.H, s.ctx, s.bits, s.q = d, L, H, ctx, bits, False
        s.scale, s.qclip, s.ste = scale, qclip, ste
        s.emb = nn.Parameter(torch.randn(V, d) * 0.5)
        s.blocks = nn.ParameterList()
        for _ in range(L):
            for (o, i) in [(3 * d, d), (d, d), (4 * d, d), (d, 4 * d)]:
                s.blocks.append(nn.Parameter(torch.randn(o, i) / math.sqrt(i)))
        for l in range(L): s.blocks[4 * l + 1].data *= 0.5 / math.sqrt(L); s.blocks[4 * l + 3].data *= 0.5 / math.sqrt(L)
        sl = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)])
        s.register_buffer('slopes', sl)

    def tensors(s): return [s.emb] + list(s.blocks)

    def weight(s, w):
        if not s.q: return w
        if getattr(s, '_eval_weights', None) is not None: return s._eval_weights[id(w)]
        q, scales, _, _ = quantize(w, s.bits, s.scale, s.qclip)
        if s.ste == 'identity':
            return w + (q * scales - w).detach()
        # Round has an identity STE; saturation has its actual zero gradient. This prevents
        # clipped outliers from growing indefinitely and coarsening every other weight's grid.
        bound = (2 ** (s.bits - 1) - 1) * scales
        clipped = w.clamp(min=-bound, max=bound)
        return clipped + (q * scales - clipped).detach()

    def forward(s, x, keep=None):
        """x: (B,T). Attention to positions within ctx-1 back (banded). keep: return last `keep` positions."""
        B, T = x.shape; d, H = s.d, s.H; hd = d // H
        E = s.weight(s.emb); h = E[x]
        pos = torch.arange(T); dist = pos.view(T, 1) - pos.view(1, T)
        mask = (dist < 0) | (dist >= s.ctx)
        bias = -s.slopes.view(H, 1, 1) * dist.clamp(min=0).float()
        bias = bias.masked_fill(mask, float('-inf'))
        for l in range(s.L):
            Wqkv, Wo, W1, W2 = [s.weight(w) for w in s.blocks[4 * l:4 * l + 4]]
            q, k, v = F.linear(rms(h), Wqkv).split(d, -1)
            q, k, v = [t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v)]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            h = h + F.linear(a.transpose(1, 2).reshape(B, T, d), Wo)
            h = h + F.linear(F.gelu(F.linear(rms(h), W1), approximate='tanh'), W2)
        if keep: h = h[:, -keep:]
        return F.linear(rms(h), E)


def parse(args):
    c = dict(d=64, L=2, H=4, ctx=256, bits=4, steps=4000, bs=32, lr=3e-3, qat=0.5,
             wd=0.0, seed=0, minutes=24.0, scale='row', qclip=0.0, warmup=200, min_lr=0.02,
             checkpoint_seconds=60.0, eval_seconds=120.0, dev_chars=10000, diag_chars=10000,
             init='', resume='', qat_lr=1.0, segment_minutes=20.0, ste='clipped')
    for a in args:
        k, v = a.split('=', 1)
        if k not in c: raise ValueError(f'unknown option: {k}')
        c[k] = type(c[k])(v)
    return c


def model(cfg):
    return GPT(cfg['d'], cfg['L'], cfg['H'], cfg['ctx'], cfg['bits'], cfg.get('scale', 'tensor'), cfg.get('qclip', 0.0), cfg.get('ste', 'identity'))


def selection_data(n):
    """Validation *after* the published 500k benchmark; never select on the benchmark."""
    path = 'data/selection.u8'
    if not os.path.exists(path) or os.path.getsize(path) < n:
        stoi = {c: i for i, c in enumerate(VOCAB)}; count = 0; buf = bytearray()
        with open('data/valid.txt', encoding='utf-8') as f:
            story = []
            for line in f:
                if '<|endoftext|>' not in line:
                    story.append(line.replace('\r', '')); continue
                text = ''.join(story).strip(); story = []
                if not text: continue
                for c in text + '\0':
                    if count >= 500000: buf.append(stoi.get(c, stoi[' ']))
                    count += 1
                if len(buf) >= n: break
        with open(path, 'wb') as f: f.write(buf)
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8, count=n).astype(np.int64))


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)


def train(name, cfg):
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _train(name, cfg)
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


def _train(name, cfg):
    resume_ck = None
    if cfg['resume']:
        resume_path = cfg['resume']
        resume_ck = torch.load(resume_path, weights_only=False)
        cfg = {**parse([]), **resume_ck['cfg'], 'resume': resume_path,
               'ste': resume_ck['cfg'].get('ste', 'identity')}
    if cfg['d'] % cfg['H'] or cfg['bits'] not in CLIP or not 0 <= cfg['qat'] <= 1:
        raise ValueError('require d divisible by H, supported bit width, and 0 <= qat <= 1')
    if cfg['scale'] not in ('tensor', 'row') or cfg['ste'] not in ('identity', 'clipped'):
        raise ValueError('invalid scale or STE mode')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    data = np.memmap('data/train.u8', dtype=np.uint8, mode='r')
    N = len(data) - 1_000_000
    dev = selection_data(cfg['dev_chars'])
    tail = torch.from_numpy(np.array(data[N:N + cfg['diag_chars']], dtype=np.int64))
    held = torch.from_numpy(np.fromfile('data/heldout.u8', dtype=np.uint8, count=cfg['diag_chars']).astype(np.int64))
    m = model(cfg)
    npar = sum(p.numel() for p in m.parameters()); print(name, cfg, 'params', npar, flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=cfg['wd'])
    start = 0; history = []; best = best_float = float('inf'); previous_seconds = 0
    parent_seconds = parent_steps = parent_tokens = 0
    source = cfg['resume'] or cfg['init']
    if source:
        ck = resume_ck or torch.load(source, weights_only=False); m.load_state_dict(ck['sd'])
        if cfg['resume']:
            opt.load_state_dict(ck['opt']); start = ck['steps']; history = ck.get('history', [])
            torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng'])
            best = ck.get('best_dev', best); best_float = ck.get('best_float_dev', best_float)
            parent_seconds = ck.get('parent_seconds', 0)
            parent_steps = ck.get('parent_steps', 0); parent_tokens = ck.get('parent_tokens', 0)
            previous_seconds = ck['seconds'] - parent_seconds
        else:
            parent_seconds = ck['seconds']; parent_steps = ck.get('total_steps', ck['steps'])
            parent_tokens = ck.get('tokens_seen', parent_steps * ck['cfg']['bs'] * ck['cfg']['ctx'])
    S, T, bs = cfg['steps'], cfg['ctx'], cfg['bs']; t0 = time.monotonic()
    S = ck.get('planned_steps', S) if cfg['resume'] else S
    qstart = int(S * (1 - cfg['qat'])); last_save = last_eval = eval_elapsed = 0; done = start; tl = ntl = 0
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def checkpoint(path):
        atomic_save(dict(cfg=cfg, sd=m.state_dict(), opt=opt.state_dict(), steps=done,
                         planned_steps=S, seconds=parent_seconds + previous_seconds + time.monotonic() - t0,
                         parent_seconds=parent_seconds, parent_steps=parent_steps, parent_tokens=parent_tokens,
                         total_steps=parent_steps + done, tokens_seen=parent_tokens + done * bs * T,
                         best_dev=best, best_float_dev=best_float, history=history,
                         torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()), path)

    def assess(phase):
        nonlocal best, best_float, last_eval, eval_elapsed
        eval_t0 = time.monotonic()
        row = dict(step=done, phase=phase, qat=bool(m.q), lr=opt.param_groups[0]['lr'],
                   seconds=round(previous_seconds + time.monotonic() - t0, 2))
        m.q = False; row['float_dev_bpc'] = evaluate(m, dev)
        row['float_heldout_probe_bpc'] = evaluate(m, held)
        m.q = True; row['quant_dev_bpc'] = evaluate(m, dev)
        row['quant_heldout_probe_bpc'] = evaluate(m, held)
        row['quant_train_tail_bpc'] = evaluate(m, tail)
        history.append(row); print(json.dumps(row), flush=True)
        if row['float_dev_bpc'] < best_float:
            best_float = row['float_dev_bpc']; checkpoint(f'out/{name}.float.pt')
        if row['quant_dev_bpc'] < best:
            best = row['quant_dev_bpc']; checkpoint(f'out/{name}.pt')
        checkpoint(f'out/{name}.last.pt')
        with open(f'out/{name}_history.json', 'w') as f: json.dump(history, f, indent=1)
        last_eval = time.monotonic() - t0
        eval_elapsed += time.monotonic() - eval_t0

    for it in range(start, S):
        # time guard: compress the schedule if we run out of time
        elapsed = time.monotonic() - t0
        el = (previous_seconds + elapsed) / 60
        if it == 50 and not cfg['resume']:
            step_seconds = (elapsed - eval_elapsed) / 50
            est = step_seconds * S
            if est > cfg['minutes']:
                # Reserve time for periodic diagnostics; retain the same step schedule on resume.
                S = max(51, int(cfg['minutes'] * 60 / step_seconds * 0.82))
                qstart = int(S * (1 - cfg['qat'])); print('steps ->', S, 'qat starts', qstart, flush=True)
        if it >= S: break
        if stop_requested or el >= cfg['minutes'] or elapsed >= cfg['segment_minutes'] * 60:
            print('wall clock limit: saving current state', flush=True); break
        if it == qstart:
            checkpoint(f'out/{name}.preqat.pt'); assess('pre_qat')
        m.q = it >= qstart
        lr = cfg['lr'] * min(1, (it + 1) / max(1, cfg['warmup'])) * (0.5 + 0.5 * math.cos(math.pi * it / S)) + cfg['lr'] * cfg['min_lr']
        if m.q: lr *= cfg['qat_lr']
        for g in opt.param_groups: g['lr'] = lr
        ix = np.random.randint(0, N - T - 1, bs)
        b = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64))
        loss = F.cross_entropy(m(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        tl += loss.item(); ntl += 1; done = it + 1
        if (it + 1) % 250 == 0:
            print(f'{it+1} loss {tl/ntl/math.log(2):.3f} bpc  q={m.q} lr={lr:.6g} {elapsed:.0f}s', flush=True); tl = ntl = 0
        if elapsed - last_save >= cfg['checkpoint_seconds'] or done == 50:
            checkpoint(f'out/{name}.last.pt'); last_save = elapsed
        if elapsed - last_eval >= cfg['eval_seconds']:
            assess('qat' if m.q else 'float')
    checkpoint(f'out/{name}.last.pt')
    assess('end')
    print('best quantised selection dev:', best, 'actual steps:', done, flush=True)


@torch.no_grad()
def evaluate(m, seq, block=None):
    """bits/char of predicting seq[0..] given token 0 before it, exactly as the JS rolling cache does."""
    was_training = m.training; m.eval(); x = torch.cat([torch.zeros(1, dtype=torch.long), seq.long()])
    pre = m.L * (m.ctx - 1); block = block or m.ctx; tot = 0.0
    if m.q:
        m._eval_weights = {}
        for w in m.tensors():
            q, scale, _, _ = quantize(w, m.bits, m.scale, m.qclip)
            m._eval_weights[id(w)] = q * scale
    try:
        for i in range(0, len(seq), block):
            j = min(i + block, len(seq)); a = max(0, i - pre)
            lg = m(x[a:j].view(1, -1), keep=j - i)[0]
            tot += F.cross_entropy(lg, x[i + 1:j + 1], reduction='sum').item()
    finally:
        m._eval_weights = None
        m.train(was_training)
    return tot / len(seq) / math.log(2)


# ---------------- export ----------------
B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def pack_bits(vals, bits):
    acc = nb = 0; out = []
    for v in vals:
        acc = (acc << bits) | int(v); nb += bits
        while nb >= 6: nb -= 6; out.append(B64[(acc >> nb) & 63])
        acc &= (1 << nb) - 1
    if nb: out.append(B64[(acc << (6 - nb)) & 63])
    return ''.join(out)


def runtime(m, scales, temp):
    """JS program. Globals used by verify_stories.js: R() reset, S(t) -> probabilities of next char."""
    js = (Path(__file__).resolve().parent / 'stories_runtime.js').read_text().strip()
    vocab = json.dumps(VOCAB[1:], ensure_ascii=True)
    return (f"C=[{m.d},{m.L},{m.H},{m.ctx},{m.bits},{temp},{int(m.scale == 'row')}];K=[{','.join(scales)}];X={vocab};" + js)


def export(name, temp=0.8):
    ck = torch.load(f'out/{name}.pt', weights_only=False); cfg = ck['cfg']
    m = model(cfg); m.load_state_dict(ck['sd'])
    bits = m.bits; M = 2 ** (bits - 1) - 1; vals = []; scales = []; codes = []
    for w in m.tensors():
        q, s, base, rows = quantize(w, bits, m.scale, m.qclip)
        scales.append('%.4g' % base)
        if rows is not None: codes.extend(rows.flatten().to(torch.int64).tolist())
        w.data.copy_(q * s)                  # python eval uses exactly the exported numbers
        vals.append((q + M).flatten().to(torch.int64).numpy())
    vals = np.concatenate(vals); m.q = False
    scale_str = pack_bits(codes, 8) if codes else ''
    wstr = scale_str + pack_bits(vals, bits)
    prog = runtime(m, scales, temp)
    b64 = base64.urlsafe_b64encode(prog.encode()).decode().rstrip('=')
    url = f'{PREFIX}{b64}.{wstr}'
    if name == 't64kfix' and len(url) > 65536:
        raise ValueError(f'64 KB tier overflow: {len(url)} URL characters')
    open(f'out/{name}_url.txt', 'w').write(url)
    held = torch.from_numpy(np.fromfile('data/heldout.u8', dtype=np.uint8).astype(np.int64))
    t0 = time.time(); bpc = evaluate(m, held); te = time.time() - t0
    bpc20 = evaluate(m, held[:20_000])
    utf8 = json.load(open('data/vocab.json'))['heldout_utf8_bytes']
    meta = dict(name=name, cfg=cfg, steps=ck.get('total_steps', ck['steps']), train_seconds=round(ck['seconds']), params=int(len(vals)),
                program_chars=len(prog), program_b64_chars=len(b64), weight_chars=len(wstr), url_chars=len(url),
                heldout_chars=len(held), heldout_bpc=round(bpc, 4), heldout_first20k_bpc=round(bpc20, 5), heldout_bits_per_byte=round(bpc * len(held) / utf8, 4),
                tokens_seen=ck.get('tokens_seen', ck['steps'] * cfg['bs'] * cfg['ctx']),
                scale_chars=len(scale_str), packed_payload_bytes=(len(codes) * 8 + len(vals) * bits + 7) // 8,
                payload_bytes=len(prog.encode()) + len(codes) + (len(vals) * bits + 7) // 8,
                selection_dev_bpc=ck.get('best_dev'), checkpoint_history=ck.get('history', []))
    json.dump(meta, open(f'out/{name}_meta.json', 'w'), indent=1)
    print(json.dumps(meta), f'eval {te:.0f}s')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('train', 'export'):
        print(__doc__); sys.exit(0 if sys.argv[1:2] in (['-h'], ['--help']) else 2)
    if sys.argv[1] == 'train': train(sys.argv[2], parse(sys.argv[3:]))
    elif sys.argv[1] == 'export': export(sys.argv[2], *[float(a) for a in sys.argv[3:]])
