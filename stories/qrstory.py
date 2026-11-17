"""QR-sized TinyStories model: character MLP over a reduced 33-symbol alphabet.

  uv run python stories/qrstory.py train --e 6 --h 32 --k 3 --levels 31 --tag L31_h32_k3_e6
  uv run python stories/qrstory.py export out/qrstory_L31_h32_k3_e6.pt --temp 0.7 --publish

Data: data/qrstory_train_raw.txt is the first 12 MiB of TinyStoriesV2-GPT4-train.txt
  curl -sSfL -r 0-12582911 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -o data/qrstory_train_raw.txt
Selection uses the last 5% of those stories; the reported score is data/heldout.u8
(get_data.py's fixed 500k-character validation benchmark) after the same mapping.
Bits/char are on the reduced alphabet and are not comparable with the bigger tiers.
"""
import argparse, copy, json, math, os, re, subprocess, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for tools/
from tools.host import QR_PREFIX

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
# index 0 is space (also the story-start padding), then punctuation, then a..z.
SPECIAL = ' .,"\'!?'
ALPHA = SPECIAL + 'abcdefghijklmnopqrstuvwxyz'
V = len(ALPHA)
STOI = {c: i for i, c in enumerate(ALPHA)}
EOT = '<|endoftext|>'
SAMPLE_LEN = 220
TRANS = {'“': '"', '”': '"', '‘': "'", '’': "'", ':': ',', ';': ',',
         '-': ' ', '–': ' ', '—': ' ', '\n': ' ', '\t': ' ', '…': '.'}


def normalize(story):
    s = ''.join(TRANS.get(c, c) for c in story.lower())
    s = ''.join(c for c in s if c in STOI)
    return re.sub(' +', ' ', s).strip()


def encode(stories, k):
    """Stream of [pad*k, story...] blocks; mask marks positions that are story chars."""
    parts, masks = [], []
    for s in stories:
        s = normalize(s)
        if not s: continue
        parts.append(np.zeros(k, np.uint8)); masks.append(np.zeros(k, bool))
        parts.append(np.frombuffer(bytes(STOI[c] for c in s), np.uint8)); masks.append(np.ones(len(s), bool))
    return np.concatenate(parts), np.concatenate(masks)


def train_stories():
    text = (ROOT / 'data/qrstory_train_raw.txt').read_text(encoding='utf-8', errors='ignore').replace('\r', '')
    stories = [s.strip() for s in text.split(EOT)[:-1] if s.strip()]  # last one may be cut
    cut = int(len(stories) * .95)
    return stories[:cut], stories[cut:]


def heldout_stories():
    vocab = json.loads((ROOT / 'data/vocab.json').read_text())['vocab']
    raw = (ROOT / 'data/heldout.u8').read_bytes()
    text = ''.join(vocab[b] for b in raw)
    return [s for s in text.split('\0') if s.strip()]


def contexts(stream, mask, k):
    idx = np.nonzero(mask)[0].astype(np.int64)
    X = np.stack([stream[idx - k + i] for i in range(k)], 1)
    return torch.from_numpy(X.astype(np.int64)), torch.from_numpy(stream[idx].astype(np.int64))


class StoryMLP(nn.Module):
    def __init__(self, e=6, h=33, k=3, levels=31):
        super().__init__()
        self.cfg = dict(e=e, h=h, k=k, levels=levels)
        self.emb = nn.Parameter(torch.randn(V, e) * .5)
        self.w1 = nn.Parameter(torch.randn(h, k * e) / math.sqrt(k * e)); self.b1 = nn.Parameter(torch.zeros(h))
        self.w2 = nn.Parameter(torch.randn(V, h) / math.sqrt(h)); self.b2 = nn.Parameter(torch.zeros(V))
        self.names = ['emb', 'w1', 'b1', 'w2', 'b2']
        self.scales = nn.ParameterDict({n: nn.Parameter(torch.zeros(())) for n in self.names})
        self.active = False

    def reset_scales(self):
        M = (self.cfg['levels'] - 1) // 2
        with torch.no_grad():
            for n in self.names:
                self.scales[n].copy_((getattr(self, n).abs().max().clamp_min(1e-4) / M).log())

    def qw(self, n):
        w = getattr(self, n)
        if not self.active: return w
        M = (self.cfg['levels'] - 1) // 2
        sc = self.scales[n].exp(); v = w / sc; v = v + (v.round() - v).detach()
        return v.clamp(-M, M) * sc

    def forward(self, X):
        z = self.qw('emb')[X].reshape(len(X), -1)
        return F.linear(torch.tanh(F.linear(z, self.qw('w1'), self.qw('b1'))), self.qw('w2'), self.qw('b2'))


def bpc(m, X, Y):
    m.eval(); tot = 0.
    with torch.no_grad():
        for xb, yb in zip(X.split(65536), Y.split(65536)):
            tot += F.cross_entropy(m(xb), yb, reduction='sum').item()
    return tot / len(Y) / math.log(2)


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, sel = train_stories()
    Xs, Ys = contexts(*encode(sel, a.k), a.k)
    stream, mask = encode(tr, a.k)
    idx = np.nonzero(mask)[0].astype(np.int64)
    print('train chars', len(idx), 'select chars', len(Ys), flush=True)
    st = torch.from_numpy(stream.astype(np.int64)); idx = torch.from_numpy(idx)
    m = StoryMLP(a.e, a.h, a.k, a.levels)
    opt = torch.optim.AdamW([{'params': [getattr(m, n) for n in m.names]},
                             {'params': list(m.scales.parameters()), 'weight_decay': 0.}], lr=a.lr, weight_decay=a.wd)
    best = (float('inf'), None); hist = []; t0 = time.monotonic()
    for i in range(a.steps):
        if i == a.warmup: m.active = True; m.reset_scales()
        m.train()
        b = idx[torch.randint(len(idx), (a.batch,))]
        xb = torch.stack([st[b - a.k + j] for j in range(a.k)], 1)
        loss = F.cross_entropy(m(xb), st[b])
        lr = a.lr * (.03 + .97 * (1 + math.cos(math.pi * i / a.steps)) / 2)
        for g in opt.param_groups: g['lr'] = lr
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        if (i + 1) % a.eval_every == 0 and m.active:
            s = bpc(m, Xs, Ys); hist.append(dict(step=i + 1, select_bpc=s, sec=round(time.monotonic() - t0, 1)))
            print(a.tag, hist[-1], flush=True)
            if s < best[0]: best = (s, copy.deepcopy(m.state_dict()))
    torch.save(dict(cfg=m.cfg, state=best[1], select_bpc=best[0], args=vars(a), history=hist), OUT / f'qrstory_{a.tag}.pt')
    print('DONE', a.tag, best[0], round(time.monotonic() - t0, 1), flush=True)


def lit(x):
    s = format(float(x), '.4g')
    return s[1:] if s.startswith('0.') else s.replace('-0.', '-.')


def export_parts(path):
    saved = torch.load(path, weights_only=False)
    cfg = saved['cfg']; m = StoryMLP(**cfg); m.load_state_dict(saved['state'])
    M = (cfg['levels'] - 1) // 2
    ints, sc = {}, {}
    for n in m.names:
        s = float(m.scales[n].detach().exp())
        ints[n] = np.rint(getattr(m, n).detach().double().numpy() / s).clip(-M, M).astype(np.int64); sc[n] = s
    return cfg, ints, sc


def program(cfg, ints, sc, T, cap):
    e, h, k, L = cfg['e'], cfg['h'], cfg['k'], cfg['levels']; M = (L - 1) // 2; ke = k * e
    oE = 0; o1 = oE + V * e; ob = o1 + h * ke; o2 = ob + h; oc = o2 + V * h; N = oc + V
    c1, cb, c2, cc = lit(sc['emb'] * sc['w1']), lit(sc['b1']), lit(sc['w2'] / T), lit(sc['b2'] / T)
    spec = ','.join(str(ord(c)) for c in SPECIAL)
    xs = '[' + ','.join(['0'] * k) + ']'
    shift = '[' + ','.join(f'x[{i}]' for i in range(1, k)) + ',j]'
    js = (f"for(n=BigInt(d),W=[],i={N};i--;)W[i]=Number(n-(n/={L}n)*{L}n)-{M};"
          f"g=_=>{{with(Math)for(x={xs},o='';o.length<{SAMPLE_LEN};){{"
          f"for(H=[],j=0;j<{h};j++)for(a=W[{ob}+j]*{cb},i=0;i<{ke};i++)H[j]=tanh(a+=W[{o1}+j*{ke}+i]*W[x[t=i/{e}|0]*{e}+i-t*{e}]*{c1});"
          f"for(p=[],t=j=0;j<{V};j++){{for(a=W[{oc}+j]*{cc},i=0;i<{h};i++)a+=W[{o2}+j*{h}+i]*H[i]*{c2};t+=p[j]=exp(a)}}"
          f"for(r=random()*t,j=0;(r-=p[j])>0;j++);o+=String.fromCharCode([{spec}][j]||j+{97 - len(SPECIAL)});x={shift}}}")
    js += "return(o).replace(/(^|[.!?]\\s)[a-z]/g,s=>s.toUpperCase())}" if cap else "return(o)}"
    consts = dict(c1=float(c1), cb=float(cb), c2=float(c2), cc=float(cc))
    return js, consts


def py_logits(cfg, ints, consts, X):
    """Float64 mirror of the JS forward pass (tempered logits)."""
    E = ints['emb'].astype(np.float64)
    z = E[X].reshape(len(X), -1)
    H = np.tanh(ints['b1'] * consts['cb'] + (z @ ints['w1'].T.astype(np.float64)) * consts['c1'])
    return ints['b2'] * consts['cc'] + (H @ ints['w2'].T.astype(np.float64)) * consts['c2']


def logsoftmax(a):
    a = a - a.max(1, keepdims=True); return a - np.log(np.exp(a).sum(1, keepdims=True))


def export(a):
    from tools.qr import qr_measure, zxing_ok
    cfg, ints, sc = export_parts(a.checkpoint)
    L = cfg['levels']; M = (L - 1) // 2
    js, consts = program(cfg, ints, sc, a.temp, not a.nocap)
    # Like qrtf.py: ASCII, no % or #; also no space/quote/backtick (< > appear in the existing programs too).
    assert js.isascii() and not any(c in js for c in '%# "`'), 'Unsafe raw URL program'
    flat = [int(v) for n in ['emb', 'w1', 'b1', 'w2', 'b2'] for v in ints[n].flat]
    packed = 1
    for v in flat: packed = packed * L + v + M
    digits = str(packed)
    url = QR_PREFIX + f'{len(js):03d}' + js + digits
    qr, ver, used, cap, modes = qr_measure(url, 20)
    # held-out score of the exact exported numbers at temperature 1
    k = cfg['k']; Xh, Yh = contexts(*encode(heldout_stories(), k), k)
    Xh, Yh = Xh.numpy(), Yh.numpy()
    lp = logsoftmax(py_logits(cfg, ints, consts, Xh) * a.temp)
    held = -lp[np.arange(len(Yh)), Yh].mean() / math.log(2)
    tr, sel = train_stories(); Xs, Ys = contexts(*encode(sel, k), k)
    lps = logsoftmax(py_logits(cfg, ints, consts, Xs.numpy()) * a.temp)
    selb = -lps[np.arange(len(Ys)), Ys.numpy()].mean() / math.log(2)
    stem = OUT / ('qrstory' if a.publish else 'qrstory_' + Path(a.checkpoint).stem)
    row = dict(model=Path(a.checkpoint).name, architecture='MLP: previous %d chars x embedding %d -> %d tanh -> %d softmax' % (k, cfg['e'], cfg['h'], V),
               config=cfg, weights=len(flat), levels=L, program_chars=len(js), digits=len(digits), url_chars=len(url),
               qr_version=ver, ecc='L', data_bits=used, capacity_bits=cap, slack_bits=cap - used, segments=modes,
               alphabet=ALPHA, alphabet_note='lowercased; curly quotes->straight; : ; -> comma; dashes/newlines -> space; other chars dropped; spaces collapsed. Output capitalises the first letter and letters after ". ", "! ", "? ".' if not a.nocap else 'lowercased reduced alphabet',
               heldout_bpc_reduced_alphabet=held, heldout_chars=int(len(Yh)),
               heldout_note='data/heldout.u8 (fixed 500k-char TinyStoriesV2 validation benchmark) after the reduced-alphabet mapping; exported quantised weights, temperature 1; not comparable with bigger tiers (different alphabet)',
               select_bpc=selb, temperature=a.temp, sample_length=SAMPLE_LEN,
               train_data='first 12 MiB of TinyStoriesV2-GPT4-train.txt, 95% of stories for training, last 5% for checkpoint selection')
    if ver > 30 or len(url) > 3492:
        print(json.dumps(row, indent=2)); raise SystemExit('Too big: v%d, %d chars' % (ver, len(url)))
    Path(str(stem) + '_url.txt').write_text(url)
    zxing_ok(qr, url, str(stem) + '_qr.png'); row['zxing_exact'] = True
    # parity fixture: first SAMPLE_LEN chars of the first 15 held-out stories that are long enough
    seqs = [normalize(s) for s in heldout_stories()]
    seqs = [[STOI[c] for c in s[:SAMPLE_LEN]] for s in seqs if len(s) >= SAMPLE_LEN][:15]
    lps = []
    for q in seqs:
        ctx = np.array([([0] * k + q)[i:i + k] for i in range(SAMPLE_LEN)])
        lps.append(logsoftmax(py_logits(cfg, ints, consts, ctx)).tolist())
    fix = dict(seqs=seqs, logprobs=lps, temperature=a.temp, alphabet=ALPHA, weights=flat, levels=L)
    fixture = OUT / 'qrstory_fixture.json'
    fixture.write_text(json.dumps(fix, separators=(',', ':')))
    res = subprocess.run(['node', 'stories/verify_qrstory.js', str(stem) + '_url.txt', str(fixture)], capture_output=True, text=True, timeout=600)
    if res.returncode: raise RuntimeError(res.stderr)
    v = json.loads(res.stdout); row.update(v)
    assert v['max_logprob_error'] < 1e-9, v['max_logprob_error']
    Path(str(stem) + '_meta.json').write_text(json.dumps(row, indent=2) + '\n')
    fixture.unlink()
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--e', type=int, default=6); t.add_argument('--h', type=int, default=33)
    t.add_argument('--k', type=int, default=3); t.add_argument('--levels', type=int, default=31)
    t.add_argument('--steps', type=int, default=30000); t.add_argument('--warmup', type=int, default=3000)
    t.add_argument('--batch', type=int, default=1024); t.add_argument('--lr', type=float, default=4e-3)
    t.add_argument('--wd', type=float, default=0.); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--eval-every', type=int, default=3000); t.add_argument('--tag', required=True)
    x = sub.add_parser('export')
    x.add_argument('checkpoint'); x.add_argument('--temp', type=float, default=.8)
    x.add_argument('--nocap', action='store_true'); x.add_argument('--publish', action='store_true')
    a = p.parse_args()
    train(a) if a.cmd == 'train' else export(a)
