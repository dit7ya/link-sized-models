import torch, torch.nn as nn, torch.nn.functional as F, math, os, random, sys, time
torch.set_num_threads(2)
names = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatnames.txt')).read().split()
random.Random(0).shuffle(names)
nval = len(names) // 8
val, tr = names[:nval], names[nval:]
chars = '\n-.0123456789abcdefghijklmnopqrstuvwxyz'; V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
LMAX = 26

def seq_data(ns):
    X = torch.zeros(len(ns), LMAX, dtype=torch.long); Y = torch.full((len(ns), LMAX), -1)
    for i, n in enumerate(ns):
        ids = [stoi[c] for c in n] + [0]
        X[i, 1:len(ids)] = torch.tensor(ids[:-1]); Y[i, :len(ids)] = torch.tensor(ids)
    return X, Y

def quant(w, M):
    if not M: return w
    s = w.abs().max().clamp(min=1e-5) / M
    return w + ((w / s).round().clamp(-M, M) * s - w).detach()

class QLin(nn.Module):
    def __init__(s, i, o, M, bias=True):
        super().__init__(); s.w = nn.Parameter(torch.randn(o, i) / math.sqrt(i)); s.b = nn.Parameter(torch.zeros(o)) if bias else None; s.M = M
    def forward(s, x): return F.linear(x, quant(s.w, s.M), s.b)

class Net(nn.Module):
    """kind: 'rnn' (tanh), 'gru', or 'mlp' (k previous chars)"""
    def __init__(s, kind, e, h, M, k=3, drop=0.0, pos=0):
        super().__init__(); s.kind, s.e, s.h, s.M, s.k, s.pos = kind, e, h, M, k, pos
        s.emb = nn.Parameter(torch.randn(V, e) * .5); s.drop = nn.Dropout(drop)
        if kind == 'rnn': s.l1 = QLin(e + h, h, M)
        elif kind == 'gru': s.lz = QLin(e + h, h, M); s.lr = QLin(e + h, h, M); s.ln = QLin(e + h, h, M)
        else: s.l1 = QLin(k * e + pos, h, M)
        s.l2 = QLin(h, V, M)
    def forward(s, x):
        B, L = x.shape; E = quant(s.emb, s.M)
        if s.kind == 'mlp':
            pad = torch.zeros(B, s.k - 1, dtype=torch.long); xx = torch.cat([pad, x], 1)
            z = torch.cat([E[xx[:, i:i + L]] for i in range(s.k)], 2)
            if s.pos: z = torch.cat([z, (torch.arange(L) / 8.).view(1, L, 1).expand(B, L, 1)], 2)
            return s.l2(s.drop(torch.tanh(s.l1(z))))
        hh = torch.zeros(B, s.h); out = []
        for t in range(L):
            xi = E[x[:, t]]
            if s.kind == 'rnn': hh = torch.tanh(s.l1(torch.cat([xi, hh], 1)))
            else:
                c = torch.cat([xi, hh], 1); z = torch.sigmoid(s.lz(c)); r = torch.sigmoid(s.lr(c))
                n = torch.tanh(s.ln(torch.cat([xi, r * hh], 1))); hh = (1 - z) * hh + z * n
            out.append(s.l2(s.drop(hh)))
        return torch.stack(out, 1)

def evaluate(m, ns):
    X, Y = seq_data(ns); m.eval()
    with torch.no_grad(): return F.cross_entropy(m(X).reshape(-1, V), Y.reshape(-1), ignore_index=-1).item() / math.log(2)

def train(kind, e, h, M, steps=12000, k=3, extra=(), extra_w=0.5, drop=0.0, wd=1e-2, seed=0, quiet=False, pos=0):
    torch.manual_seed(seed)
    m = Net(kind, e, h, M, k, drop, pos)
    Xt, Yt = seq_data(tr); Xe, Ye = seq_data(list(extra)) if extra else (None, None)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-3, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time(); best = (9, None)
    for i in range(steps):
        m.train()
        if extra and random.random() < extra_w: ix = torch.randint(0, len(Xe), (128,)); xb, yb = Xe[ix], Ye[ix]
        else: ix = torch.randint(0, len(Xt), (128,)); xb, yb = Xt[ix], Yt[ix]
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1), ignore_index=-1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (i + 1) % 1000 == 0:
            v = evaluate(m, val)
            if v < best[0]: best = (v, {a: b.clone() for a, b in m.state_dict().items()})
    m.load_state_dict(best[1])
    n = sum(p.numel() for p in m.parameters())
    if not quiet: print(f"{kind:3} e={e} h={h:3} k={k} pos={pos} levels={2*M+1 if M else 'float':5} numbers={n:6} train={evaluate(m, tr):.3f} val={best[0]:.3f} bits/char ({time.time()-t0:.0f}s)", flush=True)
    return m, best[0]

@torch.no_grad()
def sample(m, n, temp=1.0, maxlen=60):
    m.eval(); out = []; E = m.emb if not m.M else quant(m.emb, m.M)
    for _ in range(n):
        if m.kind == 'mlp':
            ctx = [0] * m.k; s = ''
            while len(s) < maxlen:
                z = E[torch.tensor(ctx)].reshape(1, -1)
                if m.pos: z = torch.cat([z, torch.tensor([[len(s) / 8.]])], 1)
                x = torch.multinomial(F.softmax(m.l2(torch.tanh(m.l1(z)))[0] / temp, 0), 1).item()
                if x == 0: break
                s += chars[x]; ctx = ctx[1:] + [x]
            out.append(s); continue
        hh = torch.zeros(1, m.h); x = 0; s = ''
        while len(s) < maxlen:
            xi = E[torch.tensor([x])]
            if m.kind == 'rnn': hh = torch.tanh(m.l1(torch.cat([xi, hh], 1)))
            else:
                c = torch.cat([xi, hh], 1); z = torch.sigmoid(m.lz(c)); r = torch.sigmoid(m.lr(c))
                nn_ = torch.tanh(m.ln(torch.cat([xi, r * hh], 1))); hh = (1 - z) * hh + z * nn_
            x = torch.multinomial(F.softmax(m.l2(hh)[0] / temp, 0), 1).item()
            if x == 0: break
            s += chars[x]
        out.append(s)
    return out

if __name__ == '__main__':
    print('clean names', len(names), 'train', len(tr), 'val', len(val))
    train('mlp', 6, 24, 30, k=3)
    train('rnn', 6, 20, 30)
    train('rnn', 5, 22, 30)
    train('gru', 6, 13, 30)
    train('gru', 5, 14, 30)
