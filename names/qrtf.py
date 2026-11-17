"""Compact quantized causal transformers, MLP controls, exact QR export.

All selection uses the original train2.py validation split (reported explicitly).
Teacher targets are cached only for training names and teacher-generated samples.
Training runs and exports go to names/qrtf_runs/ (not tracked).

  uv run python names/qrtf.py train --help
  uv run python names/qrtf.py export names/qrtf_mlp_best.pt --output /tmp/names
"""
import argparse, copy, hashlib, json, math, os, random, shutil, subprocess, sys, time, types
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train2 import Net, chars, stoi, tr, val, seq_data, V, names
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for tools/
from tools.host import QR_PREFIX

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'names/qrtf_runs'
RUNS.mkdir(exist_ok=True)

def quant(w, levels, active=True):
    if not active: return w
    m = (levels - 1) // 2
    scale = w.detach().abs().max().clamp_min(1e-8) / m
    return w + ((w / scale).round().clamp(-m, m) * scale - w).detach()

class TinyTransformer(nn.Module):
    """Optional pre-normalization; causal attention and a tanh FFN.

    V/O are identity by default; value=True learns their combined projection.

    Each head sees its own contiguous value subspace. Query/key dimension can
    differ from value dimension. ALiBi slopes are fixed, as is 1/sqrt(qdim).
    The embedding is tied to the output. No dropout is used at inference.
    """
    def __init__(self, d=12, f=20, heads=2, q=2, levels=61, norm=False,
                 context=24, layers=1, slope=1., value=False, untie=False, window=1, sinpos=False):
        super().__init__()
        self.cfg = dict(d=d,f=f,heads=heads,q=q,levels=levels,norm=norm,
                        context=context,layers=layers,slope=slope,value=value,untie=untie,window=window,sinpos=sinpos)
        self.active = True
        self.emb = nn.Parameter(torch.randn(V,d)*.35)
        self.pos = nn.Parameter(torch.randn(d)*.05)
        self.qk = nn.Parameter(torch.randn(2*heads*q,d)/math.sqrt(d))
        if value: self.value = nn.Parameter(torch.eye(d)+torch.randn(d,d)*.05)
        self.w1 = nn.Parameter(torch.randn(f,d)/math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(f))
        self.w2 = nn.Parameter(torch.randn(d,f)*.1/math.sqrt(f))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.out = nn.Parameter(torch.zeros(V))
        if untie: self.output = nn.Parameter(torch.randn(V,d)/math.sqrt(d))
    def qw(self, w): return quant(w,self.cfg['levels'],self.active)
    def forward(self, x):
        c=self.cfg; B,L=x.shape; d,h,q=c['d'],c['heads'],c['q']
        E=self.qw(self.emb)
        if c['window']>1:
            k=c['window']; assert d%k==0
            xx=F.pad(x,(k-1,0)); w=d//k
            z=torch.cat([E[xx[:,i:i+L],i*w:(i+1)*w] for i in range(k)],-1)
        else: z=E[x]
        z=z+torch.arange(L,dtype=E.dtype)[None,:,None]/8*self.qw(self.pos)
        if c.get('sinpos'): z=z+.5*torch.sin(torch.arange(L,dtype=E.dtype)[:,None]/2**(torch.arange(d,dtype=E.dtype)/2))
        def norm(a): return a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5) if c['norm'] else a
        t=torch.arange(L); delta=t[:,None]-t[None,:]
        mask=(delta<0)|((delta>=c['context']) if c['context'] else False)
        slope=torch.tensor([c['slope']/(4**i) for i in range(h)])
        bias=-slope[:,None,None]*delta
        bias=bias.masked_fill(mask,-1e9)
        for _ in range(c['layers']):
            u=norm(z)
            kq=F.linear(u,self.qw(self.qk)).reshape(B,L,2,h,q)
            query,key=kq[:,:,0].transpose(1,2),kq[:,:,1].transpose(1,2)
            value=F.linear(u,self.qw(self.value)) if c['value'] else u
            value=value.reshape(B,L,h,d//h).transpose(1,2)
            a=torch.softmax(query@key.transpose(-1,-2)/math.sqrt(q)+bias,dim=-1)
            z=z+(a@value).transpose(1,2).reshape(B,L,d)
            u=norm(z)
            z=z+F.linear(torch.tanh(F.linear(u,self.qw(self.w1),self.qw(self.b1))),self.qw(self.w2),self.qw(self.b2))
        return F.linear(norm(z),self.qw(self.output) if c['untie'] else E,self.qw(self.out))

class QuantMLP(nn.Module):
    """Identical 3-character MLP, but quantizes biases during training too."""
    def __init__(self,e=6,h=24,levels=61,k=3):
        super().__init__(); self.cfg=dict(e=e,h=h,levels=levels,k=k); self.active=True
        self.emb=nn.Parameter(torch.randn(V,e)*.5)
        self.w1=nn.Parameter(torch.randn(h,k*e+1)/math.sqrt(k*e+1)); self.b1=nn.Parameter(torch.zeros(h))
        self.w2=nn.Parameter(torch.randn(V,h)/math.sqrt(h)); self.b2=nn.Parameter(torch.zeros(V))
    def qw(self,w): return quant(w,self.cfg['levels'],self.active)
    def forward(self,x):
        c=self.cfg; B,L=x.shape; E=self.qw(self.emb)
        xx=F.pad(x,(c['k']-1,0))
        z=torch.cat([E[xx[:,i:i+L]] for i in range(c['k'])]+[(torch.arange(L)/8)[None,:,None].expand(B,L,1)],-1)
        return F.linear(torch.tanh(F.linear(z,self.qw(self.w1),self.qw(self.b1))),self.qw(self.w2),self.qw(self.b2))

class NgramAttention(QuantMLP):
    """Explicit hybrid control: learned causal attention over 3-gram features.

    This tests whether any gain is really from attention, separately from the
    narrower standard token-embedding transformers. It is labeled hybrid.
    """
    def __init__(self,e=6,h=24,levels=31,k=3,q=1,slope=.5):
        super().__init__(e,h,levels,k)
        self.cfg.update(q=q,slope=slope)
        self.qk=nn.Parameter(torch.randn(2*q,k*e+1)*.05)
        self.gate=nn.Parameter(torch.tensor([.1]))
    def forward(self,x):
        c=self.cfg;B,L=x.shape;E=self.qw(self.emb);xx=F.pad(x,(c['k']-1,0))
        z=torch.cat([E[xx[:,i:i+L]] for i in range(c['k'])]+[(torch.arange(L)/8)[None,:,None].expand(B,L,1)],-1)
        qk=F.linear(z,self.qw(self.qk));q,k=qk.chunk(2,-1)
        p=torch.arange(L);delta=p[:,None]-p[None,:]
        score=q@k.transpose(-1,-2)/math.sqrt(c['q'])-c['slope']*delta
        score=score.masked_fill(delta<0,-1e9)
        z=z+self.qw(self.gate)*(score.softmax(-1)@z)
        return F.linear(torch.tanh(F.linear(z,self.qw(self.w1),self.qw(self.b1))),self.qw(self.w2),self.qw(self.b2))

class TapTransformer(QuantMLP):
    """Structured width-h transformer with two learned causal attention heads.

    Embeddings occupy a six-dimensional subspace. Identity value projections
    write two heads into distinct subspaces; the residual writes the current
    embedding into a third. Fixed Gaussian relative biases prefer distances
    2 and 1, but content-dependent q/k scores can attend elsewhere. A sparse
    expansion/tanh FFN with identity second projection has a .1 residual, then
    an untied character classifier. Empty width-h channels cost no parameters.
    This is deliberately an explicitly local transformer control.
    """
    def __init__(self,e=6,h=24,levels=31,k=3,q=1,slope=6.,skip=.1):
        assert k==3 and q==1 and h>=3*e+1
        super().__init__(e,h,levels,k);self.cfg.update(q=q,slope=slope,skip=skip)
        self.qk=nn.Parameter(torch.randn(4,e)*.01)
    def forward(self,x):
        c=self.cfg;B,L=x.shape;E=self.qw(self.emb);z=E[x]
        qk=F.linear(z,self.qw(self.qk));q,k=qk[...,:2],qk[...,2:]
        t=torch.arange(L);delta=t[:,None]-t[None,:];contexts=[]
        for j in range(2):
            score=q[...,j,None]*k[:,None,:,j]-c['slope']*(delta-(2-j))**2
            score=score.masked_fill(delta<0,-1e9)
            contexts.append(score.softmax(-1)@z)
        features=torch.cat(contexts+[z,(t/8)[None,:,None].expand(B,L,1)],-1)
        H=torch.tanh(F.linear(features,self.qw(self.w1),self.qw(self.b1)))+c['skip']*F.pad(features,(0,c['h']-features.shape[-1]))
        return F.linear(H,self.qw(self.w2),self.qw(self.b2))

def create(kind,cfg):
    cfg=dict(cfg);lsq=cfg.pop('lsq',False)
    m=TinyTransformer(**cfg) if kind=='tf' else NgramAttention(**cfg) if kind=='hybrid' else TapTransformer(**cfg) if kind=='tap' else QuantMLP(**cfg)
    if lsq:
        m.cfg['lsq']=True
        original=list(m.named_parameters())
        m.scales=nn.ParameterDict({n:nn.Parameter((p.detach().abs().max().clamp_min(1e-4)/((cfg['levels']-1)//2)).log()) for n,p in original})
        mapping={id(p):n for n,p in original}
        def learned_qw(self,w):
            if not self.active:return w
            sc=self.scales[mapping[id(w)]].exp();v=w/sc;v=v+(v.round()-v).detach()
            return v.clamp(-(self.cfg['levels']-1)//2,(self.cfg['levels']-1)//2)*sc
        m.qw=types.MethodType(learned_qw,m)
    return m

def teacher_cache():
    path=RUNS/'teacher_targets.pt'
    extra=json.load(open('names/teacher_samples.json'))
    assert not set(extra)&set(val)
    ns=tr+extra
    digest=hashlib.sha256(json.dumps(ns).encode()+Path('names/teacher.pt').read_bytes()).hexdigest()
    if path.exists():
        cached=torch.load(path,weights_only=False)
        if cached['digest']==digest: return cached
    rng_state=torch.get_rng_state();cache_threads=torch.get_num_threads();torch.set_num_threads(4)
    T=Net('gru',16,96,0,3,.3); T.load_state_dict(torch.load('names/teacher.pt',weights_only=True)); T.eval()
    X,Y=seq_data(ns); targets=[]
    with torch.no_grad():
        for xb in X.split(128): targets.append(T(xb).softmax(-1).half())
    cached=dict(X=X,Y=Y,P=torch.cat(targets),digest=digest,nreal=len(tr),nextra=len(extra))
    torch.save(cached,path)
    torch.set_rng_state(rng_state);torch.set_num_threads(cache_threads)
    return cached

@torch.no_grad()
def evaluate(m,ns):
    m.eval(); X,Y=seq_data(ns); total=n=0
    for xb,yb in zip(X.split(128),Y.split(128)):
        total+=F.cross_entropy(m(xb).reshape(-1,V),yb.reshape(-1),ignore_index=-1,reduction='sum').item()
        n+=(yb>=0).sum().item()
    return total/n/math.log(2)

def train(args):
    if args.levels<3 or args.levels%2!=1:raise ValueError('levels must be odd and >=3')
    if not 1<=args.threads<=4:raise ValueError('Use 1-4 Torch threads')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.kind=='tf': cfg=dict(d=args.width,f=args.ff,heads=args.heads,q=args.q,levels=args.levels,norm=args.norm,context=args.context,layers=args.layers,slope=args.slope,value=args.value,untie=args.untie,window=args.window,sinpos=args.sinpos)
    else:
        cfg=dict(e=args.width,h=args.ff,levels=args.levels,k=args.context)
        if args.kind in ('hybrid','tap'): cfg.update(q=args.q,slope=args.slope)
        if args.kind=='tap':cfg['skip']=args.skip
    if args.lsq:cfg['lsq']=True
    m=create(args.kind,cfg)
    if args.init:
        saved=torch.load(args.init,weights_only=False)
        if 'state' in saved: m.load_state_dict(saved['state'])
        else:
            state={{'l1.w':'w1','l1.b':'b1','l2.w':'w2','l2.b':'b2'}.get(k,k):v for k,v in saved.items()}
            if state['w1'].shape[0]!=args.ff:
                importance=state['w1'].norm(dim=1)*state['w2'].norm(dim=0)
                if args.ff<state['w1'].shape[0]:
                    keep=importance.topk(args.ff).indices.sort().values
                    state['w1']=state['w1'][keep];state['b1']=state['b1'][keep];state['w2']=state['w2'][:,keep]
                else:
                    oldh=state['w1'].shape[0];state['w1']=torch.cat([state['w1'],m.w1.detach()[oldh:]],0);state['b1']=torch.cat([state['b1'],m.b1.detach()[oldh:]],0);state['w2']=F.pad(state['w2'],(0,args.ff-oldh))
            m.load_state_dict(state,strict=(args.kind not in ('hybrid','tap') and not args.lsq))
            if args.lsq:
                with torch.no_grad():
                    for n,p in m.named_parameters():
                        if not n.startswith('scales.'):m.scales[n].copy_((p.abs().max().clamp_min(1e-4)/((args.levels-1)//2)).log())
    if args.attention_only:
        assert args.kind in ('tap','tf','hybrid')
        for n,p in m.named_parameters():p.requires_grad_(n=='qk' or n=='scales.qk')
    cache=teacher_cache()
    if args.legacy_cache_rng:Net('gru',16,96,0,3,.3)
    X,Y,P=cache['X'],cache['Y'],cache['P']; nr=cache['nreal']
    groups=[{'params':[p for n,p in m.named_parameters() if not n.startswith('scales.')]}]
    if cfg.get('lsq'):groups.append({'params':m.scales.parameters(),'weight_decay':0.})
    opt=torch.optim.AdamW(groups,lr=args.lr,weight_decay=args.wd)
    start=time.monotonic();m.active=True
    if args.no_initial:
        best=(float('inf'),None);hist=[]
    else:
        initial=evaluate(m,val);best=(initial,copy.deepcopy(m.state_dict()))
        hist=[dict(step=0,val_bpc=initial,seconds=round(time.monotonic()-start,2))]
        torch.save(dict(kind=args.kind,cfg=cfg,state=best[1],score=initial,args=vars(args),history=hist),RUNS/(args.tag+'.pt'))
        print(args.tag,'initial',initial,flush=True)
    for i in range(args.steps):
        if time.monotonic()-start>args.max_seconds: print('Time limit: stopping',flush=True); break
        m.train(); m.active=(i>=args.warmup)
        if cfg.get('lsq') and args.warmup and i==args.warmup:
            with torch.no_grad():
                for n,p in m.named_parameters():
                    if not n.startswith('scales.'):m.scales[n].copy_((p.abs().max().clamp_min(1e-4)/((args.levels-1)//2)).log())
        nreal=round(args.batch*(1-args.extra)); ne=args.batch-nreal
        ix=torch.cat([torch.randint(nr,(nreal,)),torch.randint(nr,len(X),(ne,))])
        xb,yb=X[ix],Y[ix]; logits=m(xb); valid=yb>=0
        hard=F.cross_entropy(logits.reshape(-1,V),yb.reshape(-1),ignore_index=-1)
        soft=-(P[ix].float()*F.log_softmax(logits,-1)).sum(-1)[valid].mean()
        loss=(1-args.soft)*hard+args.soft*soft
        lr=args.lr*(.04+.96*(1+math.cos(math.pi*i/args.steps))/2)
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if (i+1)%args.eval_every==0 or i+1==args.steps:
            m.active=True; score=evaluate(m,val)
            entry=dict(step=i+1,val_bpc=score,loss=float(loss.detach()),seconds=round(time.monotonic()-start,2))
            hist.append(entry); print(args.tag,json.dumps(entry),flush=True)
            if score<best[0]:
                best=(score,copy.deepcopy(m.state_dict()))
                torch.save(dict(kind=args.kind,cfg=cfg,state=best[1],score=score,args=vars(args),history=hist),RUNS/(args.tag+'.pt'))
    if best[1] is None: raise RuntimeError('No checkpoint evaluated')
    m.load_state_dict(best[1]); m.active=True
    summary=dict(tag=args.tag,kind=args.kind,cfg=cfg,params=sum(p.numel() for p in m.parameters()),best_val_bpc=best[0],seconds=time.monotonic()-start,history=hist,args=vars(args))
    (RUNS/(args.tag+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('DONE',json.dumps({k:v for k,v in summary.items() if k not in ('history','args')}),flush=True)

def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--tag',required=True); t.add_argument('--kind',choices=['tf','mlp','hybrid','tap'],default='tf')
    for name,default,typ in [('threads',2,int),('width',12,int),('ff',20,int),('heads',2,int),('q',2,int),('levels',61,int),('context',24,int),('layers',1,int),('window',1,int),('slope',1.,float),('skip',.1,float),('steps',10000,int),('eval-every',1000,int),('batch',64,int),('warmup',500,int),('seed',0,int),('lr',.004,float),('wd',.01,float),('soft',.7,float),('extra',.6,float),('max-seconds',870,int)]: t.add_argument('--'+name,type=typ,default=default)
    for name in ['norm','value','untie','sinpos','lsq']: t.add_argument('--'+name,action='store_true')
    t.add_argument('--init');t.add_argument('--no-initial',action='store_true');t.add_argument('--legacy-cache-rng',action='store_true');t.add_argument('--attention-only',action='store_true')
    z=sub.add_parser('transfer');z.add_argument('checkpoint');z.add_argument('--tag',required=True);z.add_argument('--levels',type=int,required=True);z.add_argument('--ff',type=int);z.add_argument('--seed',type=int,default=0);z.add_argument('--slope',type=float,default=6.)
    e=sub.add_parser('export');e.add_argument('checkpoint');e.add_argument('--output');e.add_argument('--levels',type=int);e.add_argument('--publish',choices=['tf','mlp'])
    return p


# Export helpers are intentionally ordinary Python; only generated JS is golfed.
def literal(x):
    s=format(float(x),'.4g')
    return s[1:] if s.startswith('0.') else s

def runtime_tf(cfg,parts):
    d,h,q,f=cfg['d'],cfg['heads'],cfg['q'],cfg['f']
    # Decode one mixed-radix integer backwards, dequantizing each tensor with the
    # exact scale literal embedded here. Tensor metadata is part of the program.
    uniform=len({p[3] for p in parts})==1
    specs=','.join(f'[{arr.size},{literal(scale)}]' if uniform else f'[{arr.size},{levels},{literal(scale)}]' for _,arr,scale,levels in parts)
    varnames=','.join({'emb':'e','pos':'P','qk':'Q','value':'D','w1':'A','b1':'a','w2':'B','b2':'b','out':'c','output':'O'}[name] for name,*_ in parts)
    levels=parts[0][3]
    if uniform:
        decoder=f"Number(n-(n/={levels}n)*{levels}n)-{(levels-1)//2}"
        spec_args='[k,s]'
    else:
        decoder="Number(n-(n/=BigInt(r))*BigInt(r))-(r-1)/2"
        spec_args='[k,r,s]'
    decode=f"with(Math){{n=BigInt(d);[{varnames}]=[{specs}].map(({spec_args})=>Array(k).fill().map(_=>({decoder})*s));"
    linear="l=(x,w,b=0)=>w.slice(0,w.length/x.length).map((_,j)=>x.reduce((s,v,i)=>s+v*w[j*x.length+i],b[j]||0));"
    softmax="s=x=>{let t=0;return x=x.map(v=>(t+=v=exp(v),v)),x.map(v=>v/t)};"
    norm="r=x=>x.map(v=>v/sqrt(x.reduce((s,v)=>s+v*v,0)/x.length+.00001));" if cfg['norm'] else ''
    normexpr=lambda name:f'r({name})' if cfg['norm'] else name
    code=decode+linear+softmax+norm
    code+=f"f=(x,p)=>{{z=e.slice(x*{d},x*{d}+{d}).map((v,i)=>v+P[i]*p/8);"
    if cfg['layers']>1: code+=f"for(let J=0;J<{cfg['layers']};J++){{K=KK[J];U=UU[J];"
    code+=f"u={normexpr('z')};q=l(u,Q);K[p]=q.slice({h*q});U[p]="+("l(u,D)" if cfg['value'] else "u")+";"
    mask=f"i<=p-{cfg['context']}?-1e9:" if cfg['context'] else ''
    if h==1:
        code+=f"t=s(K.map((k,i)=>{mask}k.reduce((a,v,j)=>a+v*q[j],0)/{("sqrt("+str(q)+")") if q!=int(math.sqrt(q))**2 else str(int(math.sqrt(q)))}-{literal(cfg['slope'])}*(p-i)));"
    else:
        code+=f"t={str(list(range(h))).replace(' ', '')}.map(j=>s(K.map((k,i)=>{mask}q.slice(j*{q},j*{q}+{q}).reduce((a,v,t)=>a+v*k[j*{q}+t],0)/{("sqrt("+str(q)+")") if q!=int(math.sqrt(q))**2 else str(int(math.sqrt(q)))}-{literal(cfg['slope'])}*(p-i)/4**j)));"
    at='t[k]' if h==1 else f't[i/{d//h}|0][k]'
    code+=f"z=z.map((v,i)=>v+U.reduce((a,u,k)=>a+{at}*u[i],0));u=l(l({normexpr('z')},A,a).map(tanh),B,b);z=z.map((v,i)=>v+u[i]);"
    if cfg['layers']>1: code+="}"
    code+=f"return s(l({normexpr('z')},"+('O' if cfg['untie'] else 'e')+",c))};"
    code+="g=()=>{K=[];U=[];for(let o='',x=0;;){let a=f(x,o.length).map(v=>v**1.25),r=random()*a.reduce((s,v)=>s+v);for(x=0;(r-=a[x])>0;x++);if(!x)return o;o+='-.'[x-1]||(x-3).toString(36)}}}"
    if cfg['layers']>1:
        init='KK=['+','.join('[]' for _ in range(cfg['layers']))+'];UU=['+','.join('[]' for _ in range(cfg['layers']))+'];'
        code=code.replace('K=[];U=[];',init)
    if cfg.get('window',1)>1:
        win=cfg['window']; sub=d//win
        old=f"z=e.slice(x*{d},x*{d}+{d}).map((v,i)=>v+P[i]*p/8)"
        new=f"R.push(x);z=P.map((v,i)=>e[(R[max(0,p-{win-1}+(i/{sub}|0))]||0)*{d}+i]+v*p/8)"
        # max(0,..) must be padding 0, not the first nonzero token. R[0] is BOS=0.
        code=code.replace(old,new).replace('g=()=>{','g=()=>{R=[];')
    if cfg.get('sinpos'): code=code.replace('v+P[i]*p/8','v+P[i]*p/8+.5*sin(p/2**(i/2))')
    return code.replace('g=()=>{','g=_=>{')

def mlp_literal(x):
    s=format(float(x),'.3g')
    return s[1:] if s.startswith('0.') else s

def runtime_mlp(cfg,parts):
    e,h,k=cfg['e'],cfg['h'],cfg['k'];r=k*e+1;ke=k*e
    off1=V*e;offb=off1+h*r;off2=offb+h;offc=off2+V*h
    data={name:(arr,scale) for name,arr,scale,_ in parts}
    se,s1,sb1,s2,sb2=[data[n][1] for n in ['emb','w1','b1','w2','b2']]
    base=cfg['levels'];M=(base-1)//2;n=sum(a.size for _,a,_,_ in parts)
    code=f"for(n=BigInt(d),W=[],i={n};i--;)W[i]=Number(n-(n/={base}n)*{base}n)-{M};"
    code+="g=_=>{with(Math)for(x=["+','.join('0' for _ in range(k))+"],o='';;){"
    code+=f"for(H=[],j=0;j<{h};j++)for(a=W[{offb}+j]*{mlp_literal(sb1)}+W[{off1+ke}+j*{r}]*o.length*{mlp_literal(s1/8)},i=0;i<{ke};i++)H[j]=tanh(a+=W[{off1}+j*{r}+i]*W[x[t=i/{e}|0]*{e}+i-t*{e}]*{mlp_literal(se*s1)});"
    code+=f"for(p=[],t=j=0;j<{V};j++){{for(a=W[{offc}+j]*{mlp_literal(sb2*1.25)},i=0;i<{h};i++)a+=W[{off2}+j*{h}+i]*H[i]*{mlp_literal(s2*1.25)};t+=p[j]=exp(a)}}"
    code+="for(r=random()*t,j=0;(r-=p[j])>0;j++);if(!j)return o;o+='-.'[j-1]||(j-3).toString(36);x=["+','.join(f'x[{i}]' for i in range(1,k))+",j]}}"
    return code.replace('g=()=>{','g=_=>{')

def runtime_hybrid(cfg,parts):
    assert cfg['q']==1
    e,h,k=cfg['e'],cfg['h'],cfg['k'];r=k*e+1
    off1=V*e;offb=off1+h*r;off2=offb+h;offc=off2+V*h;offq=offc+V;offg=offq+2*r
    data={n:(a,sc) for n,a,sc,_ in parts}
    se,s1,sb1,s2,sb2,sq,sg=[data[n][1] for n in ['emb','w1','b1','w2','b2','qk','gate']]
    base=cfg['levels'];M=(base-1)//2;n=sum(a.size for _,a,_,_ in parts)
    code=f"for(n=BigInt(d),W=[],i={n};i--;)W[i]=Number(n-(n/={base}n)*{base}n)-{M};"
    code+=f"f=(x,y)=>{{let z=x.flatMap(t=>W.slice(t*{e},t*{e}+{e}).map(v=>v*{literal(se)})).concat(y/8),q=0,k=0,i,j,a,H,p,t,s;"
    code+=f"for(i=0;i<{r};i++)q+=W[{offq}+i]*z[i]*{literal(sq)},k+=W[{offq+r}+i]*z[i]*{literal(sq)};K.push(k);U.push(z);"
    code+=f"t=K.map((v,i)=>Math.exp(q*v-{literal(cfg['slope'])}*(y-i)));s=t.reduce((a,b)=>a+b);z=z.map((v,i)=>v+W[{offg}]*{literal(sg)}*U.reduce((a,u,j)=>a+t[j]*u[i]/s,0));"
    code+=f"for(H=[],j=0;j<{h};j++){{for(a=W[{offb}+j]*{literal(sb1)},i=0;i<{r};i++)a+=W[{off1}+j*{r}+i]*z[i]*{literal(s1)};H[j]=Math.tanh(a)}}"
    code+=f"for(p=[],j=0;j<{V};j++){{for(a=W[{offc}+j]*{literal(sb2)},i=0;i<{h};i++)a+=W[{off2}+j*{h}+i]*H[i]*{literal(s2)};p[j]=Math.exp(a)}}return p}};"
    code+=f"g=()=>{{K=[];U=[];for(let x=Array({k}).fill(0),o='';;){{let a=f(x,o.length).map(v=>v**1.25),r=Math.random()*a.reduce((s,v)=>s+v),j=0;for(;(r-=a[j])>0;j++);if(!j)return o;o+='-.'[j-1]||(j-3).toString(36);x=x.slice(1).concat(j)}}}}"
    return code.replace('g=()=>{','g=_=>{')

def runtime_tap(cfg,parts):
    e,h=cfg['e'],cfg['h'];r=3*e+1
    off1=V*e;offb=off1+h*r;off2=offb+h;offc=off2+V*h;offq=offc+V
    data={n:(a,sc) for n,a,sc,_ in parts};se,s1,sb1,s2,sb2,sq=[data[n][1] for n in ['emb','w1','b1','w2','b2','qk']]
    base=cfg['levels'];M=(base-1)//2;n=sum(a.size for _,a,_,_ in parts)
    code=f"with(Math){{for(n=BigInt(d),W=[],i={n};i--;)W[i]=Number(n-(n/={base}n)*{base}n)-{M};"
    code+=f"f=(x,y)=>{{let z=W.slice(x*{e},x*{e}+{e}).map(v=>v*{literal(se)}),q=[],i,j,a,H,p,t;"
    code+=f"for(j=0;j<4;j++){{for(a=0,i=0;i<{e};i++)a+=W[{offq}+j*{e}+i]*z[i]*{literal(sq)};q[j]=a}}K.push(q.slice(2));U.push(z);"
    code+=f"t=[0,1].flatMap(j=>{{let a=K.map((k,i)=>exp(q[j]*k[j]-{literal(cfg['slope'])}*(y-i-2+j)**2)),s=a.reduce((a,b)=>a+b);return z.map((_,i)=>U.reduce((s,u,k)=>s+a[k]*u[i],0)/s)}}).concat(z,y/8);"
    code+=f"for(H=[],j=0;j<{h};j++){{for(a=W[{offb}+j]*{literal(sb1)},i=0;i<{r};i++)a+=W[{off1}+j*{r}+i]*t[i]*{literal(s1)};H[j]=tanh(a)+.1*(t[j]||0)}}"
    code+=f"for(p=[],j=0;j<{V};j++){{for(a=W[{offc}+j]*{literal(sb2)},i=0;i<{h};i++)a+=W[{off2}+j*{h}+i]*H[i]*{literal(s2)};p[j]=exp(a)}}return p}};"
    code+="g=()=>{K=[];U=[];for(let o='',x=0;;){let a=f(x,o.length).map(v=>v**1.25),r=random()*a.reduce((s,v)=>s+v);for(x=0;(r-=a[x])>0;x++);if(!x)return o;o+='-.'[x-1]||(x-3).toString(36)}}}"
    code=code.replace(f'W.slice(x*{e},x*{e}+{e})',f'W.slice(x*={e},x+{e})')
    code=code.replace(',q=[],i,j,a,H,p,t;',',q,i,j,a,H,p,t;')
    code=code.replace('K.push(q.slice(2));U.push(z);','K[y]=q.slice(2);U[y]=z;')
    code=code.replace('for(let o=','for(o=').replace('let a=f(x,o.length).map(v=>v**1.25),','a=f(x,o.length),')
    code=code.replace('p[j]=exp(a)','p[j]=exp(a*1.25)')
    old=f'for(j=0;j<4;j++){{for(a=0,i=0;i<{e};i++)a+=W[{offq}+j*{e}+i]*z[i]*{literal(sq)};q[j]=a}}'
    new=f'q=[0,0,0,0];for(i=0;i<{4*e};i++)q[j=i/{e}|0]+=W[{offq}+i]*z[i-j*{e}]*{literal(sq)};'
    code=code.replace(old,new)
    code=code.replace('+.1*(t[j]||0)',('+'+literal(cfg.get('skip',.1))+'*(t[j]||0)') if cfg.get('skip',.1) else '')
    # Inline the forward pass into g(). The verifier observes the exact p array
    # through Math.random, then forces each validation character; no f wrapper
    # needs to consume QR bytes. This is the same hook used for the published MLP.
    begin=code.index('f=(x,y)=>{');end=code.index('return p};',begin)
    body=code[begin+len('f=(x,y)=>{'):end].replace(',H,p,t;',',H,t;')
    body=body.replace('let z=','z=',1)
    code=code[:begin]+"g=()=>{K=[];U=[];for(o='',x=0;;){let y=o.length,"+body
    code+="for(r=random()*p.reduce((s,v)=>s+v),x=0;(r-=p[x])>0;x++);if(!x)return o;o+='-.'[x-1]||(x-3).toString(36)}}}"
    return code.replace('g=()=>{','g=_=>{')

def transfer(args):
    """Move a QAT MLP to two actual attention heads, preserving learned ranges."""
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,weights_only=False);assert saved['kind']=='mlp'
    source=saved['cfg'];h=args.ff or source['h'];lsq=source.get('lsq',False)
    cfg=dict(e=source['e'],h=h,k=3,q=1,slope=args.slope,skip=0.,levels=args.levels)
    if lsq:cfg['lsq']=True
    m=create('tap',cfg);state={n:p.clone() for n,p in saved['state'].items() if not n.startswith('scales.')}
    oldh=source['h']
    if h<oldh:
        keep=(state['w1'].norm(dim=1)*state['w2'].norm(dim=0)).topk(h).indices.sort().values
        state['w1']=state['w1'][keep];state['b1']=state['b1'][keep];state['w2']=state['w2'][:,keep]
    elif h>oldh:
        state['w1']=torch.cat([state['w1'],m.w1.detach()[oldh:]],0);state['b1']=F.pad(state['b1'],(0,h-oldh));state['w2']=F.pad(state['w2'],(0,h-oldh))
    m.load_state_dict(state,strict=False)
    if lsq:
        with torch.no_grad():
            for n,sc in m.scales.items():
                key='scales.'+n
                if key in saved['state']:sc.copy_(saved['state'][key]+math.log((source['levels']-1)/(args.levels-1)))
    score=evaluate(m,val);history=[dict(step=0,val_bpc=score,seconds=0)]
    torch.save(dict(kind='tap',cfg=cfg,state=m.state_dict(),score=score,args=vars(args),history=history),RUNS/(args.tag+'.pt'))
    summary=dict(tag=args.tag,kind='tap',cfg=cfg,params=sum(p.numel() for p in m.parameters()),best_val_bpc=score,seconds=0.,history=history,args=vars(args))
    (RUNS/(args.tag+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2),flush=True)

def readable_js(source):
    # Whitespace only: preserve strings, operators, expression order and ASI.
    out=[];indent=0;quote=None;escaped=False;line=''
    for ch in source:
        if quote:
            line+=ch
            if escaped:escaped=False
            elif ch=='\\':escaped=True
            elif ch==quote:quote=None
        elif ch in ("'",'"','`'):
            quote=ch;line+=ch
        elif ch=='}':
            if line.strip():out.append('  '*indent+line.strip());line=''
            indent=max(0,indent-1);out.append('  '*indent+'}')
        elif ch in '{;':
            line+=ch;out.append('  '*indent+line.strip());line=''
            if ch=='{':indent+=1
        else:line+=ch
    if line.strip():out.append('  '*indent+line.strip())
    return '\n'.join(out)+'\n'

def publish(stem,checkpoint,row,kind):
    if 'export_error' in row:raise RuntimeError(row['export_error'])
    if row['qr_version']>31:raise RuntimeError('Published QR exceeds version 31')
    suffix='qrtf' if kind=='tf' else 'mlp_qrtf'
    shutil.copyfile(stem+'.url.txt','names/payload_url_'+suffix+'.txt')
    source=Path(stem+'.golf.js').read_text()
    Path('names/runtime_'+suffix+'.js').write_text('// Exact executable, expanded with whitespace only. See docs/RESULTS_qrtf.md.\n'+readable_js(source))
    Path('names/verification_'+suffix+'.json').write_text(json.dumps(row,indent=2)+'\n')
    target='names/qrtf_best.pt' if kind=='tf' else 'names/qrtf_mlp_best.pt'
    if Path(checkpoint).resolve()!=Path(target).resolve():shutil.copyfile(checkpoint,target)

def export(args):
    from tools.qr import qr_measure, zxing_ok
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,weights_only=False)
    kind,cfg=saved['kind'],dict(saved['cfg'])
    if args.levels: cfg['levels']=args.levels
    m=create(kind,cfg);m.load_state_dict(saved['state']);m.eval()
    if cfg.get('lsq') and args.levels and args.levels!=saved['cfg']['levels']:
        with torch.no_grad():
            for sc in m.scales.values():sc.add_(math.log((saved['cfg']['levels']-1)/(args.levels-1)))
    levels=cfg['levels'];M=(levels-1)//2
    parts=[];packed=1;dequant={}
    for name,p in m.named_parameters():
        if name.startswith('scales.'):
            dequant[name]=p.detach().double();continue
        w=p.detach().numpy();scale=float(m.scales[name].detach().exp()) if cfg.get('lsq') else max(float(np.abs(w).max()),1e-8)/M
        ints=np.rint(w/scale).clip(-M,M).astype(np.int32)
        scale=float(literal(scale))
        parts.append((name,ints,scale,levels))
        dequant[name]=torch.from_numpy(ints.astype(np.float64)*scale)
        for n in ints.flat: packed=packed*levels+int(n)+M
    if kind=='mlp':
        partmap={n:(a,s) for n,a,s,_ in parts}
        se,s1=partmap['emb'][1],partmap['w1'][1]
        dequant['emb']=torch.from_numpy(partmap['emb'][0].astype(np.float64))
        w1=partmap['w1'][0].astype(np.float64);w1[:,:-1]*=float(mlp_literal(se*s1));w1[:,-1]*=float(mlp_literal(s1/8))*8
        dequant['w1']=torch.from_numpy(w1)
        dequant['b1']=torch.from_numpy(partmap['b1'][0].astype(np.float64)*float(mlp_literal(partmap['b1'][1])))
        for n in ['w2','b2']:dequant[n]=torch.from_numpy(partmap[n][0].astype(np.float64)*float(mlp_literal(partmap[n][1]*1.25))/1.25)
    m.double();m.load_state_dict(dequant);m.active=False
    if kind=='tf':
        packed=1
        flat=[int(n) for _,a,_,_ in parts for n in a.flat]
        for n in reversed(flat): packed=packed*levels+n+M
    weights=str(packed)
    if kind=='tf': js=runtime_tf(cfg,parts)
    elif kind=='mlp': js=runtime_mlp(cfg,parts)
    elif kind=='hybrid': js=runtime_hybrid(cfg,parts)
    else: js=runtime_tap(cfg,parts)
    assert js.isascii() and not any(c in js for c in '%#'), 'Unsafe raw URL program'
    url=QR_PREFIX+f'{len(js):03d}'+js+weights
    qr,version,used,cap,modes=qr_measure(url,20)
    row=dict(url_sha256=hashlib.sha256(url.encode('ascii')).hexdigest(),model=saved.get('args',{}).get('tag',Path(args.checkpoint).stem),kind=kind,config=cfg,params=sum(a.size for _,a,_,_ in parts),levels=levels,
             program_chars=len(js),weight_digits=len(weights),payload_bytes=len(url)-len(QR_PREFIX),url_chars=len(url),
             qr_version=version,data_bits=used,slack_bits=cap-used,segments=modes,val_bpc=evaluate(m,val),
             tensors=[dict(name=n,shape=list(a.shape),scale=sc,levels=lv) for n,a,sc,lv in parts])
    stem=args.output or ('names/qrtf_runs/'+row['model'])
    Path(stem+'.url.txt').write_text(url)
    Path(stem+'.golf.js').write_text(js)
    if len(js)>=1000: row['export_error']='program exceeds loader 3-digit length'
    else:
        zxing_ok(qr,url,stem+'.png')
        row['zxing_exact']=True
        X,Y=seq_data(val)
        with torch.no_grad(): logp=m(X).log_softmax(-1)
        fixture=dict(kind=kind,config=cfg,names=val,chars=chars,logprobs=[logp[i,:len(n)+1].tolist() for i,n in enumerate(val)],
                     all_names=names,runtime_legacy=(kind in ('tap','mlp')),runtime_temperature=(.8 if kind in ('tap','mlp') else 1),params=row['params'],weights=([int(x) for _,a,_,_ in parts for x in a.flat][::-1] if kind=='tf' else [int(x) for _,a,_,_ in parts for x in a.flat]),row=row)
        Path(stem+'.fixture.json').write_text(json.dumps(fixture,separators=(',',':')))
        checked=subprocess.run(['node','names/verify_qrtf.js',stem+'.url.txt',stem+'.fixture.json'],capture_output=True,text=True,timeout=600)
        if checked.returncode: raise RuntimeError(checked.stderr)
        result=json.loads(checked.stdout);row.update(result)
        assert result['max_logprob_error']<1e-9,result['max_logprob_error']
    Path(stem+'.json').write_text(json.dumps(row,indent=2)+'\n')
    if args.publish:publish(stem,args.checkpoint,row,args.publish)
    print(json.dumps(row,indent=2),flush=True)

if __name__=='__main__':
    args=parser().parse_args()
    if args.command=='train': train(args)
    elif args.command=='export': export(args)
    elif args.command=='transfer': transfer(args)
