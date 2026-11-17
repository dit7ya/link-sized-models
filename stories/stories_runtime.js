[D,L,H,T,B,Z,J]=C;let M=(1<<B-1)-1,V=X.length+1,A=0,n=0,P=0,k=0,G=0,hd=D/H,
g=(b=B)=>{for(;n<b;n+=6){let c=d.charCodeAt(P++);A=A<<6|(c>96?c-71:c>94?63:c>64?c-65:c>47?c+4:62)}n-=b;let v=A>>n;A&=(1<<n)-1;return v},
U=J?Array.from({length:V+L*9*D},_=>g(8)):[],u=0;
A=n=0;let m=(r,c)=>{let w=new Float32Array(r*c),s=K[k++];for(let i=0;i<r;i++){let z=s*(J?2**((U[u++]-128)/32):1);for(let j=0;j<c;j++)w[i*c+j]=(g()-M)*z}return w},
mv=(w,x)=>{let c=x.length,r=w.length/c,y=new Float32Array(r);for(let i=0;i<r;i++){let s=0,o=i*c;for(let j=0;j<c;j++)s+=w[o+j]*x[j];y[i]=s}return y},
rn=x=>{let s=0;for(let v of x)s+=v*v;s=1/Math.sqrt(s/x.length+1e-5);return x.map(v=>v*s)},
ad=(x,y)=>x.map((v,i)=>v+y[i]),
E=m(V,D),W=[],Q;for(let l=0;l<L;l++)W.push([m(3*D,D),m(D,D),m(4*D,D),m(D,4*D)]);
R=_=>{Q=W.map(_=>[])};
S=t=>{let x=E.slice(t*D,t*D+D);W.forEach(([a,b,c,e],l)=>{let u=mv(a,rn(x)),q=Q[l],o=new Float32Array(D);q.push(u);q.length>T&&q.shift();let N=q.length;
for(let h=0;h<H;h++){let f=h*hd,sl=2**(-8*(h+1)/H),sc=[],mx=-1e9,z=0;for(let j=0;j<N;j++){let s=0;for(let i=0;i<hd;i++)s+=u[f+i]*q[j][D+f+i];s=s/Math.sqrt(hd)-sl*(N-1-j);sc.push(s);mx=Math.max(mx,s)}
sc=sc.map(s=>(s=Math.exp(s-mx),z+=s,s));for(let j=0;j<N;j++)for(let i=0;i<hd;i++)o[f+i]+=sc[j]/z*q[j][2*D+f+i]}
x=ad(x,mv(b,o));x=ad(x,mv(e,mv(c,rn(x)).map(v=>.5*v*(1+Math.tanh(.7978845608*(v+.044715*v*v*v))))))});return mv(E,rn(x))};
(onclick=_=>{R();let t=0,o='',id=++G,f=_=>{if(id-G)return;for(let k=0;k<6;k++){let l=S(t),mx=Math.max(...l),p=l.map(v=>Math.exp((v-mx)/Z)),r=Math.random()*p.reduce((a,b)=>a+b);for(t=0;(r-=p[t])>0&&t<V-1;t++);if(!t||o.length>4e3)return document.body.innerText=o;o+=X[t-1]}document.body.innerText=o+'_';setTimeout(f)};document.body.style.cssText='white-space:pre-wrap;font:18px/1.5 Georgia,serif;max-width:40em;margin:2em auto;padding:0 1em';f()})()
