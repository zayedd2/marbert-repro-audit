import sys,numpy as np,pandas as pd
pkg,te=sys.argv[1],sys.argv[2]
ea=pd.read_csv(pkg); t=pd.read_csv(te)
a=ea[ea.dataset=='arsarcasm'].merge(t[['id','dialect','sarcasm']],on='id')
L={'negative':0,'neutral':1,'positive':2}
def mf1(G,P,classes):  # G,P: (B,n) int arrays
    out=[]
    for c in classes:
        tp=((G==c)&(P==c)).sum(1); fp=((G!=c)&(P==c)).sum(1); fn=((G==c)&(P!=c)).sum(1)
        d=2*tp+fp+fn; out.append(np.where(d>0,2*tp/np.maximum(d,1),0))
    return 100*np.mean(out,0)
rng=np.random.default_rng(2026); B=2000; rows=[]
for col,groups in [('sarcasm',[True,False]),('dialect',['msa','egypt','gulf','levant'])]:
  for p in ['historical','independent']:
    for gv in groups:
      x=a[(a.protocol==p)&(a[col]==gv)]; ids=np.sort(x.id.unique()); n=len(ids)
      idx=rng.integers(0,n,(B,n))
      for met,cl in [('macro_f1',[0,1,2]),('f1_pn',[0,2])]:
        obs=[];bs=[]
        for s in [42,43,44]:
          v={}
          for m in ['marbert','arabert']:
            q=x[(x.model==m)&(x.seed==s)].set_index('id').loc[ids]
            v[m]=(q.gold.map(L).values,q.pred.map(L).values)
          o=mf1(v['marbert'][0][None],v['marbert'][1][None],cl)[0]-mf1(v['arabert'][0][None],v['arabert'][1][None],cl)[0]
          b=mf1(v['marbert'][0][idx],v['marbert'][1][idx],cl)-mf1(v['arabert'][0][idx],v['arabert'][1][idx],cl)
          obs.append(o);bs.append(b)
        mb=np.mean(bs,0); lo,hi=np.percentile(mb,[2.5,97.5])
        rows.append(dict(protocol=p,group=f'{col}={gv}',n=n,metric=met,seed_diffs=' / '.join(f'{d:+.2f}' for d in obs),
                         mean_diff=np.mean(obs),ci_low=lo,ci_high=hi,excludes_zero=lo>0 or hi<0))
r=pd.DataFrame(rows).round(2); pd.set_option('display.width',250); print(r.to_string(index=False))
r.to_csv(sys.argv[3],index=False)
