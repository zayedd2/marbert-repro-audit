import sys,re,unicodedata,numpy as np,pandas as pd
from sklearn.metrics import f1_score
sp,pkg=sys.argv[1],sys.argv[2]
DIAC=re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
def norm(t):
    t=unicodedata.normalize("NFKC",str(t)).strip().strip('"')
    t=re.sub(r"https?://\S+|www\.\S+"," ",t); t=re.sub(r"@\w+"," ",t); t=re.sub(r"\bRT\b"," ",t)
    t=DIAC.sub("",t).replace("ـ",""); t=re.sub("[إأآٱ]","ا",t).replace("ى","ي").replace("ة","ه")
    t=re.sub(r"[^\w\s]"," ",t); t=re.sub(r"(.)\1{2,}",r"\1\1",t); return re.sub(r"\s+"," ",t).strip().lower()
tr=pd.read_csv(f'{sp}/arsarcasm_train_full.csv'); te=pd.read_csv(f'{sp}/arsarcasm_test.csv')
trn=set(tr.text.map(norm))-{''}; te['leak']=te.text.map(norm).isin(trn)
print('leaked test tweets:',te.leak.sum())
ea=pd.read_csv(pkg); a=ea[ea.dataset=='arsarcasm'].merge(te[['id','leak']],on='id')
r=[]
for (p,m,s,l),g in a.groupby(['protocol','model','seed','leak']):
    r.append(dict(protocol=p,model=m,seed=s,leak=l,acc=100*g.correct.mean(),f1_pn=100*f1_score(g.gold,g.pred,labels=['positive','negative'],average='macro')))
r=pd.DataFrame(r); print(r.groupby(['protocol','leak','model'])[['acc','f1_pn']].mean().round(2).unstack('model'))
# clean-test gap
for p in ['historical','independent']:
    c=a[(a.protocol==p)&(~a.leak)]
    d=[100*(f1_score(*c[(c.model=='marbert')&(c.seed==s)][['gold','pred']].T.values,labels=['positive','negative'],average='macro')-f1_score(*c[(c.model=='arabert')&(c.seed==s)][['gold','pred']].T.values,labels=['positive','negative'],average='macro')) for s in [42,43,44]]
    print(p,'clean-test F1-PN gap per seed',np.round(d,2),'mean',round(np.mean(d),2))
