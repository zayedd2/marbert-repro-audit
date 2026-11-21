import json,sys,os,numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
R=sys.argv[1]+'/outputs'; MET={'arsarcasm':'f1_pn','astd':'f1_pn','ajgt':'accuracy'}
def ep(p,d,m,s): return [json.loads(l) for l in open(f'{R}/{p}/{d}/{m}/seed{s}/metrics_per_epoch.jsonl')]
rows=[]
for d,k in MET.items():
  for m in ['marbert','arabert']:
    for s in [42,43,44]:
      E=ep('independent',d,m,s)
      # recompute test metric at every epoch from preds to verify jsonl
      for e in E:
        p=pd.read_csv(f'{R}/independent/{d}/{m}/seed{s}/preds/epoch_{e["epoch"]:02d}_test_preds.csv')
        labs=sorted(set(p.gold)|set(p.pred))
        acc=100*accuracy_score(p.gold,p.pred)
        assert abs(acc-e['test']['accuracy'])<1e-6,(d,m,s,e['epoch'])
      selK=max(E,key=lambda e:(e['dev'][k],-e['epoch']))
      selM=max(E,key=lambda e:(e['dev']['macro_f1'],-e['epoch']))
      rows.append(dict(d=d,m=m,s=s,ep_selPaperMetric=selK['epoch'],test_selPaper=selK['test'][k],
        ep_selDevMacro=selM['epoch'],test_selDevMacro=selM['test'][k],testMacro_selDevMacro=selM['test']['macro_f1'],
        oracle=max(e['test'][k] for e in E)))
df=pd.DataFrame(rows); pd.set_option('display.width',250)
print(df.round(2).to_string(index=False))
g=df.groupby(['d','m'])[['test_selPaper','test_selDevMacro','testMacro_selDevMacro','oracle']].agg(['mean','std']).round(2)
print(g.to_string())
