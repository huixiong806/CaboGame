"""Evaluate a frozen network on an independent real-engine dataset/opponent roster."""
import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from train_match_value import load, report, read_model, forward, metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--data',required=True); ap.add_argument('--report',required=True)
    ap.add_argument('--compare-model')
    a=ap.parse_args()
    d=load(a.data)
    probabilities=forward(read_model(a.model),d)[0]
    result=report(read_model(a.model),d,probabilities)
    result['model_sha256']=hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    if a.compare_model:
        other=forward(read_model(a.compare_model),d)[0]
        comparison=dict(model=a.compare_model,metrics=metrics(other,d),sha256=hashlib.sha256(Path(a.compare_model).read_bytes()).hexdigest())
        _,inverse,counts=np.unique(d['seed'],return_inverse=True,return_counts=True)
        for name,loss in [('cross_entropy',lambda p: -(d['y']*np.log(np.maximum(p,1e-9))).sum(1)),('brier',lambda p: ((p-d['y'])**2).sum(1))]:
            per_match=np.bincount(inverse,weights=loss(probabilities)-loss(other))/counts
            comparison[name+'_paired_delta']=dict(mean=float(per_match.mean()),ci95=float(1.96*per_match.std(ddof=1)/np.sqrt(len(per_match))))
        result['comparison']=comparison
    Path(a.report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
