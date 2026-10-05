"""Deep Sets residual match-value network, NumPy/Adam, real-engine Monte Carlo labels.

Rows from a match are correlated: input files must use disjoint seed ranges.
Models and datasets belong under ignored research/artifacts/data/, never in Git.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np


def load(path, sort_seeds=False):
    assert not str(path).endswith('.incomplete') and '.invalid.' not in str(path), 'incomplete/rejected split'
    a = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    assert a.shape[1]==16 and np.isfinite(a).all(), 'invalid dataset shape/numbers'
    if sort_seeds:
        a=a[np.argsort(a[:,0],kind='stable')]
    seed, n = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
    assert np.equal(seed,a[:,0]).all() and (seed>=0).all()
    manifest=Path(path).with_suffix('.manifest.json')
    if manifest.exists():
        info=json.loads(manifest.read_text(encoding='utf8'))
        assert len(a)==info['rows'] and np.array_equal(np.unique(seed),np.arange(info['seed_start'],info['seed_end']+1)), 'incomplete match coverage'
        assert info['games']==info['seed_end']-info['seed_start']+1 and info['illegal_commands']==0
        if 'sha256' in info: assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==info['sha256'], 'dataset hash mismatch'
    scores, used, labels = a[:, 4:8], a[:, 8:12], a[:, 12:16]
    mask = np.arange(4)[None, :] < n[:, None]
    assert np.isin(n,[2,3,4]).all() and np.equal(a[:,1],n).all()
    assert ((scores>=0)&(scores<a[:,2:3])).all() and np.isin(used,[0,1]).all()
    assert (labels>=0).all() and np.allclose(labels.sum(1),1) and (labels[~mask]==0).all()
    minimum = np.where(mask, scores, np.inf).min(1)
    maximum = scores.max(1)
    temp = np.maximum(6., 8*np.sqrt((a[:, 2]-maximum)/8))
    prior = -(scores-minimum[:, None])/temp[:, None]
    x = np.stack([scores/100, used, np.broadcast_to(a[:, 2:3]/100, scores.shape),
                  np.broadcast_to(a[:, 3:4]/10, scores.shape),
                  np.broadcast_to(n[:, None]/4, scores.shape),
                  (scores-minimum[:, None])/100,
                  np.broadcast_to((a[:, 2]-maximum)[:, None]/100, scores.shape)], -1).astype(np.float32)
    _, first, inverse, counts = np.unique(seed, return_index=True, return_inverse=True, return_counts=True)
    assert np.equal(a[:,1:4],a[first[inverse],1:4]).all() and np.equal(labels,labels[first[inverse]]).all(), 'inconsistent match settings/winner labels'
    weights = 1/counts[inverse]
    weights /= weights.mean()
    return dict(x=x, mask=mask, n=n, prior=prior.astype(np.float32), y=labels.astype(np.float32),
                seed=seed, weights=weights.astype(np.float32))


def initialize(seed):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in [(7, 32), (32, 32), (64, 32), (32, 1)]:
        params.extend([rng.normal(0, np.sqrt(2/a), (a,b)).astype(np.float32), np.zeros(b, np.float32)])
    params[-2][:] = 0
    return params


def forward(p, d):
    x, mask, n = d['x'], d['mask'], d['n']
    h1 = np.maximum(0, x@p[0]+p[1])
    h2 = np.maximum(0, h1@p[2]+p[3])
    pool = (h2*mask[..., None]).sum(1)/n[:, None]
    cat = np.concatenate([h2, np.broadcast_to(pool[:, None,:], h2.shape)], -1)
    h3 = np.maximum(0, cat@p[4]+p[5])
    residual = (h3@p[6]+p[7])[...,0]
    logits = np.where(mask, d['prior']+residual, -1e9)
    exp = np.exp(logits-logits.max(1, keepdims=True))
    probs = exp/exp.sum(1, keepdims=True)
    return probs, (h1, h2, cat, h3, residual)


def gradients(p, d):
    probs, (h1, h2, cat, h3, residual) = forward(p, d)
    mask, n, w = d['mask'], d['n'], d['weights']
    # Small residual penalty avoids extreme extrapolation from finite Monte Carlo targets.
    delta = ((probs-d['y'])*w[:,None]/w.sum() + .0005*residual*mask*w[:,None]/w.sum())[...,None]
    flat = lambda a: a.reshape(-1, a.shape[-1])
    g6 = flat(h3).T@flat(delta)
    g7 = delta.sum((0,1))
    z3 = (delta@p[6].T)*(h3>0)
    g4, g5 = flat(cat).T@flat(z3), z3.sum((0,1))
    zcat = z3@p[4].T
    z2 = (zcat[...,:32]+zcat[...,32:].sum(1)[:,None,:]*mask[...,None]/n[:,None,None])*(h2>0)
    g2, g3 = flat(h1).T@flat(z2), z2.sum((0,1))
    z1 = (z2@p[2].T)*(h1>0)
    g0, g1 = flat(d['x']).T@flat(z1), z1.sum((0,1))
    return [g0,g1,g2,g3,g4,g5,g6,g7]


def metrics(probs, d):
    ce = -(d['y']*np.log(np.maximum(probs,1e-9))).sum(1)
    brier = ((probs-d['y'])**2).sum(1)
    return dict(cross_entropy=float(np.average(ce, weights=d['weights'])),
                brier=float(np.average(brier, weights=d['weights'])))


def subset(d, indices):
    return {k:v[indices] for k,v in d.items()}


def check_gradients(data):
    # Independent finite differences exercise pooled gradients and padded seats.
    rng=np.random.default_rng(931)
    p=[x.astype(np.float64) for x in initialize(931)]
    p[-2][:]=rng.normal(0,.03,p[-2].shape)
    d=subset(data,np.arange(min(7,len(data['n']))))
    g=gradients(p,d)
    def loss():
        probs,cache=forward(p,d)
        return (np.average(-(d['y']*np.log(probs+1e-30)).sum(1),weights=d['weights'])
                +.00025*np.average((cache[-1]**2*d['mask']).sum(1),weights=d['weights']))
    worst=0.
    for j in range(len(p)):
        for idx in rng.choice(p[j].size,min(5,p[j].size),replace=False):
            old=p[j].flat[idx]; eps=1e-5
            p[j].flat[idx]=old+eps; hi=loss()
            p[j].flat[idx]=old-eps; lo=loss()
            p[j].flat[idx]=old
            error=abs((hi-lo)/(2*eps)-g[j].flat[idx])
            worst=max(worst,error)
    assert worst<2e-5, f'finite-difference gradient mismatch: {worst}'
    return worst


def save(p, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'CABOMV01')
        for i in range(0, 8, 2):
            f.write(np.asarray(p[i].T, dtype='<f4').tobytes())
            f.write(np.asarray(p[i+1], dtype='<f4').tobytes())


def read_model(path):
    raw=Path(path).read_bytes()
    assert raw[:8]==b'CABOMV01' and len(raw)==13708
    values=np.frombuffer(raw[8:],dtype='<f4')
    assert np.isfinite(values).all() and (np.abs(values)<=100).all(), 'invalid model weights'
    params=[]; offset=0
    for a,b in [(7,32),(32,32),(64,32),(32,1)]:
        params.append(values[offset:offset+a*b].reshape(b,a).T.copy()); offset+=a*b
        params.append(values[offset:offset+b].copy()); offset+=b
    return params


def report(p, d, probs=None):
    if probs is None:
        probs = forward(p,d)[0]
    logits=np.where(d['mask'],d['prior'].astype(np.float64),-1e9)
    exp=np.exp(logits-logits.max(1,keepdims=True))
    baseline=exp/exp.sum(1,keepdims=True)
    result = dict(matches=int(len(np.unique(d['seed']))), rows=len(d['n']), baseline=metrics(baseline,d), network=metrics(probs,d))
    for name, losses in [('cross_entropy', lambda a: -(d['y']*np.log(np.maximum(a,1e-9))).sum(1)),
                         ('brier', lambda a: ((a-d['y'])**2).sum(1))]:
        diff = losses(probs)-losses(baseline)
        _, inv, counts = np.unique(d['seed'], return_inverse=True, return_counts=True)
        per_match = np.bincount(inv, weights=diff)/counts
        result[name+'_paired_delta'] = dict(mean=float(per_match.mean()), ci95=float(1.96*per_match.std(ddof=1)/np.sqrt(len(per_match))))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True); ap.add_argument('--validation',required=True); ap.add_argument('--test',required=True)
    ap.add_argument('--out',default='research/artifacts/data/match_value/model.bin'); ap.add_argument('--report',default='research/reports/MATCH_VALUE_LEARNING.json')
    ap.add_argument('--epochs',type=int,default=45); ap.add_argument('--seed',type=int,default=2718)
    ap.add_argument('--extra-train'); ap.add_argument('--extra-validation'); ap.add_argument('--extra-test')
    ap.add_argument('--extra-weight',type=float,default=.5); ap.add_argument('--init-model')
    ap.add_argument('--samples-per-epoch',type=int,default=0); ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--sort-seeds',action='store_true',help='canonicalize parallel collector output before training')
    args=ap.parse_args()
    train, val, test = (load(path,args.sort_seeds) for path in [args.train,args.validation,args.test])
    extra=None
    if args.extra_train:
        assert args.extra_validation and args.extra_test and 0<args.extra_weight<1
        extra=[load(path,args.sort_seeds) for path in [args.extra_train,args.extra_validation,args.extra_test]]
    gradient_error=check_gradients(train)
    print(f'finite_difference_max_error={gradient_error:.3g}',flush=True)
    all_sets=[train,val,test]+(extra or [])
    for i,a in enumerate(all_sets):
        for b in all_sets[i+1:]:
            assert not np.intersect1d(a['seed'],b['seed']).size, 'match seeds overlap'
    p=read_model(args.init_model) if args.init_model else initialize(args.seed)
    m=[np.zeros_like(x) for x in p]; v=[np.zeros_like(x) for x in p]
    rng=np.random.default_rng(args.seed); step=0; best=float('inf'); best_p=None; best_epoch=None
    for epoch in range(args.epochs):
        count=args.samples_per_epoch or len(train['n'])
        order=rng.permutation(len(train['n']))[:count]
        for start in range(0,len(order),512):
            if extra:
                # Sample matches uniformly, then boundaries uniformly within each match.
                # Stratifying domains avoids high-variance importance weights in tiny batches.
                k=int(512*args.extra_weight)
                batches=[]
                for dataset,size in [(train,512-k),(extra[0],k)]:
                    idx=rng.choice(len(dataset['n']),size=size,p=dataset['weights']/dataset['weights'].sum())
                    batches.append(subset(dataset,idx))
                batch={key:np.concatenate([b[key] for b in batches]) for key in train}
                batch['weights']=np.ones(512,np.float32)
            else:
                batch=subset(train,order[start:start+512])
            g=gradients(p,batch); step+=1
            for j in range(len(p)):
                m[j]=.9*m[j]+.1*g[j]; v[j]=.999*v[j]+.001*g[j]**2
                p[j]-=args.lr*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        score=metrics(forward(p,val)[0],val)['cross_entropy']
        if extra:
            score=(1-args.extra_weight)*score+args.extra_weight*metrics(forward(p,extra[1])[0],extra[1])['cross_entropy']
        if score<best: best=score; best_p=[x.copy() for x in p]; best_epoch=epoch+1
        if epoch%5==0: print(f'epoch={epoch+1} validation_ce={score:.6f} best={best:.6f}',flush=True)
    save(best_p,args.out)
    result=dict(architecture='Deep Sets 7-32-32, pooled 64-32-1 + frozen softmax residual', parameters=3425,
                training_seed=args.seed, selected_epoch=best_epoch, finite_difference_max_error=gradient_error,
                train=report(best_p,train), validation=report(best_p,val), test=report(best_p,test))
    if extra:
        result['search_policy_iteration']=dict(init_model=args.init_model,domain_weight=args.extra_weight,
                                              train=report(best_p,extra[0]),validation=report(best_p,extra[1]),test=report(best_p,extra[2]))
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    # Portable prediction cases verify Rust inference against the training implementation.
    fixture=subset(test,np.arange(min(24,len(test['n']))))
    cases=np.column_stack([fixture['n'],test['x'][:len(fixture['n']),0,2]*100,test['x'][:len(fixture['n']),0,3]*10,
                           test['x'][:len(fixture['n']),:,0]*100,test['x'][:len(fixture['n']),:,1],forward(best_p,fixture)[0]])
    np.savetxt(Path(args.out).with_suffix('.predictions.tsv'),cases,delimiter='\t',fmt='%.9g')
    print(json.dumps(result['test'],indent=2))


if __name__=='__main__': main()
