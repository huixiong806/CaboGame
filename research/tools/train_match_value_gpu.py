"""CUDA/autograd backend for the same portable match-value model.

Reuse D:/miniconda3/envs/py311/python.exe. No libtorch dependency in the game.
"""
import argparse
import json
import os
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn
from train_match_value import load, initialize, read_model, save, metrics, report, forward, subset


class ValueNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in [(7,32),(32,32),(64,32),(32,1)]])
        with torch.no_grad():
            for i,layer in enumerate(self.layers):
                layer.weight.copy_(torch.from_numpy(params[2*i].T.copy()))
                layer.bias.copy_(torch.from_numpy(params[2*i+1].copy()))

    def forward(self,d):
        h1=torch.relu(self.layers[0](d['x']))
        h2=torch.relu(self.layers[1](h1))
        pooled=(h2*d['mask'][...,None]).sum(1)/d['n'][:,None]
        cat=torch.cat([h2,pooled[:,None,:].expand_as(h2)],-1)
        residual=self.layers[3](torch.relu(self.layers[2](cat)))[...,0]
        logits=(d['prior']+residual).masked_fill(~d['mask'],-1e9)
        return logits,residual

    def portable(self):
        p=[]
        for layer in self.layers:
            p.extend([layer.weight.detach().cpu().numpy().T.copy(),layer.bias.detach().cpu().numpy().copy()])
        return p


def tensors(d,device):
    return {k:torch.as_tensor(d[k],device=device) for k in ['x','mask','n','prior','y','weights']}


def predict(model,d,batch_size=8192):
    out=[]
    with torch.no_grad():
        for i in range(0,len(d['n']),batch_size):
            logits,_=model({k:v[i:i+batch_size] for k,v in d.items()})
            out.append(torch.softmax(logits,-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap=argparse.ArgumentParser()
    for key in ['train','validation','test']: ap.add_argument('--'+key,required=True)
    ap.add_argument('--out',default='research/artifacts/data/match_value/model-gpu.bin'); ap.add_argument('--report',default='research/reports/MATCH_VALUE_GPU_LEARNING.json')
    ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--seed',type=int,default=2718)
    ap.add_argument('--batch-size',type=int,default=4096); ap.add_argument('--device',default='cuda')
    ap.add_argument('--lr',type=float,default=.001); ap.add_argument('--sort-seeds',action='store_true')
    ap.add_argument('--extra-train'); ap.add_argument('--extra-validation'); ap.add_argument('--extra-test')
    ap.add_argument('--extra-weight',type=float,default=.5); ap.add_argument('--init-model')
    ap.add_argument('--samples-per-epoch',type=int,default=0)
    a=ap.parse_args()
    assert a.batch_size>=2 and a.epochs>0
    if a.device=='cuda': assert torch.cuda.is_available(),'CUDA requested but unavailable'
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    # Preserve FP32 precision for portable CPU inference and calibration.
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    data=[load(path,a.sort_seeds) for path in [a.train,a.validation,a.test]]
    extra=[]
    if a.extra_train:
        assert a.extra_validation and a.extra_test and 0<a.extra_weight<1
        extra=[load(path,a.sort_seeds) for path in [a.extra_train,a.extra_validation,a.extra_test]]
    all_data=data+extra
    for i,d in enumerate(all_data):
        for other in all_data[i+1:]: assert not np.intersect1d(d['seed'],other['seed']).size,'match seeds overlap'
    td=[tensors(d,a.device) for d in all_data]
    p=read_model(a.init_model) if a.init_model else initialize(a.seed)
    model=ValueNet(p).to(a.device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    rng=np.random.default_rng(a.seed)
    best=metrics(predict(model,td[1]),data[1])['cross_entropy']
    if extra: best=(1-a.extra_weight)*best+a.extra_weight*metrics(predict(model,td[4]),extra[1])['cross_entropy']
    best_p=model.portable(); best_epoch=0
    if a.device=='cuda': torch.cuda.synchronize()
    train_start=time.perf_counter()
    for epoch in range(a.epochs):
        order=rng.permutation(len(data[0]['n']))[:a.samples_per_epoch or len(data[0]['n'])]
        for i in range(0,len(order),a.batch_size):
            if extra:
                k=int(a.batch_size*a.extra_weight)
                batches=[]
                for d,t,size in [(data[0],td[0],a.batch_size-k),(extra[0],td[3],k)]:
                    idx=torch.as_tensor(rng.choice(len(d['n']),size=size,p=d['weights']/d['weights'].sum()),device=a.device)
                    batches.append({key:v[idx] for key,v in t.items()})
                batch={key:torch.cat([b[key] for b in batches]) for key in td[0]}
                batch['weights']=torch.ones(a.batch_size,device=a.device)
            else:
                idx=torch.as_tensor(order[i:i+a.batch_size],device=a.device)
                batch={key:v[idx] for key,v in td[0].items()}
            opt.zero_grad(set_to_none=True)
            logits,residual=model(batch)
            ce=-(batch['y']*torch.log_softmax(logits,-1)).sum(1)
            loss=((ce+.00025*(residual.square()*batch['mask']).sum(1))*batch['weights']).sum()/batch['weights'].sum()
            loss.backward(); opt.step()
        score=metrics(predict(model,td[1]),data[1])['cross_entropy']
        if extra: score=(1-a.extra_weight)*score+a.extra_weight*metrics(predict(model,td[4]),extra[1])['cross_entropy']
        if score<best: best=score; best_p=model.portable(); best_epoch=epoch+1
        if epoch%5==0: print(f'epoch={epoch+1} validation_ce={score:.6f} best={best:.6f}',flush=True)
    if a.device=='cuda': torch.cuda.synchronize()
    training_seconds=time.perf_counter()-train_start
    model=ValueNet(best_p).to(a.device)
    predictions=[predict(model,t) for t in td]
    check=subset(data[2],np.arange(min(256,len(data[2]['n']))))
    parity=float(np.max(np.abs(forward(best_p,check)[0]-predictions[2][:len(check['n'])])))
    assert parity<2e-5,f'CUDA/NumPy prediction mismatch {parity}'
    save(best_p,a.out)
    result=dict(architecture='Deep Sets 7-32-32, pooled 64-32-1 + frozen softmax residual',parameters=3425,
                backend='PyTorch autograd',device=a.device,torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
                batch_size=a.batch_size,epochs=a.epochs,selected_epoch=best_epoch,training_seed=a.seed,
                training_seconds=training_seconds,cuda_numpy_max_error=parity,
                train=report(best_p,data[0],predictions[0]),validation=report(best_p,data[1],predictions[1]),test=report(best_p,data[2],predictions[2]))
    if extra:
        result['search_policy_iteration']=dict(init_model=a.init_model,domain_weight=a.extra_weight,
                                              train=report(best_p,extra[0],predictions[3]),validation=report(best_p,extra[1],predictions[4]),test=report(best_p,extra[2],predictions[5]))
    result['total_seconds']=time.perf_counter()-start
    Path(a.report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    fixture=subset(data[2],np.arange(min(24,len(data[2]['n']))))
    cases=np.column_stack([fixture['n'],fixture['x'][:,0,2]*100,fixture['x'][:,0,3]*10,
                           fixture['x'][:,:,0]*100,fixture['x'][:,:,1],forward(best_p,fixture)[0]])
    np.savetxt(Path(a.out).with_suffix('.predictions.tsv'),cases,delimiter='\t',fmt='%.9g')
    print(f'GPU training_seconds={training_seconds:.1f} total_seconds={result["total_seconds"]:.1f} parity={parity:.3g}',flush=True)
    print(json.dumps(result['test'],indent=2))


if __name__=='__main__': main()
