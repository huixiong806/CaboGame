"""CUDA distillation of search decisions on legally observable state/action features.

Groups stay together, and all decisions from a round stay in one split. No hand truth
or game seed is an input. This is imitation learning, not an optimal-policy certificate.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
from torch import nn


def load(path):
    p=Path(path)
    assert 'incomplete' not in p.name and '.invalid.' not in p.name
    manifest=json.loads(p.with_suffix('.manifest.json').read_text())
    raw=p.read_bytes();manifest=dict(manifest,data_sha256=hashlib.sha256(raw).hexdigest())
    examples=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    seeds=np.array([e['seed'] for e in examples],dtype=np.int64)
    assert set(seeds)==set(range(manifest['seed_start'],manifest['seed_end']+1))
    assert len(examples)==manifest['groups'] and manifest['illegal_commands']==0
    assert manifest['features']==64
    width=max(len(e['x']) for e in examples)
    x=np.zeros((len(examples),width,64),dtype=np.float32)
    mask=np.zeros(x.shape[:2],dtype=bool)
    y=np.zeros(x.shape[:2],dtype=np.float32)
    steps=set();round_n={}
    for i,e in enumerate(examples):
        assert (e['seed'],e['step']) not in steps;steps.add((e['seed'],e['step']))
        assert 2<=e['n']<=4 and round_n.setdefault(e['seed'],e['n'])==e['n']
        z=np.asarray(e['x'],dtype=np.float32)
        assert z.shape[1]==64 and np.isfinite(z).all() and 0<=e['label']<len(z)
        x[i,:len(z)]=z;mask[i,:len(z)]=True
        # Semantically identical feature rows cannot have an arbitrary preferred slot.
        same=np.max(np.abs(z-z[e['label']]),axis=1)<1e-7
        y[i,:len(z)]=same/same.sum()
    assert int(mask.sum())==manifest['action_rows']
    unique,counts=np.unique(seeds,return_counts=True)
    weights=1/counts[np.searchsorted(unique,seeds)]
    weights=weights/weights.mean()
    order=np.lexsort((np.array([e['step'] for e in examples]),seeds))
    return {'x':x[order],'mask':mask[order],'y':y[order],'weight':weights[order].astype(np.float32),'seeds':seeds[order],'manifest':manifest}


def combine(old,new,new_weight):
    width=max(old['x'].shape[1],new['x'].shape[1])
    result={}
    for key in ['x','mask','y']:
        arrays=[]
        for d in [old,new]:
            padding=[(0,0),(0,width-d[key].shape[1])]+([(0,0)] if key=='x' else [])
            arrays.append(np.pad(d[key],padding))
        result[key]=np.concatenate(arrays)
    result['weight']=np.concatenate([old['weight']*(1-new_weight)/len(old['weight']),new['weight']*new_weight/len(new['weight'])])
    result['weight']=(result['weight']/result['weight'].mean()).astype(np.float32)
    result['seeds']=np.concatenate([old['seeds'],new['seeds']])
    result['manifest']={'domains':[old['manifest'],new['manifest']],'new_domain_weight':new_weight}
    return result


class PolicyNet(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        hidden=32 if width==64 else 12
        self.layers=nn.ModuleList([nn.Linear(64,width),nn.Linear(width,hidden),nn.Linear(hidden,1)])
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):
        h=torch.relu(self.layers[0](x));h=torch.relu(self.layers[1](h))
        return x[...,55]*5+self.layers[2](h)[...,0]


def numpy_predict(weights,x):
    a=x
    for i in range(3):
        w,b=weights[2*i:2*i+2]
        a=a@w.T+b
        if i<2:a=np.maximum(a,0)
    return x[...,55]*5+a[...,0]


def metrics(net,data,device):
    ce=[];correct=[];baseline=[];draw_correct=[];idle_correct=[]
    with torch.no_grad():
        for start in range(0,len(data['x']),1024):
            sl=slice(start,start+1024)
            x=torch.as_tensor(data['x'][sl],device=device)
            mask=torch.as_tensor(data['mask'][sl],device=device)
            y=torch.as_tensor(data['y'][sl],device=device)
            logits=net(x).masked_fill(~mask,-1e9)
            loss=-(y*torch.log_softmax(logits,-1)).sum(-1)
            pred=logits.argmax(-1);base=(x[...,55]*5).masked_fill(~mask,-1e9).argmax(-1)
            ce.extend(loss.cpu().numpy());correct.extend((y.gather(1,pred[:,None])[:,0]>0).cpu().numpy())
            baseline.extend((y.gather(1,base[:,None])[:,0]>0).cpu().numpy())
    w=data['weight'];correct=np.asarray(correct);phase=data['x'][:,0,0]>0.5
    return {'cross_entropy':float(np.average(ce,weights=w)),'top1_equivalent':float(np.average(correct,weights=w)),
        'score_baseline_top1_equivalent':float(np.average(baseline,weights=w)),
        'idle_accuracy':float(np.average(correct[~phase],weights=w[~phase])),
        'drew_accuracy':float(np.average(correct[phase],weights=w[phase])),
        'rounds':len(set(data['seeds'])),'groups':len(w),'actions':int(data['mask'].sum())}


def main():
    ap=argparse.ArgumentParser()
    for split in ['train','validation','test']:ap.add_argument('--'+split,required=True)
    ap.add_argument('--out',default='research/artifacts/data/action_policy/model-v1.bin');ap.add_argument('--report',default='research/reports/ACTION_POLICY_V1_LEARNING.json')
    ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--batch-size',type=int,default=256);ap.add_argument('--lr',type=float,default=.002)
    ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int,default=2718);ap.add_argument('--init-model')
    ap.add_argument('--width',type=int,choices=[24,64],default=24)
    for split in ['train','validation','test']:ap.add_argument('--extra-'+split)
    ap.add_argument('--extra-weight',type=float,default=.5)
    args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    assert args.epochs>0 and args.batch_size>0
    if args.device=='cuda':assert torch.cuda.is_available()
    assert not Path(args.out).exists(),'refusing to overwrite a frozen model; choose a new output'
    start=time.perf_counter();original=[load(getattr(args,k)) for k in ['train','validation','test']]
    extra=[]
    if any(getattr(args,'extra_'+k) for k in ['train','validation','test']):
        assert all(getattr(args,'extra_'+k) for k in ['train','validation','test']) and 0<args.extra_weight<1
        extra=[load(getattr(args,'extra_'+k)) for k in ['train','validation','test']]
    splits=original+extra
    for i in range(len(splits)):
        for j in range(i):assert set(splits[i]['seeds']).isdisjoint(splits[j]['seeds'])
    data=[combine(a,b,args.extra_weight) for a,b in zip(original,extra)] if extra else original
    device=torch.device(args.device);net=PolicyNet(args.width).to(device)
    if args.init_model:
        b=Path(args.init_model).read_bytes();assert b[:8]==(b'CABOPL01' if args.width==24 else b'CABOPL02') and len(b)==(7500 if args.width==24 else 25100)
        w=np.frombuffer(b[8:],dtype='<f4').copy();offset=0
        with torch.no_grad():
            for layer in net.layers:
                size=layer.weight.numel();layer.weight.copy_(torch.from_numpy(w[offset:offset+size].reshape(layer.weight.shape)));offset+=size
                size=layer.bias.numel();layer.bias.copy_(torch.from_numpy(w[offset:offset+size]));offset+=size
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.0002)
    train={k:torch.as_tensor(data[0][k],device=device) for k in ['x','mask','y','weight']}
    best=metrics(net,data[1],device);best_epoch=0;checkpoint={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};history=[]
    rng=np.random.default_rng(args.seed);train_start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        # Equal total probability per round, regardless of round length.
        indices=rng.choice(len(data[0]['x']),size=len(data[0]['x']),replace=True,p=data[0]['weight']/data[0]['weight'].sum())
        loss_sum=0
        for first in range(0,len(indices),args.batch_size):
            idx=torch.as_tensor(indices[first:first+args.batch_size],device=device)
            x=train['x'][idx];mask=train['mask'][idx];y=train['y'][idx]
            logits=net(x).masked_fill(~mask,-1e9);loss=-(y*torch.log_softmax(logits,-1)).sum(-1).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        val=metrics(net,data[1],device);history.append({'epoch':epoch,'train_cross_entropy':loss_sum/len(indices),'validation':val})
        if val['cross_entropy']<best['cross_entropy']:
            best=val;best_epoch=epoch;checkpoint={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch%5==0:print(json.dumps(history[-1]),flush=True)
    training_seconds=time.perf_counter()-train_start;net.load_state_dict(checkpoint)
    weights=[]
    for layer in net.layers:weights.extend([layer.weight.detach().cpu().numpy(),layer.bias.detach().cpu().numpy()])
    parameters=1873 if args.width==24 else 6273
    flat=np.concatenate([w.reshape(-1) for w in weights]).astype('<f4');assert len(flat)==parameters
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((b'CABOPL01' if args.width==24 else b'CABOPL02')+flat.tobytes())
    fixtures=data[2]['x'][:16].reshape(-1,64)[data[2]['mask'][:16].reshape(-1)][:64]
    cpu=numpy_predict(weights,fixtures)
    with torch.no_grad():gpu=net(torch.as_tensor(fixtures,device=device)).cpu().numpy()
    parity=float(np.max(np.abs(cpu-gpu)));assert parity<2e-5
    np.savetxt(out.with_suffix('.predictions.tsv'),np.column_stack([fixtures,cpu]),fmt='%.9g',delimiter='\t')
    results={k:metrics(net,d,device) for k,d in zip(['train','validation','test'],data)}
    result={'device':str(device),'parameters':parameters,'features':64,'selected_epoch':best_epoch,'epochs':args.epochs,
        'data':[d['manifest'] for d in data],'metrics':results,'history':history,'training_seconds':training_seconds,
        'total_seconds':time.perf_counter()-start,'cuda_numpy_max_error':parity,'model_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    result['training_config']={'seed':args.seed,'learning_rate':args.lr,'batch_size':args.batch_size,'width':args.width,
        'extra_weight':args.extra_weight if extra else None,'init_model_sha256':hashlib.sha256(Path(args.init_model).read_bytes()).hexdigest() if args.init_model else None,
        'torch_version':torch.__version__,'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU'}
    if extra:
        result['original_domain_metrics']={k:metrics(net,d,device) for k,d in zip(['train','validation','test'],original)}
        result['new_domain_metrics']={k:metrics(net,d,device) for k,d in zip(['train','validation','test'],extra)}
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['history','data']}),flush=True)


if __name__=='__main__':main()
