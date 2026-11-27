"""Measure concurrent, separate deterministic Humming projections using each original Inkling attention matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import gguf
import numpy as np
import torch
from vllm_gguf_plugin.quantization.humming import prepare_gguf_humming

SHA256 = 'cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4'
UDQ2 = 'b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1'
INPUTS = {'q':0,'gate':0,'rel':0,'sink_proj':0,'k':1,'v':2,'k_conv_proj':1,'v_conv_proj':2}
SCHEDULES = {
    'serial': (('q','k','v','k_conv_proj','v_conv_proj','rel','sink_proj','gate'),),
    'two': (('q','k','rel','k_conv_proj'),('gate','v','sink_proj','v_conv_proj')),
    'three': (('q','rel'),('gate','sink_proj'),('k','v','k_conv_proj','v_conv_proj')),
    'four': (('q','rel'),('gate','sink_proj'),('k','k_conv_proj'),('v','v_conv_proj')),
}

def projections(matrices, inputs, schedule, side):
    main = torch.cuda.current_stream()
    outputs = {}
    for stream in side:
        stream.wait_stream(main)
    for step, names in enumerate(schedule):
        with torch.cuda.stream(side[step-1] if step else main):
            for name in names:
                if name in matrices:
                    outputs[name]=matrices[name](inputs[INPUTS[name]])
    for stream in side:
        main.wait_stream(stream)
    return outputs

def capture(matrices, inputs, groups):
    streams=[torch.cuda.Stream() for _ in groups[1:]]
    for _ in range(3):
        projections(matrices,inputs,groups,streams)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result=projections(matrices, inputs, groups, streams)
    torch.cuda.synchronize()
    return graph,result,streams

def replay(graph):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();graph.replay();end.record();end.synchronize()
    return 1000*begin.elapsed_time(end)

def tuned(layer, configs, name, batch):
    from vllm.utils.humming import get_heuristics_config
    policy = json.loads(layer.compute_config)
    original = get_heuristics_config(layer.layer_config, shape_m=batch, **policy, device=0)
    if not (not original['use_stream_k'] and not original['use_f16_accum'] and original['mma_type'] == 'mma'
            and tuple(original['block_shape']) == (16,64,32) and tuple(original['warp_shape']) == (16,32,32)
            and original['num_stages'] == 3 and original['num_ctas_per_sm'] == 1):
        configs[name] = dict(shape=list(layer.shape),original=original,selected=None)
        return layer
    selected = dict(original, block_shape=[16,64,64], warp_shape=[16,32,64])
    configs[name] = dict(shape=list(layer.shape), original=original, selected=selected)
    def calculate(x):
        output=layer._humming_forward(layer.layer_config, inputs=x, weight=layer.weight, weight_scale=layer.weight_scale,
            zero_point=getattr(layer,'zero_point',None), locks=layer.locks, compute_config=layer.compute_config,
            tuning_config=selected)
        return output[..., :layer.shape[0]].reshape(batch,layer.shape[0])
    return calculate

def evaluate(tensors, layer, shared_input, trials, batch=1):
    prefix=f'blk.{layer}.attn_'
    mats={}
    rows={}
    kinds={}
    for name in INPUTS:
        key=prefix+name+'.weight'
        if key not in tensors:
            if name not in ('k_conv_proj','v_conv_proj'):raise ValueError(key)
            continue
        t=tensors[key]
        kind=int(t.tensor_type)
        width,count=map(int,t.shape)
        assert width==2304 and kind in (8,12,13,14,18,30)
        rows[name],kinds[name]=count,kind
        data=np.array(t.data,copy=True)
        if kind==30:
            from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
            weight=torch.from_numpy(data.view(np.uint16).reshape(count,width)).view(torch.bfloat16).cuda()
            mats[name]=lambda x,w=weight: _fused_mul_mat_gguf(x,w,30)
        else:
            mats[name]=prepare_gguf_humming(torch.from_numpy(data).cuda(),kind,torch.bfloat16)
            assert json.loads(mats[name].compute_config)['use_batch_invariant']
    xs=[torch.empty(batch,2304,dtype=torch.bfloat16,device='cuda') for _ in range(1 if shared_input else 3)]
    rng=torch.Generator(device='cuda').manual_seed(8831+layer+int(shared_input))
    for x in xs:x.normal_(generator=rng)
    inputs=xs*3 if shared_input else xs
    configs={}
    if batch==1:
        runs={name:capture(mats,inputs,shapes) for name,shapes in SCHEDULES.items()}
    else:
        tuned_layers={name:(tuned(mat,configs,name,batch) if kinds[name]!=30 else mat) for name,mat in mats.items()}
        runs={'serial':capture(mats,inputs,SCHEDULES['serial']), 'four':capture(mats,inputs,SCHEDULES['four']),
              'serial_tuned':capture(tuned_layers,inputs,SCHEDULES['serial']),
              'four_tuned':capture(tuned_layers,inputs,SCHEDULES['four'])}
    names=list(runs);clear=torch.zeros(2*torch.cuda.get_device_properties(0).L2_cache_size,dtype=torch.uint8,device='cuda')
    times={n:{'cold':[],'warm':[],'different':{k:0 for k in mats}} for n in names}
    for t in range(trials):
        for x in xs:x.normal_(generator=rng).mul_([.125,1.,8.][t%3])
        for i in range(len(names)):
            n=names[(i+t)%len(names)];g=runs[n][0]
            clear.add_(1)
            times[n]['cold'].append(replay(g))
            times[n]['warm'].append(replay(g))
        for cap,vals,_ in runs.values():
            for out in vals.values():out.fill_(float('nan'))
            cap.replay()
        for key,ref in runs['serial'][1].items():
            assert torch.isfinite(ref).all()
            if batch>1:assert ref.shape[0]==batch and (ref[0]!=ref[-1]).any()
            for n,(_,out,_) in runs.items():
                value=out[key]
                if not torch.isfinite(value).all():raise ValueError(f'Nonfinite projection {key} under {n}')
                times[n]['different'][key]+=torch.count_nonzero(ref.view(torch.int16)!=value.view(torch.int16)).item()
    for row in times.values():
        row['cold_us']=statistics.median(row.pop('cold'))
        row['warm_us']=statistics.median(row.pop('warm'))
    return dict(layer=layer,shared_input=shared_input,trials=trials,weights=sorted(mats),schedules=times,
                quant_types=kinds,tested_bf16_values=batch*trials*sum(rows.values()),batch=batch,configs=configs)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('checkpoint',type=Path);ap.add_argument('--trials',type=int,default=24)
    ap.add_argument('--device-model', choices=('NVIDIA GeForce RTX 5090', 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'), default='NVIDIA GeForce RTX 5090')
    ap.add_argument('--batch',type=int,choices=(1,2,4,8,16),default=1)
    a=ap.parse_args()
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name()==a.device_model and torch.cuda.get_device_capability()==(12,0)
    assert os.environ['KAILLM_HUMMING_DETERMINISTIC']=='1' and 16<=a.trials<=64
    digests={'Inkling-2-Mini-IQ4_NL.gguf':SHA256,'Inkling-2-Mini-UD-Q2_K_XL.gguf':UDQ2}
    digest=digests[a.checkpoint.name]
    with a.checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    all_tensors={t.name:t for t in gguf.GGUFReader(str(a.checkpoint)).tensors}
    for l,shared in ((0,True),(1,False),(11,False),(20,True)):
        print(json.dumps(dict(checkpoint_sha256=digest,device=torch.cuda.get_device_name(),**evaluate(all_tensors,l,shared,a.trials,a.batch))),flush=True)

if __name__=='__main__':main()
