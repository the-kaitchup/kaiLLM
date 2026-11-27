"""Time the standard and candidate routed projections on the original Inkling UD-Q2_K_XL tensors."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from gguf import GGUFReader

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import get_iq_table_tensors
from vllm_gguf_plugin.triton.fused_moe.utils import run_triton_fused_moe_kernel
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from inkling_iq_gate_kernels import BLOCKS, project

NAMES = {22:'iq2_s', 19:'iq1_s', 16:'iq2_xxs', 17:'iq2_xs'}
DIGEST = 'b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1'


def graph(call):
    call()
    torch.cuda.synchronize()
    cap = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cap):
        output = call()
    return cap, output


def event_us(cap):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    cap.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def measure(reader, typ, tokens, trials, ordered_only, tuning, tune_decode):
    tensor = next(t for t in reader.tensors if t.name.endswith('ffn_gate_up_exps.weight') and int(t.tensor_type) == typ)
    w = torch.from_numpy(np.array(tensor.data,copy=True)).cuda()
    assert w.shape == (128,2304,9*BLOCKS[typ][0]) and w.dtype == torch.uint8
    rand = torch.Generator(device='cuda').manual_seed(522041+typ+tokens)
    x = torch.empty((tokens,2304),device='cuda',dtype=torch.bfloat16)
    ids = torch.empty((tokens,8),device='cuda',dtype=torch.int64)
    scrub = torch.zeros(2 * torch.cuda.get_device_properties(0).L2_cache_size, device='cuda', dtype=torch.uint8)
    def refresh():
        x.normal_(generator=rand)
        ids.copy_(torch.rand((tokens,128),generator=rand,device='cuda').topk(8).indices)
    refresh()
    assert not ops._cuda_kernel_available('ggml_moe_a8_vec', typ)
    baseline = graph(lambda: ops.ggml_moe_a8_vec(x,w,ids.int(),8,typ,2304,tokens))
    mod = importlib.import_module('vllm_gguf_plugin.triton.fused_moe.iq_quant.'+NAMES[typ])
    old_kernel = getattr(mod,NAMES[typ]+'_moe_kernel')
    table = get_iq_table_tensors(x.device)
    args = (table[BLOCKS[typ][1]],)
    if typ in (16,17):
        args += (table['ksigns_iq2xs'],)
    def retile(m,n,warps):
        if tokens==1:
            order = torch.arange(8,dtype=torch.int32,device=x.device)
            experts = ids.flatten().int()
            padded = torch.full((1,),8,dtype=torch.int32,device=x.device)
        else:
            order,experts,padded = moe_align_block_size(ids.int(),m,128)
        return run_triton_fused_moe_kernel(old_kernel,w,x,order,experts,padded,2304,8,tokens,typ,
                    extra_args=args,block_m=m,block_n=n,num_warps=warps)
    exact_tiles = [(1,16,2),(1,32,4)] if tokens==1 else [(16,32,4),(32,64,4)]
    vector_tiles = [(8,64,4),(16,64,4),(32,128,4)] if tokens==1 else [(32,64,4),(64,128,4)]
    if tune_decode:
        vector_tiles = [(16,64,4), (8,64,2), (16,32,2), (16,32,4), (16,64,2),
                        (16,128,4), (16,128,8), (32,64,4), (32,64,8), (32,128,8), (32,256,8)]
        exact_tiles = []
    calls = {'current':lambda: ops.ggml_moe_a8_vec(x,w,ids.int(),8,typ,2304,tokens)}
    for m,n,warps in exact_tiles:
        calls[f'retile-{m}-{n}-{warps}'] = lambda m=m,n=n,warps=warps:retile(m,n,warps)
    for n,chunk,warps in vector_tiles:
        calls[f'vector-{n}-{chunk}-{warps}'] = lambda n=n,chunk=chunk,warps=warps: project(x,w,ids,typ,n,chunk,warps)
    if ordered_only and tokens>1:
        calls = {"current": calls["current"]}
        tiles = ((4,16,64,4),(8,16,64,4),(8,32,32,4),(16,16,32,4),(16,32,64,8))
        if tuning:
            tiles = ((8,32,32,4), (4,64,16,4), (4,64,32,4), (8,32,16,4),
                     (8,64,16,4), (8,64,32,4), (16,32,16,4), (16,64,16,8), (16,64,32,8))
        for m,n,chunk,warps in tiles:
            calls[f'ordered-{m}-{n}-{chunk}-{warps}'] = lambda m=m,n=n,chunk=chunk,warps=warps:project(
                x,w,ids,typ,n,chunk,warps,tile_m=m,ordered=True)
    graphs = {'current':baseline}
    for name,call in calls.items():
        if name!='current':
            graphs[name] = graph(call)
    def stat():
        return dict(cold=[],warm=[],changed=0,l2_sq=0.0,ref_sq=0.0,max_abs=0.0)
    stats = {name:stat() for name in graphs}
    names = list(graphs)
    for i in range(trials):
        refresh()
        for j in range(len(names)):
            name = names[(i+j)%len(names)]
            cap,out = graphs[name]
            scrub.add_(1)
            stats[name]['cold'].append(event_us(cap))
            stats[name]['warm'].append(event_us(cap))
        ref = baseline[1]
        # Bound scratch for the full 2048-token prefill projection.
        for name,(_,out) in graphs.items():
            st = stats[name]
            for a,b in zip(out.split(512),ref.split(512),strict=True):
                delta = a.float()-b.float()
                assert torch.isfinite(a).all() and torch.isfinite(b).all() and a.dtype == b.dtype == torch.bfloat16
                st['changed'] += torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)).item()
                st['l2_sq'] += delta.square().sum().item()
                st['ref_sq'] += b.float().square().sum().item()
                st['max_abs'] = max(st['max_abs'],delta.abs().max().item())
    summary = {n:dict(cold_us=statistics.median(s['cold']),warm_us=statistics.median(s['warm']),
                      changed=s['changed'],max_abs=s['max_abs'],rel_l2=(s['l2_sq']/s['ref_sq'])**.5)
               for n,s in stats.items()}
    print(json.dumps(dict(tensor=tensor.name,ggml_type=typ,tokens=tokens,values=trials*tokens*8*2304,checkpoint_sha256=DIGEST,device=torch.cuda.get_device_name(),variants=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--trials',type=int,default=4)
    p.add_argument('--tokens',type=int,nargs='+',choices=(1,512,2048),default=[1,2048])
    p.add_argument('--ordered-only',action='store_true',help='Use the ordered butterfly implementation for prefill as well as decode')
    p.add_argument('--tune-prefill',action='store_true',help='Screen larger output tiles on the dominant two prefill tensor types')
    p.add_argument('--tune-decode',action='store_true',help='Compare B1 decode tiles and warp counts against the stock kernel')
    p.add_argument('--all-prefill-types',action='store_true',help='Time every original gate-up type at full prefill width')
    args=p.parse_args()
    if args.tune_decode and args.tokens != [1]:p.error('Decode tile tuning requires --tokens 1')
    if args.tune_prefill and (not args.ordered_only or args.tokens != [2048]):p.error('Use --tokens 2048 --ordered-only with --tune-prefill')
    assert args.model.parent.name=='8078d149286eec7193795413b9e0643da6d11fa1' and args.model.stat().st_size==9138716608
    with args.model.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==DIGEST
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0) and 2<=args.trials<=24
    reader=GGUFReader(str(args.model))
    for tokens in args.tokens:
        for typ in ((22,16) if args.tune_prefill and not args.all_prefill_types else (22,19,16,17)):
            measure(reader,typ,tokens,args.trials,args.ordered_only,args.tune_prefill,args.tune_decode)
            gc.collect()
            torch.cuda.empty_cache()
