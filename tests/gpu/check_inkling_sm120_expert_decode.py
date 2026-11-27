"""Compare one-token expert layouts on real IQ4_NL and Q8_0 Inkling weights on a single Blackwell GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

from gguf import GGUFReader, GGMLQuantizationType
import numpy as np
import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.fused_moe.standard_quant.q8_0 import q8_0_moe_kernel
from vllm_gguf_plugin.triton.fused_moe.utils import run_triton_fused_moe_kernel
from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import get_iq_table_tensors


def capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def latency(graph):
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    first.record()
    graph.replay()
    last.record()
    last.synchronize()
    return first.elapsed_time(last) * 1000


def measure(tensor, trials, iq4_kernel, gate_kernel, production):
    shared = tensor.name.endswith('ffn_gate_up_exps.weight')
    quant = int(tensor.tensor_type)
    width, count, top = (2304, 1, 8) if shared else (1152, 8, 1)
    shape = (128, 2304, width // 32 * (18 if quant == 20 else 34))
    assert tensor.data.shape == shape and quant in (20,8)
    layer = int(tensor.name.split('.')[1])
    selected = np.sort(np.random.default_rng(4721 + 7 * layer + width).choice(128, 24, replace=False))
    weights = torch.empty(shape, device='cuda', dtype=torch.uint8)
    for expert in selected:
        weights[int(expert)].copy_(torch.from_numpy(np.array(tensor.data[expert], copy=True)))
    chosen = torch.as_tensor(selected, device='cuda')
    rng = torch.Generator(device='cuda').manual_seed(8171 + width + layer)
    samples = torch.randn((trials, count, width), device='cuda', dtype=torch.bfloat16, generator=rng)
    other = chosen[torch.rand((trials-3, 24), device='cuda', generator=rng).topk(8).indices]
    routes = torch.cat((chosen.view(3, 8), other))
    x, ids = samples[0].clone(), routes[:1].clone()
    slots = torch.full((8, 4), 8, device='cuda', dtype=torch.int32)
    slots[:, 0] = torch.arange(8, device='cuda', dtype=torch.int32)
    slots = slots.flatten()
    padded = torch.tensor([32], device='cuda', dtype=torch.int32)
    table = get_iq_table_tensors(weights.device)['kvalues_iq4nl']
    original = lambda: ops.ggml_moe_a8_vec(x, weights, ids.int().reshape(-1, top), top, quant, 2304, count)
    graphs = {'original': capture(original)}
    if quant == 8:
        for tile in (16, 32, 64, 128):
            graphs['q8-n'+str(tile)] = capture(lambda n=tile: run_triton_fused_moe_kernel(
                q8_0_moe_kernel, weights, x, slots, ids.flatten().int(), padded, 2304, top, count, 8,
                block_m=4, block_n=n))
    else:
        if production:
            from vllm_gguf_plugin.inkling.pro_iq4 import project
            geometry = (8, 64, 4) if shared else (16, 64, 2)
            graphs['pro-first-pass'] = capture(lambda: project(x, weights, ids, shared, geometry))
        if shared:
            for size in (32, 64, 128):
                graphs[f'direct-{size}'] = capture(lambda n=size: gate_kernel(x, weights, ids, n))
        for register, tile, warps in [(False, 32, 4),(False, 16, 2),(True, 16, 2),(True, 32, 2),(True, 64, 4)]:
            graphs[f'iq4-{int(register)}-n{tile}-w{warps}'] = capture(
                lambda reg=register,n=tile,w=warps: run_triton_fused_moe_kernel(
                    iq4_kernel, weights, x, slots, ids.flatten().int(), padded, 2304, top, count, quant,
                    extra_args=(None if reg else table,), block_m=4, block_n=n, block_k_blocks=4,
                    num_warps=w,num_stages=2))
    # Larger than this SKU's L2. Time both a cold route and its immediate replay.
    scan_bytes = 2 * torch.cuda.get_device_properties(0).L2_cache_size
    scrub = torch.zeros(scan_bytes, device='cuda', dtype=torch.uint8)
    data = {k:{'cold':[],'warm':[],'changed':0,'max_abs':0.} for k in graphs}
    names = list(graphs)
    for trial in range(trials):
        x.copy_(samples[trial])
        ids.copy_(routes[trial:trial+1])
        for offset in range(len(names)):
            name = names[(trial+offset) % len(names)]
            graph, _ = graphs[name]
            scrub.add_(1)
            data[name]['cold'].append(latency(graph))
            data[name]['warm'].append(latency(graph))
        for cap, out in graphs.values():
            out.fill_(float('nan'))
            cap.replay()
        reference = graphs['original'][1]
        assert torch.isfinite(reference).all()
        for name, (_, output) in graphs.items():
            assert torch.isfinite(output).all(), name
            data[name]['changed'] += (reference != output).sum().item()
            data[name]['max_abs'] = max(data[name]['max_abs'], (reference.float()-output.float()).abs().max().item())
    rows = {name:{'cold_us':statistics.median(v['cold']),'warm_us':statistics.median(v['warm']),
                  'changed':v['changed'],'max_abs':v['max_abs']} for name,v in data.items()}
    return {'tensor':tensor.name,'projection':'gate_up' if shared else 'down','qtype':quant,'trials':trials,
            'values':trials * 8 * 2304,'distinct_experts':int(routes.unique().numel()),'scan_bytes':scan_bytes,'device':torch.cuda.get_device_name(),'rows':rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('iq4_kernel',type=Path)
    ap.add_argument('gate_kernel',type=Path)
    ap.add_argument('--gpu-name', choices=('NVIDIA GeForce RTX 5090','NVIDIA RTX PRO 6000 Blackwell Workstation Edition'), required=True)
    ap.add_argument('--trials',type=int,default=12)
    a=ap.parse_args()
    assert 6<=a.trials<=60
    assert a.checkpoint.parent.name=='e8a2cecf0e78914f07cc27b9be711a9b2c7561d9' and a.checkpoint.stat().st_size==17627940768
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0) and torch.cuda.get_device_name()==a.gpu_name
    assert not any(ops._cuda_kernel_available('ggml_moe_a8_vec',q) for q in (8,20))
    with a.checkpoint.open('rb') as fp:
        assert hashlib.file_digest(fp,'sha256').hexdigest()=='cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod
        spec.loader.exec_module(mod)
        return mod
    iq4=load('vllm_gguf_plugin.triton.fused_moe.iq_quant.iq4_candidate', a.iq4_kernel)
    gate=load('vllm_gguf_plugin.inkling.gate_candidate', a.gate_kernel)
    source=GGUFReader(str(a.checkpoint)).tensors
    for qtype in (GGMLQuantizationType.IQ4_NL,GGMLQuantizationType.Q8_0):
        for suffix in ('ffn_down_exps.weight','ffn_gate_up_exps.weight'):
            tensors=[t for t in source if t.tensor_type==qtype and t.name.endswith(suffix)]
            assert len(tensors)==(20 if int(qtype)==20 else 4)
            if int(qtype)==20:tensors=[tensors[0],tensors[-1]]
            for tensor in tensors:
                print(json.dumps(measure(tensor,a.trials,iq4.iq4_nl_moe_kernel,gate.direct_gate_up,'PRO 6000' in a.gpu_name)),flush=True)


if __name__=='__main__':
    main()
