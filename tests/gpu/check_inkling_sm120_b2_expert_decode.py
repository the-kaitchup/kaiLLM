"""Compare independent Inkling token rows against joined production MoE, including duplicate route IDs across slots."""
import argparse
import hashlib
import importlib.util
import sys
import json
from pathlib import Path
import statistics

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


def measure(tensor, trials, batch, iq4_nl_moe_kernel, routed_gate):
    import triton
    from vllm_gguf_plugin.inkling.moe import ExpertStage
    shared = tensor.name.endswith('ffn_gate_up_exps.weight')
    quant = int(tensor.tensor_type)
    width, count, top = (2304, batch, 8) if shared else (1152, 8 * batch, 1)
    shape = (128, 2304, width // 32 * (18 if quant == 20 else 34))
    assert tensor.data.shape == shape and quant in (20, 8)
    layer = int(tensor.name.split('.')[1])
    selected = np.sort(np.random.default_rng(4721 + 7 * layer + width).choice(128, min(128, max(24, 8 * batch)), replace=False))
    weights = torch.empty(shape, device='cuda', dtype=torch.uint8)
    for expert in selected:
        weights[int(expert)].copy_(torch.from_numpy(np.array(tensor.data[expert], copy=True)))
    chosen = torch.as_tensor(selected, device='cuda')
    rng = torch.Generator(device='cuda').manual_seed(8171 + width + layer)
    samples = torch.randn((trials, count, width), device='cuda', dtype=torch.bfloat16, generator=rng)
    routes = chosen[torch.rand((trials, batch, len(selected)), device='cuda', generator=rng).topk(8).indices]
    routes[0] = chosen[:batch * 8].view(batch, 8)
    half_overlap = torch.arange(batch, device='cuda')[:, None] * 4 + torch.arange(8, device='cuda')
    routes[1] = chosen[half_overlap % len(selected)]
    routes[2] = chosen[:8].expand(batch, 8)
    x, ids = samples[0].clone(), routes[0].clone()
    booked = batch * 8
    slots = torch.full((booked, 4), booked, device='cuda', dtype=torch.int32)
    slots[:, 0] = torch.arange(booked, device='cuda', dtype=torch.int32)
    slots = slots.flatten()
    padded = torch.tensor([booked * 4], device='cuda', dtype=torch.int32)
    table = get_iq_table_tensors(weights.device)['kvalues_iq4nl']
    # Construct an unchanged ordinary ExpertStage, keeping the authentic MMQ threshold for large down batches.
    stage = ExpertStage(quant, weights, 2304, width, None)
    graphs = {'original': capture(lambda: stage.run(x, ids, shared))}
    if quant == 20 and shared:
        def direct(tile):
            output=x.new_empty((booked, 2304))
            routed_gate[(triton.cdiv(2304, tile), booked)](x, weights, ids, output, table,
                2304, width, tile, True, num_warps=4, num_stages=1, enable_fp_fusion=False)
            return output
        for tile in (32,64):
            graphs[f'direct-n{tile}'] = capture(lambda n=tile:direct(n))
    if quant == 8:
        for tile in (16, 32, 64, 128):
            graphs['q8-n'+str(tile)] = capture(lambda n=tile: run_triton_fused_moe_kernel(
                q8_0_moe_kernel, weights, x, slots, ids.flatten().int(), padded, 2304, top, count, 8,
                block_m=4, block_n=n))
    else:
        for register, tile, warps in [(False, 32, 4),(False, 16, 2),(True, 16, 2),(True, 32, 2),(True, 64, 4)]:
            graphs[f'iq4-{int(register)}-n{tile}-w{warps}'] = capture(
                lambda reg=register,n=tile,w=warps: run_triton_fused_moe_kernel(
                    iq4_nl_moe_kernel, weights, x, slots, ids.flatten().int(), padded, 2304, top, count, quant,
                    extra_args=(None if reg else table,), block_m=4, block_n=n, block_k_blocks=4,
                    num_warps=w,num_stages=2))
    # Larger than this SKU's L2. Time both a cold route and its immediate replay.
    scan_bytes = 2 * torch.cuda.get_device_properties(0).L2_cache_size
    scrub = torch.zeros(scan_bytes, device='cuda', dtype=torch.uint8)
    data = {k:{'cold':[],'warm':[],'changed':0,'changed_values':0,'max_abs':0.} for k in graphs}
    names = list(graphs)
    for trial in range(trials):
        x.copy_(samples[trial])
        ids.copy_(routes[trial])
        for offset in range(len(names)):
            name = names[(trial+offset) % len(names)]
            graph, _ = graphs[name]
            scrub.add_(1)
            data[name]['cold'].append(latency(graph))
            data[name]['warm'].append(latency(graph))
        for cap, value in graphs.values():
            value.fill_(float('nan'))
            cap.replay()
        reference = graphs['original'][1]
        assert reference.dtype == torch.bfloat16 and torch.isfinite(reference).all()
        for name, (_, output) in graphs.items():
            assert output.dtype == reference.dtype and torch.isfinite(output).all(), name
            data[name]['changed'] += torch.count_nonzero(reference.view(torch.int16) != output.view(torch.int16)).item()
            data[name]['changed_values'] += torch.count_nonzero(reference != output).item()
            data[name]['max_abs'] = max(data[name]['max_abs'], (reference.float()-output.float()).abs().max().item())
    rows = {name:{'cold_us':statistics.median(v['cold']),'warm_us':statistics.median(v['warm']),
                  'changed':v['changed'],'changed_values':v['changed_values'],'max_abs':v['max_abs']} for name,v in data.items()}
    return {'tensor':tensor.name,'projection':'gate_up' if shared else 'down','qtype':quant,'trials':trials,
            'batch':batch,'values':trials * booked * 2304,'distinct_experts':int(routes.unique().numel()),'scan_bytes':scan_bytes,'rows':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--batch', type=int, choices=(2,4,8,16), default=2)
    parser.add_argument('--device-model', choices=('NVIDIA GeForce RTX 5090','NVIDIA RTX PRO 6000 Blackwell Workstation Edition'), default='NVIDIA GeForce RTX 5090')
    parser.add_argument('--iq4-kernel', type=Path)
    parser.add_argument('--gate-kernel', type=Path)
    args = parser.parse_args()
    if not 6 <= args.trials <= 60:
        parser.error('Use between 6 and 60 trials')

    expected_sha = 'cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4'
    with args.checkpoint.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == expected_sha
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == args.device_model
    assert torch.cuda.get_device_capability() == (12, 0)
    assert not any(ops._cuda_kernel_available('ggml_moe_a8_vec', quant) for quant in (8, 20))

    def load(path, name):
        if path is None:return importlib.import_module(name)
        key=name+'_validation'
        spec=importlib.util.spec_from_file_location(key,path)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
        return module
    quant=load(args.iq4_kernel,'vllm_gguf_plugin.triton.fused_moe.iq_quant.iq4_nl')
    gate=load(args.gate_kernel,'vllm_gguf_plugin.inkling.iq4_decode')
    qkernel=quant.iq4_nl_moe_kernel;gkernel=gate.routed_iq4_b1_kernel
    weights = GGUFReader(str(args.checkpoint)).tensors
    for quant in (GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.Q8_0):
        for suffix in ('ffn_down_exps.weight', 'ffn_gate_up_exps.weight'):
            tensors = [tensor for tensor in weights if tensor.tensor_type == quant and tensor.name.endswith(suffix)]
            assert len(tensors) == (20 if quant == GGMLQuantizationType.IQ4_NL else 4)
            if quant == GGMLQuantizationType.IQ4_NL:
                tensors = [tensors[0], tensors[-1]]
            for tensor in tensors:
                print(json.dumps(dict(checkpoint_sha256=expected_sha, device=args.device_model, **measure(tensor, args.trials, args.batch, qkernel, gkernel))), flush=True)


if __name__ == '__main__':
    main()
