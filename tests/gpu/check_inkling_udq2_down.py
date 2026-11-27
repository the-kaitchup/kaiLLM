"""Screen packed UD-Q2 down projections on the same GPU as the normal Inkling benchmark."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from uuid import UUID


CHECKPOINT = 'b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1'
FORMATS = {39: 'MXFP4', 40: 'NVFP4', 42: 'Q2_0'}
RTX_5090 = 'NVIDIA GeForce RTX 5090'
SUPPORTED = {RTX_5090, 'NVIDIA GeForce RTX 5080', 'NVIDIA GeForce RTX 5070 Ti',
             'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'}


def geometries(baseline):
    """Preserve the number of reduction blocks while varying output rows and warps per block."""
    _, blocks, _ = baseline
    for rows in (8, 16, 32, 64, 128):
        for warps in (1, 2, 4, 8):
            yield rows, blocks, warps


def confirmation_layouts(screen, baseline, kind, *, device=RTX_5090, uuid=None, batch=1):
    """Confirm only previously exact layouts that won on both the first and last measured layers."""
    samples = [row for row in screen if row['qtype'] == kind]
    expected = 1 if kind == 39 else 2
    if len(samples) != expected or len({row['tensor'] for row in samples}) != expected:
        raise ValueError('Expected the first screen of each original packing')
    for row in samples:
        if (row['checkpoint_sha256'] != CHECKPOINT or row['device'] != device
                or (uuid is not None and row.get('gpu_uuid') != uuid)
                or row['batch'] != batch or list(baseline) != row['ordinary_geometry'] or row['trials'] < 6
                or row['cpu_reference']['max_error_per_absolute_sum'] > 2e-5):
            raise ValueError('Screen belongs to a different route, geometry or reference')
    winners = []
    for geometry in geometries(baseline):
        name = 'n{}-kb{}-w{}'.format(*geometry)
        cold, warm = [], []
        for row in samples:
            qualified = row['variants'].get(name)
            if qualified is None or qualified['different_fp32_bits'] or qualified['nonfinite']:
                break
            original = row['variants']['ordinary']
            cold.append(original['median_cold_us'] / qualified['median_cold_us'])
            warm.append(original['median_warm_us'] / qualified['median_warm_us'])
        if len(cold) == expected and min(cold) >= 1.05 and min(warm) >= 1.03:
            winners.append((min(cold), min(warm), geometry))
    return [tuple(baseline), max(winners)[2]] if winners else [tuple(baseline)]


def capture(fn):
    import torch
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def replay_us(graph):
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end)


def dequantized_error(packed, kind, ids, activations, output, row_ids=(0,1,127,575,1023,1317,2048,2303)):
    """Check spatially separated original rows against explicit BF16 unpacking and a FP64 dot product."""
    import numpy as np
    import torch
    from vllm_gguf_plugin.inkling.ggml_types import dequantize

    row_ids = np.asarray(row_ids)
    experts = ids.cpu().numpy().reshape(-1)
    raw = torch.from_numpy(packed[experts[:, None], row_ids[None, :], :].copy())
    matrix = dequantize(raw, kind, 1152).to(torch.bfloat16).to(torch.float64).numpy()
    values = activations.to(device='cpu', dtype=torch.float64).numpy()
    products = matrix * values[:, None, :]
    expected = products.sum(axis=-1, dtype=np.float64)
    magnitude = np.abs(products).sum(axis=-1, dtype=np.float64)
    got = output.detach().cpu().numpy()[:, row_ids]
    if not np.isfinite(got).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite ordinary projection or dequantized reference')
    error = np.abs(got - expected)
    relative = error / np.maximum(1e-30, magnitude)
    if relative.max() > 2e-5:
        raise ValueError(f'Packed projection disagrees with the dequantized reference ({relative.max():g})')
    return {'values':int(expected.size),'max_abs':float(error.max()),'max_error_per_absolute_sum':float(relative.max())}


def measure(tensor, batch, trials, screen=None, *, uuid=None):
    import numpy as np
    import torch
    from triton.runtime.errors import OutOfResources
    from vllm_gguf_plugin.inkling import ggml_types
    from vllm_gguf_plugin.inkling.moe import ExpertStage, GEMV_ROWS
    from vllm_gguf_plugin.triton.fused_moe import new_types

    kind = int(tensor.tensor_type)
    qk, block_bytes = ggml_types.BLOCK[kind]
    packed = np.asarray(tensor.data)
    if (packed.shape != (128, 2304, 1152 // qk * block_bytes) or packed.dtype != np.uint8
            or not packed.flags.c_contiguous or tuple(map(int, tensor.shape)) != (1152, 2304, 128)):
        raise ValueError(f'Unexpected packed layout for {tensor.name}, {packed.shape}')
    routed = batch * 8
    if routed > min(GEMV_ROWS, new_types.gemv_rows(kind)):
        raise ValueError('The ordinary engine uses a grouped projection for these rows')
    weight = torch.from_numpy(np.array(packed, copy=True)).cuda()
    stage = ExpertStage(kind, weight, 2304, 1152, None)
    if stage.data.data_ptr() != weight.data_ptr() or stage._kernel(routed) != 'new':
        raise ValueError('Stock reference does not use the original packed storage')
    x = torch.empty((routed, 1152), dtype=torch.bfloat16, device='cuda')
    ids = torch.empty((batch, 8), dtype=torch.int64, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(99271 + int(tensor.name.split('.')[1]) + 8 * batch)

    def activate(t):
        x.normal_(generator=generator).mul_((.125, 1., 8.)[t % 3])
        if t == 1:
            picks = torch.randperm(128, device='cuda', generator=generator)[:8].expand(batch, 8)
        elif t == 2:
            picks = torch.randperm(128, device='cuda', generator=generator)[:routed].reshape(batch, 8)
        else:
            picks = torch.rand((batch, 128), device='cuda', generator=generator).topk(8, dim=-1).indices
        ids.copy_(picks)

    activate(0)
    original = tuple(new_types.GEMV_TILES[kind][:3])
    plans = {'ordinary':capture(lambda: stage.run(x, ids, False))}
    rejected = {}
    layouts = geometries(original) if screen is None else confirmation_layouts(
        screen, original, kind, device=torch.cuda.get_device_name(), uuid=uuid, batch=batch)
    for rows, blocks, warps in layouts:
        name = f'n{rows}-kb{blocks}-w{warps}'
        def compute(n=rows, kb=blocks, w=warps):
            return new_types.moe_new_type_gemv(x, weight, ids.reshape(-1).int().contiguous(), kind, 2304, 1,
                block_n=n, block_kb=kb, num_warps=w)
        try:
            plans[name] = capture(compute)
        except OutOfResources as error:
            rejected[name] = str(error)
    ordinary_name = f'n{original[0]}-kb{original[1]}-w{original[2]}'
    if ordinary_name not in plans:
        raise ValueError('The deployed launch geometry did not compile independently')
    stats = {name:{'cold':[],'warm':[],'different_fp32_bits':0,'nonfinite':0} for name in plans}
    names = list(plans)
    scrub = torch.zeros(2*torch.cuda.get_device_properties(0).L2_cache_size,dtype=torch.uint8,device='cuda')
    checked = []
    for t in range(trials):
        activate(t)
        order = names[t%len(names):] + names[:t%len(names)]
        for name in (order if t%2==0 else reversed(order)):
            graph,_ = plans[name]
            scrub.add_(1)
            stats[name]['cold'].append(replay_us(graph))
            stats[name]['warm'].append(replay_us(graph))
        for graph, result in plans.values():
            result.fill_(float('nan'))
            graph.replay()
        reference = plans['ordinary'][1]
        if reference.dtype != torch.float32 or reference.shape != (routed,2304) or not torch.isfinite(reference).all():
            raise ValueError('Unexpected ordinary output')
        for name,(_,output) in plans.items():
            stats[name]['nonfinite'] += int((~torch.isfinite(output)).sum().item())
            stats[name]['different_fp32_bits'] += int(torch.count_nonzero(reference.view(torch.int32)!=output.view(torch.int32)).item())
        checked.append(dequantized_error(packed,kind,ids,x,reference))
    for data in stats.values():
        data['median_cold_us'] = statistics.median(data.pop('cold'))
        data['median_warm_us'] = statistics.median(data.pop('warm'))
    if stats[ordinary_name]['different_fp32_bits'] or stats[ordinary_name]['nonfinite']:
        raise ValueError('The explicit deployed geometry does not reproduce the production projection')
    return {'tensor':tensor.name,'qtype':kind,'packing':FORMATS[kind],'block_values':qk,'block_bytes':block_bytes,
            'packed_bytes':packed.nbytes,'packed_sha256':hashlib.sha256(packed).hexdigest(),
            'batch':batch,'routed_rows':routed,'trials':trials,'values_per_variant':routed*2304*trials,
            'ordinary_geometry':original,'cpu_reference':{'values':sum(x['values'] for x in checked),
               'max_abs':max(x['max_abs'] for x in checked),'max_error_per_absolute_sum':max(x['max_error_per_absolute_sum'] for x in checked)},
            'variants':stats,'unavailable':rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--batch', type=int, choices=(1,2,4,8), default=1)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--all-layers', action='store_true')
    parser.add_argument('--confirm-screen', type=Path, help='Test the best cold-cache layout for each format across every layer')
    parser.add_argument('--device-uuid', help='Bind both screening stages to a single visible Blackwell board')
    args = parser.parse_args()
    import torch
    from gguf import GGUFReader
    if not 6 <= args.trials <= 48 or torch.cuda.device_count() != 1:
        parser.error('Use one visible device and 6–48 trials')
    device = torch.cuda.get_device_name()
    if args.device_uuid:
        uuid = args.device_uuid
        if (not uuid.startswith('GPU-') or 'GPU-' + str(UUID(uuid[4:])) != uuid
                or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid or device not in SUPPORTED):
            parser.error('The selected UUID must identify the single visible supported card')
        inventory = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,name',
                                             '--format=csv,noheader'], text=True).strip()
        if inventory.split(', ', 1) != [uuid, device]:
            parser.error('The physical GPU inventory differs from the CUDA device')
    elif device != RTX_5090 or (args.confirm_screen and args.batch != 1):
        parser.error('Pass --device-uuid to qualify another card or a batched screen')
    screen = json.loads(args.confirm_screen.read_bytes()) if args.confirm_screen else None
    with args.checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=CHECKPOINT:
            raise ValueError('Unexpected UD-Q2 checkpoint')
    tensors = defaultdict(list)
    for tensor in GGUFReader(str(args.checkpoint)).tensors:
        if tensor.name.endswith('.ffn_down_exps.weight'):
            tensors[int(tensor.tensor_type)].append(tensor)
    if {k:len(v) for k,v in tensors.items()} != {39:1,40:11,42:12}:
        raise ValueError('Unexpected original down-projection types')
    for kind, layers in sorted(tensors.items()):
        selected = layers if screen is not None or args.all_layers or len(layers)==1 else [layers[0], layers[-1]]
        for tensor in selected:
            row=measure(tensor,args.batch,args.trials,screen,uuid=args.device_uuid)
            print(json.dumps({'device':device,'gpu_uuid':args.device_uuid,'checkpoint_sha256':CHECKPOINT,**row}),flush=True)

if __name__=='__main__':
    main()
