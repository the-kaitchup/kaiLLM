"""Time the native Q8_1 and BF16 expert projections on identical published Inkling IQ4_NL weights."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize
from vllm_gguf_plugin import ops

from check_inkling_iq4_decode_routes import captured, measure


def q8_1_input(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 127
    z = blocks / torch.where(scale == 0, 1, scale)
    codes = z.sign() * (z.abs() + 0.5).floor()
    return (codes * scale.half().float()).reshape_as(x)


def compare(tensor, shared, trials):
    weights = np.asarray(tensor.data)
    experts, rows, packed_width = weights.shape
    assert int(tensor.tensor_type) == 20 and packed_width % 18 == 0 and (experts, rows) == (128, 2304)
    width = packed_width // 18 * 32
    assert width == (2304 if shared else 1152)
    compressed = torch.from_numpy(weights.copy()).cuda()
    slots = 8
    n, k = (1, slots) if shared else (slots, 1)
    rng = torch.Generator().manual_seed(91 + width)
    x = torch.randn(n, width, generator=rng).bfloat16()
    x_gpu = x.cuda()
    routes = torch.rand(trials, experts, generator=rng).topk(slots).indices.int()
    ids = routes[0].clone().reshape(n, k).cuda()
    args = (x_gpu, compressed, ids, k, 20, rows, n)
    alternatives = {
        'triton_bf16': lambda: ops.ggml_moe_a8_vec(*args),
        'native_q8_1': lambda: torch.ops._C_gguf.ggml_moe_a8_vec(*args),
    }
    graphs = {name: captured(fn) for name, fn in alternatives.items()}
    scan = torch.zeros(256 * 1024**2, device='cuda', dtype=torch.uint8)
    timings = {name: {'after_scan': [], 'immediate_replay': []} for name in graphs}
    names = list(graphs)
    for index, route in enumerate(routes):
        ids.copy_(route.reshape_as(ids))
        for name in names[index % 2:] + names[:index % 2]:
            graph, _ = graphs[name]
            scan.add_(1)
            timings[name]['after_scan'].append(measure(graph))
            timings[name]['immediate_replay'].append(measure(graph))

    selected_rows = np.linspace(0, rows - 1, 64, dtype=int)
    selected_weights = np.stack([weights[e, selected_rows] for e in routes[-1].tolist()])
    dense = torch.from_numpy(dequantize(selected_weights, GGMLQuantizationType.IQ4_NL))
    q8_ref = (dense * q8_1_input(x)[:, None, :]).sum(-1)
    blocks = selected_weights.reshape(*selected_weights.shape[:2], -1, 18)
    scale = torch.from_numpy(blocks[..., :2].copy().view(np.float16)).bfloat16()
    unit_blocks = blocks.copy()
    unit_blocks[..., :2] = np.array([0, 60], dtype=np.uint8)  # little-endian FP16 1.0
    codes = dequantize(unit_blocks.reshape(selected_weights.shape), GGMLQuantizationType.IQ4_NL)
    codes = torch.from_numpy(codes).reshape(*blocks.shape[:-1], 32).bfloat16()
    # The BF16 fallback rounds the source scale before multiplying its integer code.
    bf16_weights = (codes * scale).flatten(-2).float()
    bf16_ref = (bf16_weights * x.float()[:, None, :]).sum(-1)
    diagnostics = {}
    for name, (_, out) in graphs.items():
        ref = q8_ref if name == 'native_q8_1' else bf16_ref
        got = out[:, selected_rows.tolist()].cpu().float()
        assert got.shape == ref.shape == (8,64)
        def relative_rms(left, right):
            return float(torch.linalg.vector_norm(left-right) / torch.linalg.vector_norm(right))
        own_error = relative_rms(got,ref)
        diagnostics[name] = {
            'relative_rms_to_arithmetic_reference': own_error,
            'relative_rms_to_bf16_reference': relative_rms(got,bf16_ref),
            'max_absolute_error_to_arithmetic_reference': float((got-ref).abs().max()),
        }
        if not torch.isfinite(got).all() or own_error > 0.01:
            raise ValueError(f'{name} projection differs from its independent reference: {diagnostics[name]}')
    return dict(tensor=tensor.name, gpu=torch.cuda.get_device_name(), independent_values_per_path=q8_ref.numel(),
                median_us={k:{s:statistics.median(v) for s,v in groups.items()} for k,groups in timings.items()},
                numerical=diagnostics, samples_us=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('native_library', type=Path)
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or args.trials < 4 or args.trials % 2:
        raise ValueError('Select one physical device and an even trial count of at least four')
    if ops._cuda_kernel_available('ggml_moe_a8_vec', 20):
        raise ValueError('The baseline must select the installed BF16 Triton implementation')
    with args.native_library.open('rb') as stream:
        native_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    torch.ops.load_library(str(args.native_library.resolve()))
    tensors = GGUFReader(str(args.model)).tensors
    for suffix, shared in [('ffn_gate_up_exps.weight',True),('ffn_down_exps.weight',False)]:
        t = next(t for t in tensors if t.name.endswith(suffix) and int(t.tensor_type)==20)
        print(json.dumps(dict(native_sha256=native_sha, **compare(t,shared,args.trials))), flush=True)


if __name__ == '__main__':
    main()
