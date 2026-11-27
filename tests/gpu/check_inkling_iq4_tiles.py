"""Compare Inkling's existing Triton expert projection with larger tiles on one RTX 3090 or 5090."""
import argparse
import json
import statistics

import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.inkling.iq4_prefill import tiled_prefill
from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m


def measure(call, repeats):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    graph.replay()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        durations.append(begin.elapsed_time(end))
    return result, statistics.median(durations)


def check(tokens, shared, repeats):
    experts, rows, top = 128, 2304, 8
    cols = 2304 if shared else 1152
    g = torch.Generator(device='cuda').manual_seed(912 + tokens + cols)
    weight = torch.randint(256, (experts, rows, cols // 32, 18), generator=g, device='cuda', dtype=torch.uint8)
    scales = torch.randn((experts, rows, cols // 32), generator=g, device='cuda', dtype=torch.float16) * .02
    weight[..., :2] = scales.view(torch.uint8).reshape(experts, rows, cols // 32, 2)
    del scales
    weight = weight.reshape(experts, rows, -1)
    ids = torch.rand((tokens, experts), generator=g, device='cuda').topk(top).indices
    x = torch.randn((tokens if shared else tokens * top, cols), generator=g, device='cuda', dtype=torch.bfloat16)
    ref_ids = ids if shared else ids.reshape(-1, 1)
    def baseline():
        return ops.ggml_moe_a8_vec(x, weight, ref_ids.int().contiguous(), top if shared else 1, 20, rows, x.shape[0])
    ref, baseline_ms = measure(baseline, repeats)
    for tile in (16, 32):
        actual, tiled_ms = measure(lambda: tiled_prefill(x, weight, ids, shared, rows, tile), repeats)
        diff = (actual.float() - ref.float()).abs()
        result = dict(timing='cuda_events_around_graph_replay', replays=repeats, input_tokens=tokens,
                      projection='gate_up' if shared else 'down', baseline_tile=get_triton_moe_block_m(20),
                      candidate_tile=tile, baseline_ms=baseline_ms, candidate_ms=tiled_ms,
                      speedup=baseline_ms/tiled_ms, identical_values=(diff == 0.).sum().item(),
                      total_values=diff.numel(), max_abs_delta=diff.max().item())
        print(json.dumps(result), flush=True)
        if result['identical_values'] != result['total_values']:
            raise ValueError('Changing only the expert row tile must preserve every BF16 output')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', choices=['3090', '5090'], default='5090')
    p.add_argument('--tokens', type=int, nargs='+', default=[512, 2048])
    p.add_argument('--repeats', type=int, default=7)
    a = p.parse_args()
    if torch.cuda.device_count() != 1 or not torch.cuda.get_device_name().endswith('GeForce RTX ' + a.gpu):
        raise ValueError(f'Select exactly one GeForce RTX {a.gpu} with CUDA_VISIBLE_DEVICES')
    if ops._cuda_kernel_available('ggml_moe_a8_vec', 20) or get_triton_moe_block_m(20) != 4:
        raise ValueError('The reference must use the measured catalog policy and four-row IQ4_NL Triton kernel')
    for tokens in a.tokens:
        for shared in (True, False):
            check(tokens, shared, a.repeats)
