"""Compare two route layouts for the same IQ4_NL expert kernel during single-token decode."""
import argparse
import json
import statistics

import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m


def captured(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def measure(graph):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    graph.replay()
    end.record()
    end.synchronize()
    return 1000 * begin.elapsed_time(end)


def compare(projection, trials, tiles):
    from vllm_gguf_plugin.triton.fused_moe.iq_quant.iq4_nl import iq4_nl_moe_kernel
    from vllm_gguf_plugin.triton.fused_moe.utils import run_triton_fused_moe_kernel
    from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import get_iq_table_tensors

    shared = projection == 'gate_up'
    experts, width, slots = 128, 2304, 8
    cols = 2304 if shared else 1152
    tokens, top_k = (1, slots) if shared else (slots, 1)
    gen = torch.Generator(device='cuda').manual_seed(817 + cols)
    weight = torch.randint(256, (experts, width, cols//32, 18), device='cuda', generator=gen, dtype=torch.uint8)
    scales = torch.randn(weight.shape[:-1], device='cuda', generator=gen, dtype=torch.float16) * 0.02
    weight[..., :2] = scales.view(torch.uint8).reshape(*weight.shape[:-1], 2)
    del scales
    weight = weight.flatten(2)
    x = torch.randn(tokens, cols, device='cuda', generator=gen, dtype=torch.bfloat16)
    routes = torch.rand(trials, experts, device='cuda', generator=gen).topk(slots).indices
    ids = routes[0].clone()
    def canonical():
        return ops.ggml_moe_a8_vec(x, weight, ids.int().reshape(tokens, top_k), top_k, 20, width, tokens)

    def direct(block):
        # One live route per physical tile. The kernel masks slots numbered >= slots.
        padded = torch.full((slots, block), slots, device='cuda', dtype=torch.int32)
        padded[:, 0] = torch.arange(slots, device='cuda', dtype=torch.int32)
        order = padded.flatten()
        count = torch.tensor([order.numel()], device='cuda', dtype=torch.int32)
        values = get_iq_table_tensors(weight.device)['kvalues_iq4nl']

        def apply():
            return run_triton_fused_moe_kernel(iq4_nl_moe_kernel, weight, x, order, ids.int(), count,
                                               width, top_k, tokens, 20, extra_args=(values,), block_m=block)
        return apply

    # Retain the captured buffers while replaying all routes on the same stream.
    calls = [canonical, *(direct(block) for block in tiles)]
    graphs = [captured(call) for call in calls]
    samples = [{state: [] for state in ('after_scan', 'immediate_replay')} for _ in graphs]
    scan = torch.zeros(256 * 1024**2, device='cuda', dtype=torch.uint8)
    checked = 0
    for index, route in enumerate(routes):
        ids.copy_(route)
        for offset in range(len(graphs)):
            which = (index + offset) % len(graphs)
            graph, _ = graphs[which]
            # Both cache conditions use the same route set for each implementation.
            scan.add_(1)
            samples[which]['after_scan'].append(measure(graph))
            samples[which]['immediate_replay'].append(measure(graph))
        reference = graphs[0][1]
        if not torch.isfinite(reference).all().item():
            raise ValueError('Nonfinite reference output')
        for _, candidate in graphs[1:]:
            if not torch.equal(reference, candidate):
                raise RuntimeError('Single-token expert projection differed on route set ' + str(index))
        checked += reference.numel()
    for name, timing in zip(['baseline', *tiles], samples, strict=True):
        yield dict(projection=projection, row_tile=name, input_cols=cols, output_rows=width, trials=trials,
                   median_us={k:statistics.median(v) for k,v in timing.items()}, exact_values_per_candidate=checked,
                   replays_us=timing, gpu=torch.cuda.get_device_name())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--tiles', type=int, choices=(4, 16, 32), nargs='+', default=[4, 16, 32])
    args = parser.parse_args()
    if args.trials < 4 or args.trials % 2 or torch.cuda.device_count() != 1:
        raise ValueError('Select one physical GPU and an even trial count of at least four')
    if ops._cuda_kernel_available('ggml_moe_a8_vec', 20) or get_triton_moe_block_m(20) != 4:
        raise ValueError('This comparison is specific to the deployed four-row IQ4_NL Triton fallback')
    for projection in ('gate_up', 'down'):
        for sample in compare(projection, args.trials, args.tiles):
            print(json.dumps(sample), flush=True)
