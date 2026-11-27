"""Screen deterministic B1 Humming schedules on ordinary Inkling attention weights on a selected physical GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


CHECKPOINT_SHA256 = "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"
UDQ2_SHA256 = "b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1"
SUPPORTED_DEVICES = {
    "NVIDIA GeForce RTX 3090": (8, 6),
    "NVIDIA GeForce RTX 5090": (12, 0),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (12, 0),
}


def capture(call):
    import torch
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    torch.cuda.synchronize()
    return graph, result


PROJECTIONS = {
    "output-kv": (
        (0, "attn_output", (2304, 3072)), (11, "attn_k", (1024, 2304)),
        (20, "attn_output", (2304, 3072)), (16, "attn_v", (1024, 2304)),
    ),
    "q-gate-bias": (
        (0, "attn_gate", (3072, 2304)), (20, "attn_q", (3072, 2304)),
        (0, "attn_rel", (384, 2304)), (20, "attn_rel", (384, 2304)),
        (0, "attn_sink_proj", (24, 2304)), (20, "attn_sink_proj", (24, 2304)),
    ),
}


def schedules(base, rows, multiprocessors):
    """Keep all K reductions inside one warp and compare narrow output tiles on sparse grids."""
    if base["use_stream_k"] or base["use_f16_accum"] or base["warp_shape"][2] != base["block_shape"][2]:
        raise ValueError("The comparison requires the deterministic FP32 production schedule")
    for block_n, warp_n, block_k, stages in ((64, 64, 32, 2), (64, 64, 32, 3),
                                            (64, 64, 64, 2), (64, 64, 64, 3),
                                            (32, 32, 32, 2), (32, 32, 32, 3),
                                            (64, 32, 32, 2), (64, 32, 32, 3),
                                            (32, 32, 64, 2), (32, 32, 64, 3),
                                            (64, 32, 64, 2), (64, 32, 64, 3)):
        assert rows % block_n == 0
        yield dict(base, block_shape=[16, block_n, block_k], warp_shape=[16, warp_n, block_k],
                   num_ctas_per_sm=1, num_sms=min(multiprocessors, rows // block_n), num_stages=stages)


def profile_weight(tensor, shape, samples, inputs, mixed=False):
    import numpy as np
    import torch
    from vllm.utils.humming import get_heuristics_config, humming_forward
    from vllm_gguf_plugin.quantization.humming import prepare_gguf_humming

    rows, width = shape
    from vllm_gguf_plugin.quantization.gguf_repack import FORMATS
    kind = int(tensor.tensor_type)
    if (kind not in FORMATS or (not mixed and kind != 8) or tuple(tensor.shape) != (width, rows)):
        raise ValueError(f"Unexpected original matrix {tensor.name} {tensor.shape}")
    packed = np.array(tensor.data, copy=True).reshape(rows, -1)
    _, _, block, size = FORMATS[kind]
    if packed.dtype != np.uint8 or packed.shape != (rows, width // block * size):
        raise ValueError("Unexpected original GGUF packed row shape")
    layer = prepare_gguf_humming(torch.from_numpy(packed).cuda(), kind, torch.bfloat16)
    policy = json.loads(layer.compute_config)
    assert policy["use_batch_invariant"] and not policy["use_f16_accum"]
    default = get_heuristics_config(layer.layer_config, shape_m=1, **policy, device=0)
    x = torch.empty((1, width), device="cuda", dtype=torch.bfloat16)
    rng = torch.Generator(device="cuda").manual_seed(221)
    x.normal_(generator=rng)
    reference, ref_output = capture(lambda: layer(x))
    ref_output.fill_(float("nan"))
    reference.replay()
    if not torch.isfinite(ref_output).all():
        raise ValueError(f"Untouched production left a nonfinite output for {tensor.name}")
    device = torch.cuda.get_device_properties(0)

    graphs = {"production": reference}
    outputs = {"production": ref_output}
    configs = {"production": default}
    # The native GEMM executes the padded grid. Its output contains the logical
    # rows only, as every selected GGUF tensor has a multiple of eight rows.
    padded_rows = layer.layer_config.shape_n
    variants = {"production_explicit": default}
    for cfg in schedules(default, padded_rows, device.multi_processor_count):
        name = f"n{cfg['block_shape'][1]}_wn{cfg['warp_shape'][1]}_k{cfg['block_shape'][2]}_s{cfg['num_stages']}"
        if not mixed or name == "n64_wn32_k64_s3":
            variants[name] = cfg
    for name, cfg in variants.items():
        graph, output = capture(lambda: humming_forward(
            layer.layer_config, inputs=x, weight=layer.weight, weight_scale=layer.weight_scale,
            zero_point=getattr(layer, "zero_point", None), locks=layer.locks, compute_config=layer.compute_config,
            tuning_config=cfg)[..., :rows].reshape(1, rows))
        if output.shape != ref_output.shape:
            raise ValueError("The candidate changed the output shape")
        configs[name], graphs[name], outputs[name] = cfg, graph, output

    errors = {name: {"different_bf16_bits": 0, "nonfinite": 0, "largest_abs_error": 0.0}
              for name in graphs}
    for idx in range(inputs):
        # Half-precision edge cases and ordinary BF16 inputs, always without activation quantization.
        x.normal_(generator=rng).mul_((1.0, 0.125, 8.0)[idx % 3])
        ref_output.fill_(float("nan"))
        reference.replay()
        if not torch.isfinite(ref_output).all():
            raise ValueError(f"Production returned nonfinite values for {tensor.name} trial {idx}")
        for name in list(graphs)[1:]:
            actual = outputs[name]
            actual.fill_(float("nan"))
            graphs[name].replay()
            result = errors[name]
            result["different_bf16_bits"] += torch.count_nonzero(ref_output.view(torch.int16) != actual.view(torch.int16)).item()
            finite = torch.isfinite(actual)
            result["nonfinite"] += torch.count_nonzero(~finite).item()
            finite_errors = torch.where(finite, (ref_output.float()-actual.float()).abs(), 0.0)
            result["largest_abs_error"] = max(result["largest_abs_error"], finite_errors.max().item())
    flush = torch.empty(2 * device.L2_cache_size, dtype=torch.uint8, device="cuda").zero_()
    durations = {name: [] for name in graphs}
    names = list(graphs)
    for turn in range(samples):
        sequence = names[turn % len(names):] + names[:turn % len(names)]
        for name in (sequence if turn % 2 == 0 else reversed(sequence)):
            flush.add_(1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            durations[name].append(begin.elapsed_time(end)*1000.)
    return {"tensor": tensor.name, "quant_type": kind, "physical_rows_input_columns": shape, "padded_output_rows": padded_rows,
            "raw_weight_sha256": hashlib.sha256(packed).hexdigest(),
            "input_rows":inputs, "bf16_values_per_candidate":inputs*rows, "configs":configs,
            "parity":errors,"cold_l2_graph_microseconds":durations,
            "median_microseconds":{k:statistics.median(v) for k,v in durations.items()}}


def main():
    import gguf
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=19)
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--projections", choices=(*PROJECTIONS, "all", "mixed"), default="output-kv")
    parser.add_argument("--device-model", choices=tuple(SUPPORTED_DEVICES), default="NVIDIA GeForce RTX 3090")
    args=parser.parse_args()
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!=args.device_model
            or torch.cuda.get_device_capability(0)!=SUPPORTED_DEVICES[args.device_model]
            or os.environ.get("KAILLM_HUMMING_DETERMINISTIC")!="1" or args.samples<9 or args.inputs<16):
        raise ValueError("Select the specified GPU and the deterministic workload")
    mixed = args.projections == "mixed"
    digest = UDQ2_SHA256 if mixed else CHECKPOINT_SHA256
    with args.checkpoint.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest()!=digest:
            raise ValueError("Wrong Inkling checkpoint")
    tensors={t.name:t for t in gguf.GGUFReader(str(args.checkpoint)).tensors}
    if mixed:
        representatives = {}
        for tensor in tensors.values():
            parts = tensor.name.split(".")
            if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
                continue
            if parts[2] not in ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_rel", "attn_output"):
                continue
            kind = int(tensor.tensor_type)
            if kind not in (8, 12, 13, 14, 18):
                raise ValueError(f"Unexpected original mixed quantization {tensor.name}: {kind}")
            width, rows = map(int, tensor.shape)
            representatives.setdefault((kind, rows, width), (int(parts[1]), parts[2], (rows, width)))
        selected = list(representatives.values())
    else:
        selected = [w for group in PROJECTIONS.values() for w in group] if args.projections == "all" else PROJECTIONS[args.projections]
    for idx, suffix, shape in selected:
        name=f"blk.{idx}.{suffix}.weight"
        result=profile_weight(tensors[name],shape,args.samples,args.inputs,mixed)
        print(json.dumps({"checkpoint_sha256":digest,"device":torch.cuda.get_device_name(0),**result}),flush=True)


if __name__=="__main__":
    main()
