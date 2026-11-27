"""Screen two-request routed IQ gate/up projections using the original Inkling expert weights."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

from check_inkling_udq2_down import CHECKPOINT, capture, replay_us


GEOMETRIES = ((16, 64, 4), (8, 64, 2), (16, 32, 4), (16, 64, 2), (16, 128, 4),
              (32, 32, 4), (32, 64, 4), (32, 64, 8), (32, 128, 8), (64, 64, 8))


def confirmation(screen, kind, gpu, original, expected_tensors):
    rows = [r for r in screen if r["kind"] == kind]
    if ({r["tensor"] for r in rows} != set(expected_tensors) or len(rows) != len(expected_tensors)
            or any((r["gpu_uuid"], r["batch"], r["checkpoint_sha256"], r["baseline"]) !=
                   (gpu, 2, CHECKPOINT, original) for r in rows)):
        raise ValueError("The initial gate screen used different weights, a different card, or a different batch")
    names = set.intersection(*(set(r["variants"]) for r in rows))
    selected = []
    for name in names:
        if name == "baseline":
            continue
        cold = []
        warm = []
        for row in rows:
            v = row["variants"][name]
            if v["different_bf16_words"]:
                break
            base = row["variants"]["baseline"]
            cold.append(base["cold_us"] / v["cold_us"])
            warm.append(base["warm_us"] / v["warm_us"])
        if len(cold) == len(rows) and min(cold) >= 1.04 and min(warm) >= 1.02:
            selected.append((min(cold), min(warm), name))
    return {max(selected)[2]} if selected else set()


def measure(tensor, gpu, trials, screen, representatives):
    import numpy as np
    import torch
    from triton.runtime.errors import OutOfResources
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.inkling.iq_gate import BLOCKS, measured_tiles, run_projection
    from vllm_gguf_plugin.inkling.moe import ExpertStage

    kind = int(tensor.tensor_type)
    packed = np.asarray(tensor.data)
    if packed.shape != (128, 2304, 9 * BLOCKS[kind][0]) or packed.dtype != np.uint8 or not packed.flags.c_contiguous:
        raise ValueError(f"Unexpected packed layout {tensor.name}")
    weight = torch.from_numpy(packed.copy()).cuda()
    stage = ExpertStage(kind, weight, 2304, 2304, None)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2),
                             parallel_config=SimpleNamespace(world_size_across_dp=1), speculative_config=None)
    stage.iq_gate_tiles = measured_tiles(torch.cuda.get_device_name(), kind, config, 1, weight.shape[-1])
    original = list(stage.iq_gate_tiles[2])
    if original != [1, 16, 64, 4] or ops._cuda_kernel_available("ggml_moe_a8_vec", kind):
        raise ValueError("The installed two-row engine uses an unexpected original kernel")
    tiles = {f"n{n}-k{k}-w{w}": (1, n, k, w) for n, k, w in GEOMETRIES}
    if screen is not None:
        names = confirmation(screen, kind, gpu, original, representatives)
        tiles = {n: t for n, t in tiles.items() if n in names}
    x = torch.zeros((2, 2304), dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros((2, 8), dtype=torch.int64, device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(84355 + int(tensor.name.split(".")[1]))
    graphs = {"baseline": capture(lambda: stage.run(x, ids, True))}
    control, reference = capture(lambda: ops.ggml_moe_a8_vec(x, weight, ids.int(), 8, kind, 2304, 2))
    unavailable = {}
    for name, tile in tiles.items():
        try:
            graphs[name] = capture(lambda tile=tile: run_projection(x, weight, ids, kind, tile))
        except OutOfResources as error:
            unavailable[name] = str(error)
    data = {name: {"cold": [], "warm": [], "different_bf16_words": 0} for name in graphs}
    names = list(graphs)
    scrub = torch.zeros(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.uint8, device="cuda")
    for i in range(trials):
        x.normal_(generator=rng).mul_((0.125, 1., 8.)[i % 3])
        picks = torch.randperm(128, generator=rng, device="cuda")[:16].view(2, 8)
        repeated = (0, 3, 8)[i % 3]
        picks[1, :repeated] = picks[0, :repeated]
        ids.copy_(picks)
        order = names[i % len(names):] + names[:i % len(names)]
        for name in order if i % 2 == 0 else reversed(order):
            graph = graphs[name][0]
            scrub.add_(1)
            data[name]["cold"].append(replay_us(graph))
            data[name]["warm"].append(replay_us(graph))
        # Poison destinations before replay to catch omitted stores.
        reference.fill_(float("nan"))
        control.replay()
        if reference.dtype != torch.bfloat16 or tuple(reference.shape) != (16, 2304) or not torch.isfinite(reference).all():
            raise ValueError("The independent stock projection returned an invalid result")
        for name, (graph, output) in graphs.items():
            output.fill_(float("nan"))
            graph.replay()
            if output.dtype != reference.dtype or not torch.isfinite(output).all():
                raise ValueError("Nonfinite output or dtype change for " + name)
            data[name]["different_bf16_words"] += torch.count_nonzero(
                reference.view(torch.int16) != output.view(torch.int16)).item()
    if data["baseline"]["different_bf16_words"]:
        raise ValueError("Installed engine and the stock projection returned different BF16 words")
    for v in data.values():
        v["cold_us"] = statistics.median(v.pop("cold"))
        v["warm_us"] = statistics.median(v.pop("warm"))
    return dict(kind=kind, tensor=tensor.name, packed_sha256=hashlib.sha256(packed).hexdigest(),
                checkpoint_sha256=CHECKPOINT, gpu_uuid=gpu, device=torch.cuda.get_device_name(), trials=trials,
                batch=2, routed_rows=16, values_per_variant=trials * 16 * 2304, baseline=original,
                variants=data, unavailable=unavailable)


def main():
    import torch
    from gguf import GGUFReader
    from vllm_gguf_plugin.inkling.iq_gate import BLOCKS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device-uuid", required=True)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--confirm-screen", type=Path)
    a = p.parse_args()
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5080"
            or not a.device_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != a.device_uuid
            or not 6 <= a.trials <= 48):
        p.error("Select one physical RTX 5080 and 6–48 trials")
    with a.checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected GGUF")
    groups = defaultdict(list)
    for tensor in GGUFReader(str(a.checkpoint)).tensors:
        if tensor.name.endswith(".ffn_gate_up_exps.weight"):
            groups[int(tensor.tensor_type)].append(tensor)
    if sum(map(len, groups.values())) != 24 or set(groups) != set(BLOCKS):
        raise ValueError("All four gate packings and all 24 original tensors are required")
    screen = json.loads(a.confirm_screen.read_bytes()) if a.confirm_screen else None
    for tensors in groups.values():
        reps = tensors[:1] if len(tensors) == 1 else [tensors[0], tensors[-1]]
        for tensor in tensors if screen is not None else reps:
            print(json.dumps(measure(tensor, a.device_uuid, a.trials, screen, [t.name for t in reps])), flush=True)


if __name__ == "__main__":
    main()
