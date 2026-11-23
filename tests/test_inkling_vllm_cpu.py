# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini through vLLM's real layers on CPU: kaiLLM's GGUF metadata / adapter / loader, the hybrid KV
planner and allocator (48 attention slots + 48 conv states), the Inkling attention builder and vLLM's short-conv
(mamba) builder, stepped like the scheduler (decodes first): one prefill, chunked prefill, token-by-token decode,
two requests batched.

    KAILLM_HY4_ROOT="$PWD" PYTHONPATH="$PWD/tests/gpu/site" CUDA_VISIBLE_DEVICES= \
    python tests/test_inkling_vllm_cpu.py toy.gguf tok.i32 [--llama llama.f32]

Gates: every scenario reproduces the one-pass prefill, agrees with the float64 PyTorch reference about as well as
its float32 run does (KL within 3x, max |dlogit| within 5x, top-1 no worse), and with llama.cpp when given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def compare(name, got, ref):
    got, ref = got.double(), ref.double()
    lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(got, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (got.argmax(-1) == ref.argmax(-1)).double().mean().item()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    jump = (kl > 1e-3).nonzero()
    first = f"  first KL>1e-3 at {int(jump[0])}" if len(jump) else ""
    print(f"{name:10s} tokens {got.shape[0]:4d}  rel max|dlogit| {rel:.2e}  mean KL {kl.mean().item():.2e}  "
          f"top-1 {top1:.4f}{first}")
    return rel, kl.mean().item(), top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("tokens", type=Path)
    ap.add_argument("--llama", type=Path, default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu", help="cuda: kaiLLM's GPU kernels (GGML, new types, ITQ); judged by "
                                                     "KL / top-1 vs the float64 reference (--gpu-kl, default 1e-3)")
    ap.add_argument("--gpu-kl", type=float, default=1e-3)
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(8)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    sys.path.insert(0, str(ROOT / "tests"))
    if a.device == "cpu":
        import inkling_cpu_ops
        inkling_cpu_ops.install()         # kaiLLM's GGML ops are Triton / CUDA only: CPU stand-ins for this test
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader

    ea = EngineArgs(model=str(a.gguf), dtype=a.dtype, enforce_eager=True, max_model_len=256,
                    max_num_batched_tokens=256, max_num_seqs=4, enable_prefix_caching=False)
    vc = ea.create_engine_config()
    if a.device == "cpu":
        vc.device_config.device = torch.device("cpu")
        vc.device_config.device_type = "cpu"
        vc.compilation_config.custom_ops = ["none"]
    else:
        torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}")
    toks = np.fromfile(a.tokens, dtype="<i4").tolist()
    seq_a = toks[:80]
    seq_b = toks[::-1][:50]
    outs = {}
    with set_current_vllm_config(vc), torch.inference_mode():
        ensure_model_parallel_initialized(1, 1)
        model = get_model_loader(vc.load_config).load_model(vllm_config=vc, model_config=vc.model_config)
        from vllm.platforms import current_platform
        current_platform.update_block_size_for_backend(vc)       # attention page >= conv-state page (as the runner)
        print("block size", vc.cache_config.block_size, "mamba block", vc.cache_config.mamba_block_size,
              "mamba page padded", vc.cache_config.mamba_page_size_padded)
        from inkling_runner import Runner
        run = Runner(vc, model, num_blocks=64, device=a.device)
        print("KV cache groups:", run.group_info, "layout", run.layout)
        outs["single"] = run.step([(0, seq_a, 0, True)]); run.release(0)
        cuts = (0, 7, 30, 31, 80)
        outs["chunked"] = torch.cat([run.step([(1, seq_a[s:e], s, True)]) for s, e in zip(cuts[:-1], cuts[1:])])
        run.release(1)
        out = [run.step([(2, seq_a[:60], 0, True)])]
        out += [run.step([(2, seq_a[i:i + 1], i, False)]) for i in range(60, 80)]
        outs["decode"] = torch.cat(out); run.release(2)
        la, lb = run.step([(3, seq_a[:60], 0, True), (4, seq_b[:30], 0, True)]).split([60, 30])
        oa, ob = [la], [lb]
        for i in range(20):
            ya, yb = run.step([(3, seq_a[60 + i:61 + i], 60 + i, False),
                               (4, seq_b[30 + i:31 + i], 30 + i, False)]).split([1, 1])
            oa.append(ya); ob.append(yb)
        outs["batch-a"], outs["batch-b"] = torch.cat(oa), torch.cat(ob)
        outs["single-b"] = run.step([(5, seq_b, 0, True)]); run.release(5)

    sys.path[:0] = [str(ROOT / "runtime-source"), str(ROOT / "runtime-source/gguf-reader"), str(ROOT / "tests")]
    from test_inkling_ref_cpu import load_dense
    from vllm_gguf_plugin.inkling import ref
    hp, w = load_dense(a.gguf)
    vocab = hp.vocab
    with torch.no_grad():
        ref.set_dtype(torch.float64)
        ref_a = ref.InklingReference(hp, w).forward(torch.tensor(seq_a))
        ref_b = ref.InklingReference(hp, w).forward(torch.tensor(seq_b))
        ref.set_dtype(torch.float32)
        r32 = ref.InklingReference(hp, w).forward(torch.tensor(seq_a))
        r32_b = ref.InklingReference(hp, w).forward(torch.tensor(seq_b))
    outs = {k: t.cpu() for k, t in outs.items()}
    v = lambda t: t[:, :vocab]                                           # noqa: E731
    print("reference float32 vs float64 (noise floor per sequence; the toy has near-ties):")
    fa, fb = compare("ref32 a", r32, ref_a), compare("ref32 b", r32_b, ref_b)
    print("kaiLLM (vLLM layers) vs reference float64:")
    res = [(compare(k, v(outs[k]), ref_a), fa) for k in ("single", "chunked", "decode", "batch-a")]
    res.append((compare("batch-b", v(outs["batch-b"]), ref_b), fb))
    if a.llama is not None:
        ll = torch.from_numpy(np.fromfile(a.llama, dtype="<f4").reshape(-1, vocab))[:80]
        print("kaiLLM vs llama.cpp (CPU, F32):")
        compare("single", v(outs["single"]), ll)
    print("self-consistency (multi-step vs one prefill):")
    selfc = [compare(k, v(outs[k]), v(outs["single"])) for k in ("chunked", "decode", "batch-a")]
    selfc.append(compare("batch-b", v(outs["batch-b"]), v(outs["single-b"])))
    if a.device == "cpu":
        # KL within 3x the float32 rounding of the same sequence (+1e-6), max |dlogit| within 5x (ITQ experts add
        # the float32 rounding of their reconstruction), top-1 no worse than float32's own
        ok = all(r <= 5 * max(f[0], 1e-6) and kl <= 3 * max(f[1], 1e-9) + 1e-6 and t >= f[2]
                 for (r, kl, t), f in res)
        ok_self = all(r <= 3 * max(fa[0], 1e-6) and t == 1.0 for r, _, t in selfc)
    else:   # GPU kernels round activations (int8 / BF16): KL bound instead of float32 noise
        ok = all(kl <= a.gpu_kl and t >= 0.97 for (r, kl, t), f in res)
        ok_self = all(kl <= a.gpu_kl and t >= 0.97 for r, kl, t in selfc)
    print(f"vs reference: {'PASS' if ok else 'FAIL'}; self-consistency: {'PASS' if ok_self else 'FAIL'}")
    sys.exit(0 if ok and ok_self else 1)


if __name__ == "__main__":
    main()
