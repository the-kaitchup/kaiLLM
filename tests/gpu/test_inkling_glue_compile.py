# SPDX-License-Identifier: Apache-2.0
"""Inkling glue compiled by torch.compile (inkling/glue.py, KAILLM_INKLING_COMPILE=1) vs the reference ops, at the
model's shapes (D 2304, 24 / 8 heads x 128, 16 relative features, 128 experts top-8, F 1152, loop rank 64), token
counts 1 / 8 / 32 / 64 / 300: max relative error and the number of kernels per call (reference -> compiled), eager
and inside a CUDA graph.  The loop controller (<= 32 tokens) and the depth mix (<= 64 tokens) are the Triton
pairs (triton_glue.py); the controller is fed strided views as in the model (mixed[:, j], messages[:, j]); mixes over
2 / 4 / 6 states with 1 / 3 / 4 queries (the model's lists and query counts).  GPU only, < 1 GB.

    KAILLM_INKLING_COMPILE=1 $PY tests/gpu/test_inkling_glue_compile.py
"""

from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("KAILLM_INKLING_COMPILE", "1")


def kernels(fn, *args):
    from torch.profiler import ProfilerActivity, profile
    fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages()
               if (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) > 0)


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main():
    from vllm_gguf_plugin.inkling import glue as G
    from vllm_gguf_plugin.inkling import ref as R
    assert G.ENABLED, "set KAILLM_INKLING_COMPILE=1"
    torch.manual_seed(0)
    dev = "cuda"
    D, H, Hkv, hd, nrel, E, k, F, r = 2304, 24, 8, 128, 16, 128, 8, 1152, 64
    ok = True
    w = {"query3": torch.randn(3, D, device=dev), "down": torch.randn(r, 2 * D, device=dev) / 60,
         "gain": torch.randn(D, r, device=dev) / 8, "forget": torch.randn(D, r, device=dev) / 8,
         "norm": torch.rand(D, device=dev) + 0.5, "q_clip": torch.rand(H, hd, device=dev) + 0.5,
         "rel_clip": torch.rand(H, nrel, device=dev) + 0.5, "router": torch.randn(E, D, device=dev) / 50,
         "router_bias": torch.randn(E, device=dev) / 10, "scale": torch.rand(E, device=dev) + 0.5}
    for T in (1, 8, 32, 64, 300):
        x = torch.randn(T, D, device=dev)
        blocks = [torch.randn(T, D, device=dev) for _ in range(3)]
        msg = torch.randn(T, D, device=dev)
        mixed = torch.randn(T, 3, D, device=dev)           # the controller's inputs in the model: strided views
        msgs = torch.randn(T, 4, D, device=dev) * 0.7
        att = torch.randn(T, H, hd, device=dev)
        gate = torch.randn(T, H * hd, device=dev).to(torch.bfloat16)
        blocks1 = blocks[:1]
        blocks5 = [torch.randn(T, D, device=dev) * (j + 1) for j in range(5)]
        q1, q4 = torch.randn(1, D, device=dev), torch.randn(4, D, device=dev)
        q, kk = torch.randn(T, H * hd, device=dev) * 3, torch.randn(T, Hkv * hd, device=dev) * 3
        v, rl = torch.randn(T, Hkv * hd, device=dev), torch.randn(T, H * nrel, device=dev) * 2
        tau = torch.rand(T, device=dev) + 1
        ids = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
        gu = torch.randn(T * k, 2 * F, device=dev).to(torch.bfloat16)
        y = torch.randn(T * k, D, device=dev)
        wts = torch.rand(T, k, device=dev)
        cases = [
            ("depth mix", lambda: R.mix(blocks, x, w["query3"]), lambda: G.mix(blocks, x, w["query3"])),
            ("depth mix S2 q1", lambda: R.mix(blocks1, x, q1), lambda: G.mix(blocks1, x, q1)),
            ("depth mix S6 q4", lambda: R.mix(blocks5, x, q4), lambda: G.mix(blocks5, x, q4)),
            ("loop controller", lambda: R.control(x, msg, w["down"], w["gain"], w["forget"]),
             lambda: G.control(x, msg, w["down"], w["gain"], w["forget"])),
            ("controller views", lambda: R.control(mixed[:, 1], msgs[:, 2], w["down"], w["gain"], w["forget"]),
             lambda: G.control(mixed[:, 1], msgs[:, 2], w["down"], w["gain"], w["forget"])),
            ("rms norm", lambda: R.rms(x, w["norm"], 1e-6), lambda: G.rms(x, w["norm"], 1e-6)),
            ("rms -> bf16", lambda: R.rms(x, w["norm"], 1e-6).to(torch.bfloat16),
             lambda: G.rms(x, w["norm"], 1e-6, torch.bfloat16)),
            ("rms pair", lambda: (R.rms(x, w["norm"], 1e-6), R.rms(x, w["norm"], 1e-6).to(torch.bfloat16)),
             lambda: G.rms_pair(x, w["norm"], 1e-6, torch.bfloat16)),
            ("attention gate", lambda: (att.reshape(T, -1) * torch.sigmoid(gate.float())).to(torch.bfloat16),
             lambda: G.gate_attn(att, gate, torch.bfloat16)),
            ("post projection", lambda: G._post_projection(q, kk, v, rl, w["q_clip"], w["rel_clip"], tau, 4.0, 1.5,
                                                           1e-6, H, Hkv, hd, nrel, True),
             lambda: G.post_projection(q, kk, v, rl, w["q_clip"], w["rel_clip"], tau, 4.0, 1.5, 1e-6, H, Hkv, hd,
                                       nrel, True)),
            ("routing", lambda: G._route(x, w["router"], w["router_bias"], k, 2.5),
             lambda: G.route(x, w["router"], w["router_bias"], k, 2.5)),
            ("gate_up epilogue", lambda: G._gate_up_epilogue(gu, w["scale"], ids, F, 1.5, 3.0, torch.bfloat16),
             lambda: G.gate_up_epilogue(gu, w["scale"], ids, F, 1.5, 3.0, torch.bfloat16)),
            ("combine", lambda: G._combine(y, wts, w["scale"], ids, torch.bfloat16),
             lambda: G.combine(y, wts, w["scale"], ids, torch.bfloat16)),
            ("combine -> f32", lambda: G._combine(y, wts, w["scale"], ids, torch.float32),
             lambda: G.combine(y, wts, w["scale"], ids.int(), torch.float32)),
        ]
        for name, ref_fn, got_fn in cases:
            a, b = ref_fn(), got_fn()
            a = a if isinstance(a, (tuple, list)) else (a,)
            b = b if isinstance(b, (tuple, list)) else (b,)
            if name == "routing":                 # ids exact (ties aside; int32 on the GPU), weights close
                err = max(rel(b[1], a[1]), 0.0 if torch.equal(a[0].long(), b[0].long()) else 1.0)
            else:
                err = max(rel(gb, ga) for ga, gb in zip(a, b))
            tol = 2e-2 if name in ("gate_up epilogue", "combine", "rms -> bf16", "rms pair", "attention gate") else 1e-4
            good = err <= tol
            ok &= good
            n_ref, n_got = kernels(ref_fn), kernels(got_fn)
            # the compiled function inside a CUDA graph (decode steps replay graphs)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                got_fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                st = got_fn()
            g.replay()
            torch.cuda.synchronize()
            st = st if isinstance(st, (tuple, list)) else (st,)
            gerr = max(rel(x1, x2) for x1, x2 in zip(st, b) if x1.dtype.is_floating_point)
            ok &= gerr <= tol
            print(f"T={T:3d} {name:17s} rel {err:.1e} graph {gerr:.1e}  kernels {n_ref:3d} -> {n_got:3d} "
                  f"{'ok' if good and gerr <= tol else 'FAIL'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
