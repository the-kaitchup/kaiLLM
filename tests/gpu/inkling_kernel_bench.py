# SPDX-License-Identifier: Apache-2.0
"""Decode-step kernel timings at Inkling 2 Mini shapes, each op replayed inside a CUDA graph (as the server's decode
steps): the routed-expert down projection (D 2304 x F 1152, 128 experts, top-8; NVFP4 / Q2_0 / MXFP4) through the
grouped kernel (moe_align_block_size + moe_new_type) vs the decode GEMV (moe_new_type_gemv, tile sweep), the loop
controller (rank 64) through torch (compiled when KAILLM_INKLING_COMPILE=1) vs the Triton pair, and each glue
function eager vs compiled (depth mixes over 2 / 4 / 6 states with 1 / 3 queries, attention gate, ...), timed with
20 calls per graph (kernel time) and 1 call per graph (with the replay overhead).  Correctness of the same paths:
test_inkling_moe_kernels.py / test_inkling_glue_compile.py.  GPU only, < 1 GB.

    KAILLM_INKLING_COMPILE=1 $PY tests/gpu/inkling_kernel_bench.py [--tokens 1 2 4 8 16] [--sweep] [--parts moe ctrl glue]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def graph_us(fn, reps: int = 20, rounds: int = 5, replays: int = 1) -> float:
    """Median per-call time (us) of fn called `reps` times inside one CUDA graph (each round replays the graph
    `replays` times: reps=1, replays=300 also counts the replay overhead, as one decode op between others would not)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (reps * replays))
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--sweep", action="store_true", help="GEMV tile sweep (block_n x block_kb x warps)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parts", nargs="+", default=["moe", "ctrl", "glue"])
    a = ap.parse_args()
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    from vllm_gguf_plugin.inkling import glue as G
    from vllm_gguf_plugin.triton.fused_moe import new_types as N
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_inkling_moe_kernels import random_blocks
    gen = np.random.default_rng(0)
    torch.manual_seed(0)
    E, k, D, F, r = 128, 8, 2304, 1152, 64
    res = {"moe_down": [], "controller": [], "glue": []}
    names = {N.NVFP4: "NVFP4", N.Q2_0: "Q2_0", N.MXFP4: "MXFP4"}
    for t in ((N.NVFP4, N.Q2_0, N.MXFP4) if "moe" in a.parts else ()):
        W = random_blocks(t, D, F, E, gen).cuda()
        mb = W[0].numel() / 1e6
        for T in a.tokens:
            ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).cuda().int()
            x = (torch.randn(T * k, F) * 0.5).cuda().to(torch.bfloat16)
            flat = ids.reshape(-1).contiguous()

            def grouped():
                sorted_ids, expert_ids, n_post = moe_align_block_size(ids.reshape(-1, 1), N.moe_block_m(), E)
                return N.moe_new_type(x, W, sorted_ids, expert_ids, n_post, t, D, 1, T * k, out_dtype=torch.float32)

            ref = grouped()
            row = {"type": names[t], "T": T, "rows": T * k, "distinct_MB": round(len(set(flat.tolist())) * mb, 2),
                   "grouped_us": round(graph_us(grouped), 2)}
            default = N.GEMV_TILES[t][:3]
            configs = [(None, None, None)]
            if a.sweep:
                configs += [(bn, kb, w) for bn in (8, 16, 32) for kb in (1, 2, 4) for w in (1, 2, 4)
                            if (bn, kb, w) != default]
            best = None
            for bn, kb, w in configs:
                try:
                    y = N.moe_new_type_gemv(x, W, flat, t, D, 1, block_n=bn, block_kb=kb, num_warps=w)
                    err = float((y - ref).abs().max() / ref.abs().max())
                    us = graph_us(lambda: N.moe_new_type_gemv(x, W, flat, t, D, 1, block_n=bn, block_kb=kb,
                                                              num_warps=w))
                except Exception as e:      # a tile the compiler rejects (registers / shared memory)
                    print(f"  {names[t]} T={T} tile {bn}/{kb}/{w}: {type(e).__name__}", flush=True)
                    continue
                if bn is None:
                    row["gemv_us"], row["gemv_rel_vs_grouped"] = round(us, 2), err
                    bn, kb, w = default
                row.setdefault("tiles", {})[f"{bn},{kb},{w}"] = round(us, 2)
                if best is None or us < best[0]:
                    best = (us, bn, kb, w, err)
            row["best"] = {"us": round(best[0], 2), "block_n": best[1], "block_kb": best[2], "warps": best[3],
                           "rel": best[4]}
            row["gemv_GBps"] = round(T * k * mb * 1e3 / row["gemv_us"], 1)
            res["moe_down"].append(row)
            print(json.dumps(row), flush=True)
        del W
    down = torch.randn(r, 2 * D, device="cuda") / 60
    gain, forget = torch.randn(D, r, device="cuda") / 8, torch.randn(D, r, device="cuda") / 8
    for T in (sorted(set(a.tokens) | {32, 64}) if "ctrl" in a.parts else ()):
        mixed = torch.randn(T, 3, D, device="cuda")
        msgs = torch.randn(T, 4, D, device="cuda") * 0.7
        cur, msg = mixed[:, 1], msgs[:, 2]
        G.CONTROL_KERNEL = False
        torch_us = graph_us(lambda: G.control(cur, msg, down, gain, forget))
        ref = G.control(cur, msg, down, gain, forget)
        G.CONTROL_KERNEL = True
        got = G.control(cur, msg, down, gain, forget)
        row = {"T": T, "torch_us": round(torch_us, 2), "compiled": bool(G.ENABLED),
               "triton_us": round(graph_us(lambda: G.control(cur, msg, down, gain, forget)), 2),
               "rel": float((got - ref).abs().max() / ref.abs().max())}
        res["controller"].append(row)
        print(json.dumps(row), flush=True)
    for T in ((1, 8, 32, 64) if "glue" in a.parts else ()):
        for name, ref_fn, got_fn in glue_cases(T, G):
            row = {"name": name, "T": T, "eager_us": round(graph_us(ref_fn), 2),
                   "compiled_us": round(graph_us(got_fn), 2),
                   "eager_us_1call": round(graph_us(ref_fn, reps=1, rounds=1, replays=300), 2),
                   "compiled_us_1call": round(graph_us(got_fn, reps=1, rounds=1, replays=300), 2)}
            res["glue"].append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


def glue_cases(T, G):
    """(name, eager reference, compiled) at the model's shapes; depth mixes over S states (the model's lists hold
    1-5 blocks + the current state) with 1 query (FFN / output / SWA attention) or 3 (full-attention q / k / v)."""
    from vllm_gguf_plugin.inkling import ref as R
    dev = "cuda"
    D, H, hd = 2304, 24, 128
    out = []
    for S in (2, 4, 6):
        blocks = [torch.randn(T, D, device=dev) for _ in range(S - 1)]
        cur = torch.randn(T, D, device=dev)
        for nq in (1, 3):
            q = torch.randn(nq, D, device=dev)
            out.append((f"depth mix S={S} q={nq}", lambda b=blocks, c=cur, q=q: R.mix(b, c, q),
                        lambda b=blocks, c=cur, q=q: G._fn("mix", c)(b, c, q)))
            if T <= G.MIX_MAX_T:
                from vllm_gguf_plugin.inkling.triton_glue import mix_fused
                out.append((f"depth mix S={S} q={nq} triton", lambda b=blocks, c=cur, q=q: R.mix(b, c, q),
                            lambda b=blocks, c=cur, q=q: mix_fused(b, c, q)))
    att = torch.randn(T, H, hd, device=dev)
    gate = torch.randn(T, H * hd, device=dev).to(torch.bfloat16)
    out.append(("attention gate", lambda: (att.reshape(T, -1) * torch.sigmoid(gate.float())).to(torch.bfloat16),
                lambda: G.gate_attn(att, gate, torch.bfloat16)))
    # Exercise the eight glue operations with the same tensor shapes.
    r_, E, k, F, Hkv, nrel = 64, 128, 8, 1152, 8, 16
    msg = torch.randn(T, D, device=dev)
    down, gn, fg = (torch.randn(r_, 2 * D, device=dev) / 60, torch.randn(D, r_, device=dev) / 8,
                    torch.randn(D, r_, device=dev) / 8)
    out.append(("loop controller", lambda: R.control(cur, msg, down, gn, fg),
                lambda: G.control(cur, msg, down, gn, fg)))
    qq, kk = torch.randn(T, H * hd, device=dev) * 3, torch.randn(T, Hkv * hd, device=dev) * 3
    vv, rl = torch.randn(T, Hkv * hd, device=dev), torch.randn(T, H * nrel, device=dev) * 2
    tau = torch.rand(T, device=dev) + 1
    qc, rc = torch.rand(H, hd, device=dev) + 0.5, torch.rand(H, nrel, device=dev) + 0.5
    args = (qq, kk, vv, rl, qc, rc, tau, 4.0, 1.5, 1e-6, H, Hkv, hd, nrel, True)
    out.append(("post projection", lambda: G._post_projection(*args), lambda: G.post_projection(*args)))
    router, rb = torch.randn(E, D, device=dev) / 50, torch.randn(E, device=dev) / 10
    out.append(("routing", lambda: G._route(cur, router, rb, k, 2.5), lambda: G.route(cur, router, rb, k, 2.5)))
    ids = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
    gu = torch.randn(T * k, 2 * F, device=dev).to(torch.bfloat16)
    sc = torch.rand(E, device=dev) + 0.5
    out.append(("gate_up epilogue", lambda: G._gate_up_epilogue(gu, sc, ids, F, 1.5, 3.0, torch.bfloat16),
                lambda: G.gate_up_epilogue(gu, sc, ids, F, 1.5, 3.0, torch.bfloat16)))
    yy, wts = torch.randn(T * k, D, device=dev), torch.rand(T, k, device=dev)
    out.append(("combine -> f32", lambda: G._combine(yy, wts, sc, ids, torch.float32),
                lambda: G.combine(yy, wts, sc, ids, torch.float32)))
    x = torch.randn(T, D, device=dev)
    norm = torch.rand(D, device=dev) + 0.5
    out.append(("rms -> bf16", lambda: R.rms(x, norm, 1e-6).to(torch.bfloat16),
                lambda: G.rms(x, norm, 1e-6, torch.bfloat16)))
    out.append(("rms pair", lambda: (R.rms(x, norm, 1e-6), R.rms(x, norm, 1e-6).to(torch.bfloat16)),
                lambda: G.rms_pair(x, norm, 1e-6, torch.bfloat16)))
    return out


if __name__ == "__main__":
    sys.exit(main())
