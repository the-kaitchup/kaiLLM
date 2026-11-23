# SPDX-License-Identifier: Apache-2.0
"""Triton MoE GEMMs for the expert formats new in Inkling GGUFs (MXFP4, NVFP4, Q1_0, Q2_0, float, int8 + per-expert
scale) vs a float32 reference built from kaiLLM's bit-exact torch dequantization; both routings (activation shared by
the top-k slots of a token, and one activation row per slot).  MXFP4 / NVFP4 / Q2_0 also through the decode GEMV
(moe_new_type_gemv: one program row per routed row, no sorting).  GPU only.

    $PY tests/gpu/test_inkling_moe_kernels.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def random_blocks(t, rows, cols, E, gen):
    from vllm_gguf_plugin.inkling import ggml_types as G
    qk, size = G.BLOCK[t]
    nb = rows * cols // qk
    raw = gen.integers(0, 256, size=(E * nb, size), dtype=np.uint8)
    if t in (G.Q1_0, G.Q2_0):
        raw[:, :2] = (gen.standard_normal(E * nb) * 0.02).astype(np.float16).view(np.uint8).reshape(-1, 2)
    elif t == G.MXFP4:
        raw[:, 0] = gen.integers(118, 126, size=E * nb)                      # 2^-10 .. 2^-3
    elif t == G.NVFP4:
        raw[:, :4] = gen.integers(0x20, 0x3F, size=(E * nb, 4))              # moderate UE4M3 scales
    return torch.from_numpy(raw).reshape(E, rows, cols // qk * size)


def main():
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    from vllm_gguf_plugin.inkling import ggml_types as G
    from vllm_gguf_plugin.triton.fused_moe import new_types as N
    gen = np.random.default_rng(0)
    torch.manual_seed(0)
    E, k = 16, 4
    ok = True
    for (rows, cols) in ((256, 512), (512, 128 * 9)):
        for T in (1, 5, 64):
            for shared in (True, False):
                ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).cuda().int()
                xr = T if shared else T * k
                x = (torch.randn(xr, cols) * 0.5).cuda().to(torch.bfloat16)
                route = ids if shared else ids.reshape(-1, 1)
                sorted_ids, expert_ids, n_post = moe_align_block_size(route, N.moe_block_m(), E)
                cases = [(t, None) for t in (G.MXFP4, G.NVFP4, G.Q1_0, G.Q2_0)]
                cases += [("bf16", None), ("int8", None)]
                for t, _ in cases:
                    if isinstance(t, int):
                        if cols % G.BLOCK[t][0]:
                            continue
                        W = random_blocks(t, rows, cols, E, gen).cuda()
                        Wf = G.dequantize(W.reshape(E * rows, -1), t).reshape(E, rows, cols)
                        y = N.moe_new_type(x, W, sorted_ids, expert_ids, n_post, t, rows, k if shared else 1, xr,
                                           out_dtype=torch.float32)
                        name = {39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}[t]
                    elif t == "bf16":
                        W = (torch.randn(E, rows, cols) * 0.05).cuda().to(torch.bfloat16)
                        Wf = W.float()
                        y = N.moe_dense(x, W, sorted_ids, expert_ids, rows, k if shared else 1, xr,
                                        out_dtype=torch.float32)
                        name = "BF16"
                    else:
                        W = torch.randint(-127, 128, (E, rows, cols), dtype=torch.int8).cuda()
                        sc = (torch.rand(E) * 0.01 + 0.001).cuda()
                        Wf = W.float() * sc[:, None, None]
                        y = N.moe_dense(x, W, sorted_ids, expert_ids, rows, k if shared else 1, xr, scale=sc,
                                        out_dtype=torch.float32)
                        name = "int8+scale"
                    xa = x.float()[torch.arange(T * k, device="cuda") // k] if shared else x.float()
                    flat = ids.reshape(-1).long()
                    if t == "int8":       # the kernel scales after the dot: int8 weights are exact in BF16
                        ref = torch.einsum("rk,rnk->rn", xa, W.float()[flat]) * sc[flat][:, None]
                    else:                 # dequantized weights rounded to the activation dtype, as the kernels do
                        ref = torch.einsum("rk,rnk->rn", xa, Wf.to(x.dtype).float()[flat])
                    rel = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                    good = rel < 2e-3
                    ok &= good
                    print(f"{name:10s} [{rows}x{cols}] T={T:3d} {'shared' if shared else 'rows  '} rel {rel:.2e} "
                          f"{'ok' if good else 'FAIL'}")
                    if isinstance(t, int) and t in N.GEMV_TYPES:
                        yg = N.moe_new_type_gemv(x, W, ids.reshape(-1).int().contiguous(), t, rows,
                                                 k if shared else 1)
                        relg = float((yg - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                        good = relg < 2e-3
                        ok &= good
                        print(f"{name + ' gemv':10s} [{rows}x{cols}] T={T:3d} {'shared' if shared else 'rows  '} "
                              f"rel {relg:.2e} {'ok' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
