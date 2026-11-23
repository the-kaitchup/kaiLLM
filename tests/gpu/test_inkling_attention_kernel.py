# SPDX-License-Identifier: Apache-2.0
"""Inkling Triton relative-bias paged attention vs the torch reference (ref.attention per request) on random paged
caches (BF16 and FP8 E4M3): 24 / 8 heads, head dim 128, mixed decode and prefill rows, decode-only batches with and
without the KV split (and CUDA-graph padding rows), sliding (window 1024) and full layers, contexts past the
1,024-distance extent up to 20,000 keys, shuffled blocks.  GPU only.

    $PY tests/gpu/test_inkling_attention_kernel.py
"""

from __future__ import annotations

import itertools
import math
import sys
from types import SimpleNamespace

import torch


def main():
    from vllm_gguf_plugin.inkling import triton_attention as TA
    from vllm_gguf_plugin.inkling.attention import rel_attention_torch
    from vllm_gguf_plugin.inkling.triton_attention import rel_attention_triton
    torch.manual_seed(0)
    H, Hkv, D, nrel, E, bs = 24, 8, 128, 16, 1024, 16
    ok = True
    cases = [([(1, 3000), (1, 700), (40, 40), (130, 2200)], 16), ([(1, 5)] * 16, 16), ([(300, 300)], 16),
             ([(1, 3000), (1, 700), (1, 1), (1, 1025), (1, 64)], 16),
             ([(1, 3000), (1, 700), (1, 1), (1, 1025), (1, 64)], 0),        # decode without the KV split
             ([(1, 20000), (0, 0), (0, 0)], 16)]                            # long context + graph padding rows
    for kv_dtype, window, (case, splits) in itertools.product((torch.bfloat16, torch.float8_e4m3fn), (1024, None),
                                                           cases):
        TA.SPLITS = splits
        reqs = case
        nblocks = sum(-(-L // bs) for _, L in reqs) + 4
        cache = torch.randn(nblocks, bs, Hkv, 2 * D, dtype=torch.bfloat16, device="cuda").to(kv_dtype)
        cache = cache.permute(0, 2, 1, 3)
        perm = torch.randperm(nblocks - 1, device="cuda") + 1
        tables, qsl, seq, pos, used = [], [0], [], [], 0
        for ql, L in reqs:
            nb = -(-L // bs)
            tables.append(perm[used:used + nb]); used += nb
            qsl.append(qsl[-1] + ql); seq.append(L)
            pos.append(torch.arange(L - ql, L))
        width = max(1, max(len(t) for t in tables))
        bt = torch.zeros(len(reqs), width, dtype=torch.int32, device="cuda")
        for i, t in enumerate(tables):
            bt[i, :len(t)] = t
        T = qsl[-1]
        q = (torch.randn(T, H, D, device="cuda") * 0.08).to(torch.bfloat16)
        rel = torch.randn(T, H, nrel, device="cuda") * 0.5
        sink = torch.randn(T, H, device="cuda").to(torch.bfloat16)     # the sink projection's output (model dtype)
        bias = torch.randn(E, nrel, device="cuda")
        md = SimpleNamespace(num_reqs=len(reqs), query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                             query_start_loc=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
                             seq_lens_cpu=torch.tensor(seq, dtype=torch.int32),
                             seq_lens=torch.tensor(seq, dtype=torch.int32, device="cuda"), block_table=bt)
        if all(ql <= 1 for ql, _ in reqs):           # decode-only: the builder's persistent (request, 0) tiles
            md.tiles = torch.stack([torch.arange(len(reqs)), torch.zeros(len(reqs), dtype=torch.long)],
                                   1).to(torch.int32).cuda()
        positions = torch.cat(pos).cuda()
        scale = 1 / math.sqrt(D)
        ref = rel_attention_torch(q, rel, sink, bias, positions, cache, md, scale, window, E).float()
        got = rel_attention_triton(q, rel, sink, bias, positions, cache, md, scale, window, E).float()
        T = qsl[-1]
        rel_err = float((got[:T] - ref[:T]).abs().max() / ref[:T].abs().max())
        good = rel_err < 2e-2
        ok &= good
        print(f"{str(kv_dtype)[6:]:14s} window {window} splits {splits:2d} reqs {case[:4]}"
              f"{'...' if len(case) > 4 else ''}: rel max {rel_err:.2e} {'ok' if good else 'FAIL'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
