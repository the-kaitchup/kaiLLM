# SPDX-License-Identifier: Apache-2.0
"""Compare each target verify query with its standalone decode attention output."""
import argparse
import math
from types import SimpleNamespace

import torch

from vllm_gguf_plugin.inkling.triton_attention import rel_attention_triton


def check(device, window):
    torch.manual_seed(71)
    heads, kv_heads, dim, block, extent = 6, 2, 32, 16, 32
    # The CPU Triton interpreter implements FP16 dot, but interprets BF16 dot
    # operands as their raw integer payload. Exercise the GPU's real BF16 path there.
    dtype = torch.bfloat16 if device == 'cuda' else torch.float16
    cache = torch.randn(21, block, kv_heads, 2*dim, dtype=dtype, device=device).transpose(1, 2)
    def ints(items):
        return torch.tensor(items, device=device, dtype=torch.int32)
    table = torch.randperm(20, device=device, dtype=torch.int32).reshape(2, 10)
    start, lengths = [0, 6, 8], [143, 79]
    positions = torch.cat([torch.arange(n-(b-a), n, device=device) for a,b,n in zip(start,start[1:],lengths)])
    q = torch.randn(8, heads, dim, device=device, dtype=dtype)
    # Integer inputs keep the small relative-bias projection identical across GEMM shapes.
    rel = torch.randint(-2, 3, (8, heads, 4), device=device).float()
    bias = torch.randint(-2, 3, (extent, 4), device=device).float()
    sink = torch.randn(8, heads, device=device)
    def rowmeta(starts, seq, bt, tiles=None, width=1):
        return SimpleNamespace(num_reqs=len(seq),query_start_loc=ints(starts),seq_lens=ints(seq),
                               block_table=bt,tiles=tiles,verify_width=width)
    def apply(rows, md):
        return rel_attention_triton(q[rows],rel[rows],sink[rows],bias,positions[rows],cache,md,
                                    1/math.sqrt(dim),window,extent)
    actual = apply(slice(None),rowmeta(start,lengths,table,width=6))
    assert torch.isfinite(actual).all() and 0 < actual.abs().max() <= cache[..., dim:].abs().max().float() * 1.01
    for req, (a,b) in enumerate(zip(start,start[1:])):
        for i in range(a,b):
            seq=lengths[req]-(b-i)+1
            serial=rowmeta([0,1],[seq],table[req:req+1],tiles=ints([[0,0]]))
            torch.testing.assert_close(actual[i:i+1],apply(slice(i,i+1),serial),rtol=0,atol=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--device',default='cuda');device=cli.parse_args().device
    for window in (None, 87):
        check(device,window)
    print('Every verifier attention row equals its standalone target decode.')
