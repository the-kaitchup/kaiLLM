"""Replay eight expert contributions in the target's measured four-accumulator FP32 order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine_fma4(Y, W, IDS, SCALE, OUT, D: tl.constexpr, BLOCK: tl.constexpr, SCALED: tl.constexpr):
    token = tl.program_id(1)
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pieces = ()
    for j in tl.static_range(4):
        lo = token * 8 + j
        hi = lo + 4
        w0 = tl.load(W + lo).to(tl.float32)
        w1 = tl.load(W + hi).to(tl.float32)
        if SCALED:
            w0 *= tl.load(SCALE + tl.load(IDS + lo))
            w1 *= tl.load(SCALE + tl.load(IDS + hi))
        low = tl.load(Y + lo * D + column).to(tl.float32) * w0
        high = tl.load(Y + hi * D + column).to(tl.float32)
        pieces += (tl.fma(high, w1, low),)
    result = (pieces[0] + pieces[2]) + (pieces[1] + pieces[3])
    tl.store(OUT + token * D + column, result)


def combine_fma4(y, weights, scale, ids):
    tokens = weights.shape[0]
    dimension = y.shape[1]
    out = torch.empty((tokens, dimension), device=y.device, dtype=torch.float32)
    _combine_fma4[dimension // 128, tokens](
        y, weights, ids, scale if scale is not None else weights, out, dimension, 128,
        scale is not None, num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out
