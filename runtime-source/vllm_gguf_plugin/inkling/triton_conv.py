# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 short convolution of a decode-only batch (one token per request) in one Triton kernel: reads the token
and its state's n_conv - 1 history columns, writes `(x + conv(x) [+ dynamic taps]) / sqrt((2 if dynamic else 1)
n_conv + 1)` (ref.convolve) and shifts the history in place.  Rows whose state index is the null block (CUDA-graph
padding) compute garbage into it, as vLLM's own conv kernels do.  No host synchronisation (CUDA-graph safe).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_decode_kernel(X, W, DYN, ST, SIDX, OUT, C, offset, scale,
                        stride_x, stride_dyn, stride_sb, stride_sc, stride_sk, stride_sidx, stride_out,
                        NC: tl.constexpr, HAS_DYN: tl.constexpr, CH: tl.constexpr, BLOCK_C: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    sidx = tl.load(SIDX + t * stride_sidx).to(tl.int64)
    x = tl.load(X + t * stride_x + c, mask=cm, other=0.0).to(tl.float32)
    st = ST + sidx * stride_sb + (offset + c) * stride_sc
    out = x * tl.load(W + c * NC + (NC - 1), mask=cm, other=0.0)            # tap n_conv - 1: the current token
    if HAS_DYN:
        head = c // CH
        out += x * tl.load(DYN + t * stride_dyn + head * NC, mask=cm, other=0.0).to(tl.float32)          # lag 0
    prev = x
    for j in tl.static_range(NC - 1):
        # history column NC - 2 - j holds x[t - 1 - j]
        h = tl.load(st + (NC - 2 - j) * stride_sk, mask=cm, other=0.0).to(tl.float32)
        out += h * tl.load(W + c * NC + (NC - 2 - j), mask=cm, other=0.0)
        if HAS_DYN:
            out += h * tl.load(DYN + t * stride_dyn + head * NC + (j + 1), mask=cm, other=0.0).to(tl.float32)
        # shift: column NC - 2 - j takes the newer value (the token for the last column)
        tl.store(st + (NC - 2 - j) * stride_sk, prev.to(ST.dtype.element_ty), mask=cm)
        prev = h
    tl.store(OUT + t * stride_out + c, ((x + out) * scale).to(OUT.dtype.element_ty), mask=cm)


def conv_decode(states: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offset: int, sidx: torch.Tensor,
                dynamic: torch.Tensor | None = None, n_heads: int = 1, out_dtype=None) -> torch.Tensor:
    """states [slots, channels, n_conv - 1] (oldest first, any strides), x [T, C] (T rows = T requests; bf16 / f16 /
    f32, widened in the kernel), weight [C, n_conv] float32, sidx [T] state index of each row, dynamic
    [T, n_heads * n_conv] (any float dtype) -> [T, C] out_dtype (default x.dtype)."""
    T, C = x.shape
    n_conv = weight.shape[-1]
    out = torch.empty_like(x, dtype=out_dtype or x.dtype)
    w = weight if (weight.dtype == torch.float32 and weight.is_contiguous()) else weight.float().contiguous()
    dyn = dynamic
    if dyn is not None and dyn.stride(-1) != 1:
        dyn = dyn.contiguous()
    xin = x if x.stride(-1) == 1 else x.contiguous()
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * n_conv + 1)
    BLOCK_C = 256
    _conv_decode_kernel[(T, triton.cdiv(C, BLOCK_C))](
        xin, w, dyn if dyn is not None else w, states, sidx, out, C, int(offset), scale,
        xin.stride(0), dyn.stride(0) if dyn is not None else 0, states.stride(0), states.stride(1), states.stride(2),
        sidx.stride(0), out.stride(0),
        NC=n_conv, HAS_DYN=dynamic is not None, CH=max(C // max(n_heads, 1), 1), BLOCK_C=BLOCK_C, num_warps=4)
    return out
