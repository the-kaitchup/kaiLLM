# SPDX-License-Identifier: Apache-2.0
"""Reproduce the observed one-row, eight-expert FP32 mixture reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix(source, weights, destination, dimension: tl.constexpr, block: tl.constexpr):
    token = tl.program_id(1)
    columns = tl.program_id(0) * block + tl.arange(0, block)
    valid = columns < dimension
    sums = ()
    for part in tl.static_range(4):
        low = token * 8 + part
        high = low + 4
        a = tl.load(source + low * dimension + columns, valid, 0) * tl.load(weights + low)
        b = tl.load(source + high * dimension + columns, valid, 0) * tl.load(weights + high)
        sums += (a + b,)
    result = (sums[0] + sums[1]) + (sums[2] + sums[3])
    tl.store(destination + token * dimension + columns, result, valid)


def combine(source, weights, *, block, warps):
    rows, dimension = source.shape
    tokens = weights.shape[0]
    assert rows == tokens * 8 and dimension == 2304 and weights.shape == (tokens, 8)
    assert source.dtype == weights.dtype == torch.float32 and source.is_contiguous() and weights.is_contiguous()
    assert source.is_cuda and source.device == weights.device
    result = torch.empty((tokens, dimension),device=source.device,dtype=torch.float32)
    _mix[(triton.cdiv(dimension,block),tokens)](source,weights,result,dimension,block,
                                                num_warps=warps,num_stages=1,enable_fp_fusion=False)
    return result
