# SPDX-License-Identifier: Apache-2.0
"""Tiled YAQA Hadamard candidates with the original arithmetic and scale types."""

import torch
import triton
import triton.language as tl


@triton.jit
def _power_hadamard_output(
    X,
    Scale,
    Global,
    Y,
    N: tl.constexpr,
    P: tl.constexpr,
    LOG_P: tl.constexpr,
    PRE: tl.constexpr,
    POST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, factor = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, BLOCK)
    offsets = row * N + factor * P + i
    value = tl.load(X + offsets, i < P, 0).to(tl.float32)
    if PRE:
        value *= tl.load(Scale + factor * P + i, i < P, 0)
    for level in tl.static_range(0, LOG_P):
        step = 1 << level
        other = tl.gather(value, i ^ step, 0)
        value = tl.where((i & step) == 0, value + other, other - value)
    value *= N**-0.5
    if POST:
        value *= tl.load(Scale + factor * P + i, i < P, 0)
        value *= tl.load(Global)
    tl.store(Y + offsets, value, i < P)


@triton.jit
def _factor_hadamard_rows(
    X,
    H,
    Scale,
    Global,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    POST: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BP: tl.constexpr,
):
    row = tl.program_id(0)
    a = tl.program_id(1) * BR + tl.arange(0, BR)
    b = tl.arange(0, BK)
    cols = tl.program_id(2) * BP + tl.arange(0, BP)
    hoff = b[None, :] * K + a[:, None] if TRANSPOSE else a[:, None] * K + b[None, :]
    h = tl.load(H + hoff, (a[:, None] < K) & (b[None, :] < K), 0).to(tl.float32)
    value = tl.load(
        X + row * N + b[:, None] * P + cols[None, :],
        (b[:, None] < K) & (cols[None, :] < P),
        0,
    )
    result = tl.dot(h, value, input_precision="tf32x3")
    offset = a[:, None] * P + cols[None, :]
    active = (a[:, None] < K) & (cols[None, :] < P)
    if POST:
        result *= tl.load(Scale + offset, active, 0)
        result *= tl.load(Global)
    tl.store(Y + row * N + offset, result, active)


def hadamard_candidate(
    x,
    factor,
    scale,
    *,
    transpose=False,
    pre=True,
    global_scale=None,
    output_dtype=torch.float32,
    block_rows=16,
    block_cols=16,
    power_num_warps=None,
):
    """Apply the existing factorization, optionally fusing its final scale.

    Input transforms use ``pre=True`` and return FP32. For output transforms,
    pass ``pre=False``, the original scalar ``global_scale`` tensor, and the
    activation output dtype. A float64 global scale stays float64 in the final
    multiplication, matching the standalone output-scaling kernel.
    """
    n = x.shape[-1]
    k = factor.shape[0]
    if (
        not x.is_cuda
        or factor.shape != (k, k)
        or k < 1
        or n % k
        or n // k < 1
        or (n // k) & (n // k - 1)
    ):
        raise ValueError("Invalid CUDA YAQA Hadamard factor geometry")
    if scale.shape != (n,) or factor.device != x.device or scale.device != x.device:
        raise ValueError("YAQA Hadamard factors and scales must match the input")
    post = global_scale is not None
    if post and (pre or global_scale.numel() != 1 or global_scale.device != x.device):
        raise ValueError("YAQA output Hadamard needs one device scalar and pre=False")
    if block_rows not in (16, 32) or block_cols not in (16, 32):
        raise ValueError("YAQA Hadamard row/column tiles must be 16 or 32")
    if power_num_warps is not None and (
        type(power_num_warps) is not int
        or power_num_warps not in (4, 8, 16)
        or k != 1
        or pre
        or not post
    ):
        raise ValueError("Explicit power warp resources require a K1 output transform")
    output = torch.empty(x.shape, dtype=output_dtype, device=x.device)
    if not x.numel():
        return output
    source = x.reshape(-1, n).contiguous()
    factor = factor.contiguous()
    scale = scale.contiguous()
    p = n // k
    # Component-proved wider tiles only for the actual one-request prefill cases.
    if (
        power_num_warps is None
        and torch.cuda.get_device_capability(source.device) == (8, 6)
        and post
        and not pre
        and block_rows == 16
        and block_cols == 32
        and source.shape[0] in (512, 2048)
        and (n, k) in ((6144, 12), (7680, 60))
        and output_dtype in (torch.float32, torch.bfloat16)
    ):
        block_cols = 128 if (n, source.shape[0]) in ((6144, 512), (7680, 2048)) else 64
    scalar = global_scale if post else scale
    temporary = (
        output if k == 1 else torch.empty(x.shape, dtype=torch.float32, device=x.device)
    )
    power_launch_options = (
        {} if power_num_warps is None else {"num_warps": power_num_warps}
    )
    _power_hadamard_output[(source.shape[0], k)](
        source,
        scale,
        scalar,
        temporary,
        N=n,
        P=p,
        LOG_P=p.bit_length() - 1,
        PRE=pre,
        POST=post and k == 1,
        BLOCK=triton.next_power_of_2(p),
        enable_fp_fusion=pre,
        **power_launch_options,
    )
    if k > 1:
        _factor_hadamard_rows[
            (source.shape[0], triton.cdiv(k, block_rows), triton.cdiv(p, block_cols))
        ](
            temporary,
            factor,
            scale,
            scalar,
            output,
            N=n,
            K=k,
            P=p,
            TRANSPOSE=transpose,
            POST=post,
            BR=block_rows,
            BK=max(16, triton.next_power_of_2(k)),
            BP=block_cols,
            num_warps=4,
            enable_fp_fusion=pre,
        )
    return output
