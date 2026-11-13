# SPDX-License-Identifier: Apache-2.0
"""Unselected fused original Half/safety preparation plus residual Half A."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_high_low(
    X,
    HalfX,
    LowX,
    SafeRows,
    Padding,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    """Keep original high/safety operations; add a separately rounded residual."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + columns, columns < N, 0)
    if HAS_PADDING:
        inactive = tl.load(Padding + row)
        value = tl.where(inactive, 0.0, value)
    safe = tl.max(tl.abs(value), 0) <= 65504.0
    tl.store(SafeRows + row, safe.to(tl.int32))
    tl.store(
        HalfX + row * N + columns,
        value.to(tl.float16, fp_downcast_rounding="rtne"),
        columns < N,
    )
    high = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    low = value - high
    tl.store(
        LowX + row * N + columns,
        low.to(tl.float16, fp_downcast_rounding="rtne"),
        columns < N,
    )


def prepare(source, padding=None):
    if (
        source.device.type != "cuda"
        or source.dtype != torch.float32
        or source.ndim != 2
        or not source.is_contiguous()
        or not 0 < source.shape[1] <= 65536
        or source.shape[0] < 1
    ):
        raise ValueError(
            "Contiguous CUDA F32 rows with bounded positive width required"
        )
    rows, width = source.shape
    if padding is not None and (
        padding.device != source.device
        or padding.dtype != torch.bool
        or padding.shape != (rows,)
        or not padding.is_contiguous()
    ):
        raise ValueError("Actual contiguous device bool inactive-row mask required")
    high = torch.empty_like(source, dtype=torch.float16)
    low = torch.empty_like(source, dtype=torch.float16)
    safe = torch.empty((rows,), device=source.device, dtype=torch.int32)
    _prepare_high_low[(rows,)](
        source,
        high,
        low,
        safe,
        source if padding is None else padding,
        N=width,
        BLOCK=triton.next_power_of_2(width),
        HAS_PADDING=padding is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return high, low, safe
