# SPDX-License-Identifier: Apache-2.0
"""Unselected component: two original LFM3 RMS rows in one program.

This module has no model hooks, runtime flag, device discovery, or weight cache.
The first RMS has no residual. The second reads the original decoder residual
and returns a fresh residual tensor. Padding rows retain ordinary RMS semantics.
"""

import triton
import triton.language as tl


@triton.jit
def _lfm3_dual_rms_norm(
    X,
    Weight1,
    Weight2,
    Residual,
    Y,
    ResidualOut,
    N: tl.constexpr,
    EPSILON1: tl.constexpr,
    EPSILON2: tl.constexpr,
    ACTIVATION_PRECAST1: tl.constexpr,
    ACTIVATION_PRECAST2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < N
    value = tl.load(X + row * N + column, active, other=0).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / N
    normalized = value * tl.rsqrt(variance + EPSILON1)
    if ACTIVATION_PRECAST1 and X.dtype.element_ty != tl.float32:
        normalized = normalized.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(
            tl.float32
        )
    gain = tl.load(Weight1 + column, active, other=0)
    output = normalized * gain
    if X.dtype.element_ty != tl.float32:
        output = output.to(X.dtype.element_ty, fp_downcast_rounding="rtne")

    # Replace ONLY the first store and second load with their typed boundary.
    value = output.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
    residual = tl.load(Residual + row * N + column, active, other=0).to(tl.float32)
    value = value + residual
    if X.dtype.element_ty != tl.float32:
        value = value.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(ResidualOut + row * N + column, value, active)
    variance = tl.sum(value * value, axis=0) / N
    normalized = value * tl.rsqrt(variance + EPSILON2)
    if ACTIVATION_PRECAST2 and X.dtype.element_ty != tl.float32:
        normalized = normalized.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(
            tl.float32
        )
    gain = tl.load(Weight2 + column, active, other=0)
    output = normalized * gain
    if X.dtype.element_ty != tl.float32:
        output = output.to(X.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + row * N + column, output, active)


def _validate(x, weight1, weight2, residual):
    import torch

    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != 2048:
        raise ValueError("Dual RMS component requires positive [M,2048] activations")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Dual RMS component requires CUDA BF16 or F32 activations")
    for weight in (weight1, weight2):
        if (
            weight.ndim != 1
            or weight.shape[0] != x.shape[1]
            or weight.dtype != torch.float32
            or weight.device != x.device
        ):
            raise ValueError("Each original RMS gain must be matching CUDA F32[N]")
    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("Original decoder residual must match the activation")


def dual_rms(
    x,
    weight1,
    weight2,
    residual,
    epsilon1,
    epsilon2,
    *,
    activation_precast1=False,
    activation_precast2=False,
):
    """Return final normalized activation and fresh residual; inputs are read-only."""
    _validate(x, weight1, weight2, residual)
    source = x.contiguous()
    residual_source = residual.contiguous()
    output = x.new_empty(x.shape)
    residual_output = x.new_empty(x.shape)
    _lfm3_dual_rms_norm[(x.shape[0],)](
        source,
        weight1.contiguous(),
        weight2.contiguous(),
        residual_source,
        output,
        residual_output,
        N=2048,
        EPSILON1=epsilon1,
        EPSILON2=epsilon2,
        ACTIVATION_PRECAST1=activation_precast1,
        ACTIVATION_PRECAST2=activation_precast2,
        BLOCK=2048,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, residual_output
