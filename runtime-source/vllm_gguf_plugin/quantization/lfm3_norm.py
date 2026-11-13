# SPDX-License-Identifier: Apache-2.0
"""Fused LFM3 RMS normalization retaining the checkpoint's cast order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lfm3_rms_norm(
    X,
    Weight,
    Residual,
    Y,
    ResidualOut,
    N: tl.constexpr,
    EPSILON: tl.constexpr,
    ACTIVATION_PRECAST: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < N
    value = tl.load(X + row * N + column, active, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(Residual + row * N + column, active, other=0).to(tl.float32)
        value = value + residual
        if X.dtype.element_ty != tl.float32:
            value = value.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(
                tl.float32
            )
        tl.store(ResidualOut + row * N + column, value, active)
    variance = tl.sum(value * value, axis=0) / N
    normalized = value * tl.rsqrt(variance + EPSILON)
    if ACTIVATION_PRECAST and X.dtype.element_ty != tl.float32:
        normalized = normalized.to(X.dtype.element_ty, fp_downcast_rounding="rtne").to(
            tl.float32
        )
    gain = tl.load(Weight + column, active, other=0)
    output = normalized * gain
    if X.dtype.element_ty != tl.float32:
        output = output.to(X.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + row * N + column, output, active)


def fused_lfm3_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    activation_precast: bool = False,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Normalize in FP32, retaining activation rounding and the FP32 gain.

    A residual addition rounds to ``x.dtype`` before computing the variance.
    ``activation_precast`` also rounds the normalized value before multiplying
    by the original FP32 gain. The returned residual is the rounded sum, and
    neither input is modified. Noncontiguous views are copied before execution.
    """
    if not x.is_cuda:
        raise ValueError("Fused LFM3 normalization requires CUDA activations")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("LFM3 normalization requires FP16, BF16, or FP32 activations")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("LFM3 normalization requires a nonempty last dimension")
    n = x.shape[-1]
    if weight.ndim != 1 or weight.shape[0] != n or weight.dtype != torch.float32:
        raise ValueError(
            "LFM3 normalization requires one original FP32 gain per channel"
        )
    if weight.device != x.device:
        raise ValueError("LFM3 activation and gain must share a device")
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("LFM3 residual must match activation shape, dtype, and device")
    output = x.new_empty(x.shape)
    residual_output = x.new_empty(x.shape) if residual is not None else None
    if x.numel():
        source = x.reshape(-1, n).contiguous()
        residual_source = (
            residual.reshape(-1, n).contiguous() if residual is not None else source
        )
        _lfm3_rms_norm[(source.shape[0],)](
            source,
            weight.contiguous(),
            residual_source,
            output,
            residual_output if residual_output is not None else output,
            N=n,
            EPSILON=epsilon,
            ACTIVATION_PRECAST=activation_precast,
            HAS_RESIDUAL=residual is not None,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4 if n <= 4096 else 8,
            enable_fp_fusion=False,
        )
    return output if residual_output is None else (output, residual_output)
