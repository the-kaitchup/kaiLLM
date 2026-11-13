# SPDX-License-Identifier: Apache-2.0
"""Opt-in DS4-Half body normalization; original packed weights retained.

Only previously tiny-unsafe finite bounded rows may be newly admitted. Rows
already safe retain the original native arithmetic, and other unsafe rows use
the original compact repair. This is a standalone component, not runtime policy.
"""

import torch
import triton
import triton.language as tl

from . import native_body_extended as baseline
from .compact_decode import load_gguf


@triton.jit
def _sanitize_scaled(X, Clean, Unsafe, Inverse, K: tl.constexpr, BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCKS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + cols, cols < K, other=0)
    absolute = tl.abs(x)
    finite_blocks = tl.min((absolute <= 2047.0).to(tl.int32), axis=1)
    bounded = tl.min(finite_blocks, axis=0) != 0
    block_amax = tl.max(absolute, axis=1)
    amax = tl.max(block_amax, axis=0)
    old_scale_ok = (block_amax == 0) | (block_amax >= 0.0078125)
    old_safe = bounded & (tl.min(old_scale_ok.to(tl.int32), axis=0) != 0)
    # Extract an exact binary exponent. No log approximation is involved.
    exponent = ((amax.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    can_scale = bounded & (amax >= 2.0**-120) & ~old_safe
    new_exponent = tl.minimum(tl.maximum(133 - exponent, 1), 253)
    scale = tl.where(can_scale, (new_exponent << 23).to(tl.float32, bitcast=True), 1.0)
    scaled = x * scale
    scaled_amax = block_amax * scale
    scale_ok = (scaled_amax == 0) | (scaled_amax >= 0.0078125)
    safe = bounded & (tl.min(scale_ok.to(tl.int32), axis=0) != 0)
    # Reciprocal of a normal binary power is exact in this admitted range.
    inverse_bits = (254 - new_exponent) << 23
    inverse = tl.where(safe & can_scale, inverse_bits.to(tl.float32, bitcast=True), 1.0)
    tl.store(Clean + row * K + cols, tl.where(safe, scaled, 1.0), cols < K)
    tl.store(Unsafe + row, (~safe).to(tl.int32))
    tl.store(Inverse + row, inverse)


@triton.jit
def _repair_scaled_gemm(
    x_ptr,
    w_ptr,
    values_ptr,
    y_ptr,
    unsafe_ptr,
    inverse_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Same F32 TF32x3 operations as compact._compact_gemm; no dtype conversion.
    batches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    unsafe = tl.load(unsafe_ptr + batches, batches < M, other=0)
    inverse = tl.load(inverse_ptr + batches, batches < M, other=1.0)
    scaled = (batches < M) & (inverse != 1.0)
    if tl.sum(scaled.to(tl.int32), axis=0) != 0:
        outrows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        outmask = scaled[:, None] & (outrows[None, :] < N)
        output = tl.load(
            y_ptr + batches[:, None] * N + outrows[None, :], outmask, other=0
        )
        tl.store(
            y_ptr + batches[:, None] * N + outrows[None, :],
            output * inverse[:, None],
            outmask,
        )
    if tl.sum(unsafe, axis=0) != 0:
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        columns = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            k = start * BLOCK_K + columns
            w = load_gguf(
                w_ptr,
                rows[:, None] * K + k[None, :],
                (rows[:, None] < N) & (k[None, :] < K),
                w_ptr,
                w_ptr,
                values_ptr,
                QTYPE,
            )
            x = tl.load(
                x_ptr + batches[:, None] * K + k[None, :],
                (batches[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.float32)
            acc = tl.dot(x, tl.trans(w), acc, input_precision="tf32x3")
        tl.store(
            y_ptr + batches[:, None] * N + rows[None, :],
            acc,
            (batches[:, None] < M) & (rows[None, :] < N) & (unsafe[:, None] != 0),
        )


@triton.jit
def _repair_scaled_gemv(
    x_ptr,
    w_ptr,
    values_ptr,
    y_ptr,
    unsafe_ptr,
    inverse_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Same F32 operations/tree as compact._compact_gemv, with guarded stores.
    batches = tl.arange(0, BLOCK_M)
    unsafe = tl.load(unsafe_ptr + batches, batches < M, other=0)
    inverse = tl.load(inverse_ptr + batches, batches < M, other=1.0)
    scaled = (batches < M) & (inverse != 1.0)
    if tl.sum(scaled.to(tl.int32), axis=0) != 0:
        outrows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = scaled[:, None] & (outrows[None, :] < N)
        output = tl.load(y_ptr + batches[:, None] * N + outrows[None, :], mask, other=0)
        tl.store(
            y_ptr + batches[:, None] * N + outrows[None, :],
            output * inverse[:, None],
            mask,
        )
    if tl.sum(unsafe, axis=0) != 0:
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        columns = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            k = start * BLOCK_K + columns
            w = load_gguf(
                w_ptr,
                rows[:, None] * K + k[None, :],
                (rows[:, None] < N) & (k[None, :] < K),
                w_ptr,
                w_ptr,
                values_ptr,
                QTYPE,
            )
            x = tl.load(
                x_ptr + batches[:, None] * K + k[None, :],
                (batches[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        tl.store(
            y_ptr + batches[:, None] * N + rows[None, :],
            acc,
            (batches[:, None] < M) & (rows[None, :] < N) & (unsafe[:, None] != 0),
        )


def sanitize(x):
    clean = torch.empty_like(x)
    unsafe = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    inverse = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _sanitize_scaled[(x.shape[0],)](
        x,
        clean,
        unsafe,
        inverse,
        K=x.shape[1],
        BLOCKS=triton.next_power_of_2(triton.cdiv(x.shape[1], 32)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return clean, unsafe, inverse


@torch.library.custom_op("gguf_yaqa::body_half_power2", mutates_args=())
def guarded_power2(
    x: torch.Tensor,
    packed: torch.Tensor,
    values: torch.Tensor,
    qtype: int,
    k: int,
    library: str,
) -> torch.Tensor:
    rows = x.shape[0]
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not (
            (qtype == 13 and rows in (1, 2, 4, 8, 512, 514, 516, 1024, 2048))
            or (qtype == 14 and rows in (1, 2, 4, 8))
        )
        or not baseline._is_sm86(x.device)
    ):
        raise ValueError("Unqualified power2 body shape/type/device")
    layout = baseline.rhs_layout(library, x.device.index, qtype, rows)
    if layout not in (0, 2):
        raise ValueError("Power2 normalization requires qualified Half RHS layout")
    clean, unsafe, inverse = sanitize(x)
    output = baseline.native_raw(clean, packed, qtype, k, library)
    m, n = output.shape
    if rows <= 8:
        _repair_scaled_gemv[(triton.cdiv(n, 4),)](
            x,
            packed,
            values,
            output,
            unsafe,
            inverse,
            M=m,
            N=n,
            K=k,
            QTYPE=qtype,
            BLOCK_M=triton.next_power_of_2(m),
            BLOCK_N=4,
            BLOCK_K=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _repair_scaled_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            x,
            packed,
            values,
            output,
            unsafe,
            inverse,
            M=m,
            N=n,
            K=k,
            QTYPE=qtype,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return output


@guarded_power2.register_fake
def _fake(x, packed, values, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]))


def apply(x, prepared):
    return guarded_power2(
        x,
        prepared.packed,
        prepared.values,
        int(prepared.qtype),
        int(prepared.k),
        str(prepared.library),
    )
