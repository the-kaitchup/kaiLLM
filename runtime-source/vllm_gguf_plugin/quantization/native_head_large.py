# SPDX-License-Identifier: Apache-2.0
"""Unselected generic Q5 head extension; original whole-M compact repair."""

import torch
import triton
import triton.language as tl

from .compact_decode import load_gguf

INT_MAX = 2**31 - 1


def tile_policy(rows, canonical_rounding, canonical_dtype, n):
    if (
        canonical_rounding
        and canonical_dtype in (torch.float16, torch.bfloat16)
        and n >= 32768
    ):
        if rows <= 16:
            return 16, 64, 64
        if rows <= 32:
            return 32, 64, 64
        if rows < 128:
            return 64, 64, 64
        return 128, 32, 64
    return 16, 32, 128


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _sanitize(
    x_ptr,
    clean_ptr,
    unsafe_ptr,
    M,
    K: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCKS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(x_ptr + row * STRIDE_XM + cols * STRIDE_XK, cols < K, other=0).to(
        tl.float32
    )
    absolute = tl.abs(x)
    bounded = tl.min((absolute <= 2047.0).to(tl.int32), axis=1)
    maximum = tl.max(absolute, axis=1)
    scale = (maximum == 0) | (maximum >= 0.0078125)
    safe = tl.min(bounded * scale.to(tl.int32), axis=0) != 0
    tl.store(clean_ptr + row * K + cols, tl.where(safe, x, 1.0), cols < K)
    tl.store(unsafe_ptr + row, (~safe).to(tl.int32))


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _repair_large(
    x_ptr,
    w_ptr,
    y_ptr,
    grid_ptr,
    sign_ptr,
    values_ptr,
    fast_ptr,
    unsafe_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CANONICAL_ROUNDING: tl.constexpr = False,
):
    batches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    unsafe = tl.load(unsafe_ptr + batches, batches < M, other=0)
    mask = (batches[:, None] < M) & (rows[None, :] < N)
    result = tl.load(fast_ptr + batches[:, None] * N + rows[None, :], mask, other=0)
    if tl.sum(unsafe, axis=0) != 0:
        columns = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            k = start * BLOCK_K + columns
            w = load_gguf(
                w_ptr,
                rows[:, None] * K + k[None, :],
                (rows[:, None] < N) & (k[None, :] < K),
                grid_ptr,
                sign_ptr,
                values_ptr,
                QTYPE,
            )
            x = tl.load(
                x_ptr + batches[:, None] * STRIDE_XM + k[None, :] * STRIDE_XK,
                (batches[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.float32)
            if CANONICAL_ROUNDING and x_ptr.dtype.element_ty != tl.float32:
                w = w.to(x_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
                acc = tl.dot(x.to(w.dtype), tl.trans(w), acc)
            else:
                acc = tl.dot(x, tl.trans(w), acc, input_precision="tf32x3")
        result = tl.where(unsafe[:, None] != 0, acc, result)
    tl.store(y_ptr + batches[:, None] * N + rows[None, :], result, mask)


def validate(inputs, packed):
    return (
        inputs.ndim == 2
        and inputs.is_cuda
        and inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and inputs.shape[0] > 16
        and inputs.shape[1] == 2048
        and inputs.shape[0] * 128000 <= INT_MAX
        and packed.device == inputs.device
        and packed.dtype == torch.uint8
        and tuple(packed.shape) == (128000, 1408)
        and packed.is_contiguous()
        and packed.data_ptr() % 128 == 0
    )


@torch.library.custom_op("gguf_yaqa::native_q5_large_head", mutates_args=())
def project(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    grid: torch.Tensor,
    signs: torch.Tensor,
    values: torch.Tensor,
    direct_library: str,
    mma_library: str,
    canonical_rounding: bool,
    canonical_dtype: int,
) -> torch.Tensor:
    if not validate(inputs, packed):
        raise ValueError("Validated native Q5 positive large-row geometry required")
    from .native_head import _native_head_op, _native_mma_head_op

    dtype = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}[canonical_dtype]
    rows, k = inputs.shape
    clean = torch.empty((rows, k), device=inputs.device, dtype=torch.float32)
    unsafe = torch.empty((rows,), device=inputs.device, dtype=torch.int32)
    _sanitize[(rows,)](
        inputs,
        clean,
        unsafe,
        rows,
        K=k,
        STRIDE_XM=inputs.stride(0),
        STRIDE_XK=inputs.stride(1),
        BLOCKS=triton.next_power_of_2(triton.cdiv(k, 32)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    output = torch.empty((rows, 128000), device=inputs.device, dtype=torch.float32)
    for start in range(0, rows, 8):
        count = min(8, rows - start)
        source = clean[start : start + count]
        if count == 1:
            tile = _native_head_op(source, packed, 13, k, direct_library)
        else:
            if count < 8:
                source = torch.nn.functional.pad(source, (0, 0, 0, 8 - count))
            tile = _native_mma_head_op(source, packed, 13, k, mma_library)
        output[start : start + count].copy_(tile[:count])
    bm, bn, bk = tile_policy(rows, canonical_rounding, dtype, 128000)
    _repair_large[(triton.cdiv(rows, bm), triton.cdiv(128000, bn))](
        inputs,
        packed,
        output,
        grid,
        signs,
        values,
        output,
        unsafe,
        M=rows,
        N=128000,
        K=k,
        STRIDE_XM=inputs.stride(0),
        STRIDE_XK=inputs.stride(1),
        QTYPE=13,
        CANONICAL_ROUNDING=canonical_rounding,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output


@project.register_fake
def _fake(
    inputs,
    packed,
    grid,
    signs,
    values,
    direct_library,
    mma_library,
    canonical_rounding,
    canonical_dtype,
):
    return inputs.new_empty((inputs.shape[0], packed.shape[0]), dtype=torch.float32)


def apply(
    prepared, inputs, packed, grid, signs, values, canonical_rounding, canonical_dtype
):
    if not prepared.supports_large_rows(inputs.shape[0]):
        raise ValueError("Prepared native identities do not admit this row geometry")
    dtype = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[canonical_dtype]
    return project(
        inputs,
        packed,
        grid,
        signs,
        values,
        prepared.identity["path"],
        prepared.mma_identity["path"],
        canonical_rounding,
        dtype,
    )
