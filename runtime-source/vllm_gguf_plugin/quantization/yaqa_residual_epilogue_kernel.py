# SPDX-License-Identifier: Apache-2.0
"""Exact qualified split-sum JIT; original unsafe function object/math unchanged."""

import triton
import triton.language as tl

from .yaqa_mma import _overwrite_unsafe


@triton.jit
def fused_sum_unsafe(
    Partial,
    X,
    HalfX,
    SafeRows,
    W,
    Lut,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    BITS: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tl.static_assert(SPLITS > 1)
    tl.static_assert(BM == 16 and BN == 32 and BK == 64)
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = batch[:, None] * M + columns[None, :]
    valid = (batch[:, None] < BATCH) & (columns[None, :] < M)
    # Native sum_register_partials starts POSITIVE ZERO, including split0.
    value = tl.full((BM, BN), 0.0, tl.float32)
    for split in tl.static_range(SPLITS):
        current = tl.load(Partial + split * (BATCH * M) + offsets, valid, 0.0)
        value = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, current],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(Y + offsets, value, valid)
    # Different sum/dot layouts may map the same output to different lanes.
    # Restore the old kernel-boundary ordering within this disjoint CTA tile.
    tl.debug_barrier()
    # Exact old min(SafeRows), padded-safe1, and original prepared GEMM call.
    # Neither it nor its GEMM reads Y; an unsafe tile replaces the fast sum.
    _overwrite_unsafe(
        X,
        HalfX,
        SafeRows,
        W,
        Lut,
        Y,
        M=M,
        N=N,
        BATCH=BATCH,
        BITS=BITS,
        BM=BM,
        BN=BN,
        BK=BK,
    )
