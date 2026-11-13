# SPDX-License-Identifier: Apache-2.0
"""Ampere-compatible packed YAQA kernels; no expanded weight cache."""

import torch
import triton
import triton.language as tl

from .yaqa_sm120_b1_policy import k1_post_eligible


@triton.jit
def _power_hadamard(
    X,
    Scale,
    Y,
    N: tl.constexpr,
    P: tl.constexpr,
    LOG_P: tl.constexpr,
    PRE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    factor = tl.program_id(1)
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
    tl.store(Y + offsets, value, i < P)


@triton.jit
def _factor_hadamard(
    X,
    H,
    Scale,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    POST: tl.constexpr,
    GLOBAL: tl.constexpr,
    BK: tl.constexpr,
    BP: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BP + tl.arange(0, BP)
    a = tl.arange(0, BK)
    b = tl.arange(0, BK)
    hoff = b[None, :] * K + a[:, None] if TRANSPOSE else a[:, None] * K + b[None, :]
    h = tl.load(H + hoff, (a[:, None] < K) & (b[None, :] < K), 0).to(tl.float32)
    value = tl.load(
        X + row * N + b[:, None] * P + cols[None, :],
        (b[:, None] < K) & (cols[None, :] < P),
        0,
    )
    result = tl.dot(h, value, input_precision="tf32x3")
    offset = a[:, None] * P + cols[None, :]
    if POST:
        result *= tl.load(Scale + offset, (a[:, None] < K) & (cols[None, :] < P), 0)
    result *= GLOBAL
    tl.store(Y + row * N + offset, result, (a[:, None] < K) & (cols[None, :] < P))


@triton.jit(
    do_not_specialize=["TOTAL"],
    do_not_specialize_on_alignment=["TOTAL"],
)
def _scale_output(X, Scale, Global, Y, N: tl.constexpr, TOTAL, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < TOTAL, 0)
    value *= tl.load(Scale + i % N, i < TOTAL, 0)
    value *= tl.load(Global)
    tl.store(Y + i, value, i < TOTAL)


def _hadamard(x, factor, scale, transpose=False, pre=True):
    n = x.shape[-1]
    k = factor.shape[0]
    p = n // k
    temporary = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _power_hadamard[(x.numel() // n, k)](
        x,
        scale,
        temporary,
        N=n,
        P=p,
        LOG_P=p.bit_length() - 1,
        PRE=pre,
        BLOCK=triton.next_power_of_2(p),
    )
    if k == 1:
        return temporary
    output = torch.empty_like(temporary)
    _factor_hadamard[(x.numel() // n, triton.cdiv(p, 32))](
        temporary,
        factor,
        scale,
        output,
        N=n,
        K=k,
        P=p,
        TRANSPOSE=transpose,
        POST=False,
        GLOBAL=1.0,
        BK=max(16, triton.next_power_of_2(k)),
        BP=32,
    )
    return output


@triton.jit
def _decode(
    W,
    Lut,
    Scales,
    rows,
    cols,
    N: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
):
    if CODEC == 0:
        tile = (rows // 16) * (N // 16) + cols // 16
        p = (rows % 16) * 16 + cols % 16
        if PERMUTE:
            p = (
                (p // 16 % 8) * 32
                + (p // 2 % 4) * 8
                + (p // 8 % 2) * 4
                + (p // 128) * 2
                + p % 2
            )
        start = (p // 2) * (2 * BITS)
        word, shift = start // 16, start % 16
        lo = tl.load(W + tile * (16 * BITS) + word).to(tl.uint16).to(tl.uint32)
        hi = (
            tl.load(W + tile * (16 * BITS) + (word + 1) % (16 * BITS))
            .to(tl.uint16)
            .to(tl.uint32)
        )
        state = (((lo << 16) | hi) >> (16 - shift)) & 65535
        hashed = state * (state + 1)
        index = (hashed >> 6) & 511
        value = tl.load(Lut + index * 2 + p % 2).to(tl.float32)
        value *= tl.where((p % 2 == 0) & (((hashed >> 15) & 1) != 0), -1.0, 1.0)
    else:
        offset = rows * N + cols
        if CODEC == 1:
            byte = tl.load(W + offset // 5).to(tl.uint8).to(tl.int32)
            power = tl.where(
                offset % 5 == 0,
                1,
                tl.where(
                    offset % 5 == 1,
                    3,
                    tl.where(offset % 5 == 2, 9, tl.where(offset % 5 == 3, 27, 81)),
                ),
            )
            code = (byte // power) % 3 - 1
        else:
            byte = tl.load(W + offset // 2).to(tl.uint8).to(tl.int32)
            code = ((byte >> ((offset % 2) * 4)) & 15) - 8
        value = code.to(tl.float32) * tl.load(Scales + rows).to(tl.float32)
        # Legacy bool flags: 0=FP16, 1=BF16; 2 retains scalar FP32.
        if BF16 == 1:
            value = value.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        elif BF16 == 0:
            value = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    return value


@triton.jit
def _yaqa_gemv(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    # Shape dimensions are all multiples of 16, including the final row block.
    cols = tl.arange(0, BK)
    accum = tl.full((BATCH, ROWS), 0, tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        c = begin * BK + cols
        weight = _decode(
            W,
            Lut,
            Scales,
            tl.minimum(rows[:, None], M - 1),
            tl.minimum(c[None, :], N - 1),
            N,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
        )
        for b in tl.static_range(BATCH):
            value = tl.load(X + b * N + c, c < N, 0).to(tl.float32)
            total = tl.sum(weight * value[None, :], 1)
            accum += tl.where(tl.arange(0, BATCH)[:, None] == b, total[None, :], 0.0)
    tl.store(
        Y + tl.arange(0, BATCH)[:, None] * M + rows[None, :], accum, rows[None, :] < M
    )


@triton.jit
def _yaqa_gemv_vectorized(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    batches = tl.arange(0, BATCH)
    cols = tl.arange(0, BK)
    accum = tl.zeros((BATCH, ROWS), tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        c = begin * BK + cols
        weight = _decode(
            W,
            Lut,
            Scales,
            tl.minimum(rows[:, None], M - 1),
            tl.minimum(c[None, :], N - 1),
            N,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
        )
        value = tl.load(X + batches[:, None] * N + c[None, :], c[None, :] < N, 0).to(
            tl.float32
        )
        accum += tl.sum(weight[None, :, :] * value[:, None, :], axis=2)
    tl.store(Y + batches[:, None] * M + rows[None, :], accum, rows[None, :] < M)


@triton.jit
def _yaqa_gemv_pairs(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
    PACKED_LUT: tl.constexpr = False,
):
    tl.static_assert(CODEC == 0)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    batches = tl.arange(0, BATCH)
    pairs = tl.arange(0, BK)
    accum = tl.zeros((BATCH, ROWS), tl.float32)
    for begin in range(tl.cdiv(N, 2 * BK)):
        cols = (begin * BK + pairs) * 2
        r = tl.minimum(rows[:, None], M - 1)
        c = tl.minimum(cols[None, :], N - 2)
        tile = (r // 16) * (N // 16) + c // 16
        p = (r % 16) * 16 + c % 16
        if PERMUTE:
            p = (
                (p // 16 % 8) * 32
                + (p // 2 % 4) * 8
                + (p // 8 % 2) * 4
                + (p // 128) * 2
            )
        start = (p // 2) * (2 * BITS)
        word, shift = start // 16, start % 16
        lo = tl.load(W + tile * (16 * BITS) + word).to(tl.uint16).to(tl.uint32)
        hi = (
            tl.load(W + tile * (16 * BITS) + (word + 1) % (16 * BITS))
            .to(tl.uint16)
            .to(tl.uint32)
        )
        state = (((lo << 16) | hi) >> (16 - shift)) & 65535
        hashed = state * (state + 1)
        index = (hashed >> 6) & 511
        if PACKED_LUT:
            tl.static_assert(Lut.dtype.element_ty == tl.int32)
            pair = tl.load(Lut + index).to(tl.uint32)
            pair ^= ((hashed >> 15) & 1) << 15
            first = (
                (pair & 65535)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
            )
            second = (
                (pair >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            )
        else:
            first = tl.load(Lut + index * 2).to(tl.float32)
            first *= tl.where(((hashed >> 15) & 1) != 0, -1.0, 1.0)
            second = tl.load(Lut + index * 2 + 1).to(tl.float32)
        x0 = tl.load(X + batches[:, None] * N + cols[None, :], cols[None, :] < N, 0).to(
            tl.float32
        )
        x1 = tl.load(
            X + batches[:, None] * N + cols[None, :] + 1, cols[None, :] + 1 < N, 0
        ).to(tl.float32)
        products = (
            first[None, :, :] * x0[:, None, :] + second[None, :, :] * x1[:, None, :]
        )
        accum += tl.sum(products, axis=2)
    tl.store(Y + batches[:, None] * M + rows[None, :], accum, rows[None, :] < M)


@triton.jit
def _yaqa_pair_rows(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
    PACKED_LUT: tl.constexpr,
):
    row = tl.program_id(1)
    _yaqa_gemv_pairs(
        X + row * N,
        W,
        Lut,
        Scales,
        Y + row * M,
        M,
        N,
        1,
        CODEC,
        BITS,
        PERMUTE,
        BF16,
        ROWS,
        BK,
        PACKED_LUT,
    )


@triton.jit
def _yaqa_scalar_rows(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
    VECTORIZED: tl.constexpr,
):
    row = tl.program_id(1)
    if VECTORIZED:
        _yaqa_gemv_vectorized(
            X + row * N,
            W,
            Lut,
            Scales,
            Y + row * M,
            M,
            N,
            1,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
            ROWS,
            BK,
        )
    else:
        _yaqa_gemv(
            X + row * N,
            W,
            Lut,
            Scales,
            Y + row * M,
            M,
            N,
            1,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
            ROWS,
            BK,
        )


@triton.jit
def _split_bf16x3(value):
    finite = tl.abs(value) < float("inf")
    bounded = tl.minimum(
        tl.maximum(value, -3.3895313892515355e38), 3.3895313892515355e38
    )
    high = tl.where(finite, bounded, value).to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.where(finite, value - high.to(tl.float32), 0.0)
    middle = residual.to(tl.bfloat16, fp_downcast_rounding="rtne")
    low = (residual - middle.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    return high, middle, low


@triton.jit
def _yaqa_gemm_bf16x3(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tl.static_assert(CODEC == 0)
    tl.static_assert(Lut.dtype.element_ty == tl.bfloat16)
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        c = begin * BK + cols
        weight = _decode(
            W,
            Lut,
            Scales,
            tl.minimum(rows[None, :], M - 1),
            tl.minimum(c[:, None], N - 1),
            N,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
        ).to(tl.bfloat16)
        value = tl.load(
            X + batch[:, None] * N + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < N),
            0,
        ).to(tl.float32)
        high, middle, low = _split_bf16x3(value)
        accum = tl.dot(high, weight, accum)
        accum = tl.dot(middle, weight, accum)
        accum = tl.dot(low, weight, accum)
    tl.store(
        Y + batch[:, None] * M + rows[None, :],
        accum,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


@triton.jit
def _yaqa_gemm_fp16_native(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WEIGHT_HALF_EXACT: tl.constexpr = False,
):
    tl.static_assert(CODEC == 0)
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        c = begin * BK + cols
        weight = _decode(
            W,
            Lut,
            Scales,
            tl.minimum(rows[None, :], M - 1),
            tl.minimum(c[:, None], N - 1),
            N,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
        )
        value = tl.load(
            X + batch[:, None] * N + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < N),
            0,
        ).to(tl.float32)
        if Lut.dtype.element_ty == tl.bfloat16:
            half_weight = weight.to(tl.float16)
            if WEIGHT_HALF_EXACT:
                exact_weights = True
            else:
                exact_weights = (
                    tl.min(
                        tl.min((half_weight.to(tl.float32) == weight).to(tl.int32), 0),
                        0,
                    )
                    != 0
                )
            safe_input = tl.max(tl.max(tl.abs(value), 0), 0) <= 65504.0
            if exact_weights & safe_input:
                accum = tl.dot(value.to(tl.float16), half_weight, accum)
            else:
                accum = tl.dot(value, weight, accum, input_precision="tf32x3")
        else:
            accum = tl.dot(value, weight, accum, input_precision="tf32x3")
    tl.store(
        Y + batch[:, None] * M + rows[None, :],
        accum,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


@triton.jit
def _prepare_half_input(X, HalfX, SafeRows, N: tl.constexpr, BLOCK: tl.constexpr):
    """Cast each transformed row once and retain an overflow flag on device."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + columns, columns < N, 0)
    safe = tl.max(tl.abs(value), 0) <= 65504.0
    tl.store(SafeRows + row, safe.to(tl.int32))
    tl.store(
        HalfX + row * N + columns,
        value.to(tl.float16, fp_downcast_rounding="rtne"),
        columns < N,
    )


@triton.jit
def _prepare_half_input_with_padding(
    X, HalfX, SafeRows, Padding, N: tl.constexpr, BLOCK: tl.constexpr
):
    """Cast live rows unchanged; normalize only framework-declared inactive rows."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + columns, columns < N, 0)
    inactive = tl.load(Padding + row)
    value = tl.where(inactive, 0.0, value)
    safe = tl.max(tl.abs(value), 0) <= 65504.0
    tl.store(SafeRows + row, safe.to(tl.int32))
    tl.store(
        HalfX + row * N + columns,
        value.to(tl.float16, fp_downcast_rounding="rtne"),
        columns < N,
    )


@triton.jit(
    do_not_specialize=["BATCH"],
    do_not_specialize_on_alignment=["BATCH"],
)
def _yaqa_gemm_fp16_prepared(
    X,
    HalfX,
    SafeRows,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tl.static_assert(CODEC == 0)
    tl.static_assert(Lut.dtype.element_ty == tl.bfloat16)
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    safe_rows = tl.load(SafeRows + batch, batch < BATCH, 1)
    all_safe = tl.min(safe_rows, 0) != 0
    acc = tl.zeros((BM, BN), tl.float32)
    if all_safe:
        for begin in range(tl.cdiv(N, BK)):
            c = begin * BK + columns
            weight = _decode(
                W,
                Lut,
                Scales,
                tl.minimum(rows[None, :], M - 1),
                tl.minimum(c[:, None], N - 1),
                N,
                CODEC,
                BITS,
                PERMUTE,
                BF16,
            )
            value = tl.load(
                HalfX + batch[:, None] * N + c[None, :],
                (batch[:, None] < BATCH) & (c[None, :] < N),
                0,
            )
            acc = tl.dot(value, weight.to(tl.float16), acc)
    else:
        # Preserve the evaluated path's BK32 decisions when a row overflows.
        fallback_columns = tl.arange(0, 32)
        for begin in range(tl.cdiv(N, 32)):
            c = begin * 32 + fallback_columns
            weight = _decode(
                W,
                Lut,
                Scales,
                tl.minimum(rows[None, :], M - 1),
                tl.minimum(c[:, None], N - 1),
                N,
                CODEC,
                BITS,
                PERMUTE,
                BF16,
            )
            value = tl.load(
                X + batch[:, None] * N + c[None, :],
                (batch[:, None] < BATCH) & (c[None, :] < N),
                0,
            ).to(tl.float32)
            safe = tl.max(tl.max(tl.abs(value), 0), 0) <= 65504.0
            if safe:
                acc = tl.dot(value.to(tl.float16), weight.to(tl.float16), acc)
            else:
                acc = tl.dot(value, weight, acc, input_precision="tf32x3")
    tl.store(
        Y + batch[:, None] * M + rows[None, :],
        acc,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


@triton.jit(
    do_not_specialize=["BATCH"],
    do_not_specialize_on_alignment=["BATCH"],
)
def _yaqa_gemm(
    X,
    W,
    Lut,
    Scales,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH,
    CODEC: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        c = begin * BK + cols
        weight = _decode(
            W,
            Lut,
            Scales,
            tl.minimum(rows[None, :], M - 1),
            tl.minimum(c[:, None], N - 1),
            N,
            CODEC,
            BITS,
            PERMUTE,
            BF16,
        )
        x = tl.load(
            X + batch[:, None] * N + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < N),
            0,
        ).to(tl.float32)
        accum += tl.dot(x, weight, input_precision="tf32x3")
    tl.store(
        Y + batch[:, None] * M + rows[None, :],
        accum,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


def _small_register_mma_eligible(layer, batch):
    """Select by physical rows and actual packed-kernel capabilities.

    The register kernel supports masked row fragments. Its measured advantage
    is the small-matrix domain; larger rows keep the prepared GEMM route.
    """
    return (
        2 <= batch <= 32
        and layer.codec == "qtip"
        and 1 <= int(layer.bits) <= 4
        and layer.permutation
        and min(layer.shape) >= 32
        and layer.shape[0] % 16 == 0
        and layer.shape[1] % 16 == 0
        and getattr(layer, "native_fp16_gemm", False)
        and getattr(layer, "cuda_mma_key", None) is not None
    )


def packed_yaqa_linear(x, layer, prepared_source=None, prepared_residual=None):
    m, n = layer.shape
    source = x.reshape(-1, n).contiguous()
    batch = source.shape[0]
    qtip = layer.codec == "qtip"
    if (
        qtip
        and prepared_source is None
        and getattr(layer, "cuda_hin_high_low_key", None) is not None
    ):
        from .yaqa_hin_high_low import maybe_individual

        prepared = maybe_individual(x, source, layer)
        if prepared is not None:
            prepared_source, high, low, safe = prepared
            prepared_residual = (high, low, safe)
    if prepared_residual is not None:
        if prepared_source is None:
            raise ValueError("Prepared residual requires original transformed RawF32")
        from .yaqa_hin_high_low import validate_prepared

        validate_prepared(prepared_source, prepared_residual, batch, n)
    if prepared_source is not None and (
        not qtip
        or n != 2048
        or layer.hadamard_input.shape != (1, 1)
        or getattr(layer, "cuda_power_key", None) is None
        or not 1 <= batch <= 32
        or not prepared_source.is_cuda
        or prepared_source.device != source.device
        or prepared_source.dtype != torch.float32
        or prepared_source.shape != source.shape
        or not prepared_source.is_contiguous()
    ):
        raise ValueError("Prepared source must be the original CUDA input transform")
    if qtip:
        if prepared_source is not None:
            source = prepared_source
        elif batch <= 32 and getattr(layer, "cuda_power_key", None) is not None:
            from .yaqa_power import cuda_power_input

            resource_key = getattr(layer, "cuda_power_input_resource_key", None)
            if resource_key is not None:
                from .yaqa_power_input_resource import power_input

                source = power_input(
                    source, layer.input_scale, layer.cuda_power_key, resource_key
                )
            else:
                source = cuda_power_input(
                    source, layer.input_scale, layer.cuda_power_key
                )
        elif batch == 1 and layer.hadamard_input.shape[0] > 1:
            from .yaqa_hadamard import hadamard_candidate

            source = hadamard_candidate(
                source,
                layer.hadamard_input,
                layer.input_scale,
                transpose=True,
                pre=True,
                block_cols=32,
            )
        else:
            factor = layer.hadamard_input
            if batch in (512, 2048) and getattr(
                layer, "_native_device_capability", None
            ) == (8, 6):
                factor = getattr(layer, "hadamard_input_prefill", factor)
            source = _hadamard(source, factor, layer.input_scale, True)
        weight, lut, scales = layer.trellis, layer.tlut, layer.SV
        codec = 0
    else:
        weight = layer.packed
        lut = weight
        scales = (
            getattr(layer, "post_output_scale", layer.row_scales)
            if layer.codec == "ternary"
            else layer.scales
        )
        codec = 1 if layer.codec == "ternary" else 2
    from .yaqa_raw_factor_post import maybe_apply as maybe_raw_factor_post

    raw_output = maybe_raw_factor_post(x, source, layer)
    if raw_output is not None:
        return raw_output.reshape(*x.shape[:-1], m)
    result = torch.empty((batch, m), dtype=torch.float32, device=x.device)
    args = (source, weight, lut, scales, result)
    kwargs = dict(
        M=m,
        N=n,
        BATCH=batch,
        CODEC=codec,
        BITS=int(layer.bits),
        PERMUTE=layer.permutation,
        BF16=(
            2
            if not qtip and layer.dtype == torch.float32
            else layer.dtype == torch.bfloat16
        ),
    )
    small_register_mma = _small_register_mma_eligible(layer, batch)
    sm120 = getattr(layer, "_native_device_capability", None) == (12, 0)
    small_fallback_limit = 8 if sm120 else 32
    small_gemm = (
        (3 <= batch <= small_fallback_limit or small_register_mma)
        and m >= 32
        and n >= 32
    )
    from .yaqa_prefill import prefill_tile, transient_prefill_linear

    transient_tile = prefill_tile(layer, batch) if qtip else None
    row_parallel_gate = (
        qtip
        and 1 < batch <= 8
        and m <= 16
        and layer.gemm_precision == "native-fp16-mma"
    )
    if (
        not qtip
        and getattr(layer, "cuda_ternary_key", None) is not None
        and batch <= 8
        and source.dtype == torch.bfloat16
    ):
        from .yaqa_scalar import cuda_ternary

        result = cuda_ternary(source, weight, scales, m, n, layer.cuda_ternary_key)
    elif getattr(layer, "validate_b1", False) or row_parallel_gate:
        if qtip and getattr(layer, "cuda_k34_key", None) is not None:
            from .yaqa_cuda_k34 import cuda_k34_gemv

            result = cuda_k34_gemv(
                source, weight, lut, m, n, int(layer.bits), layer.cuda_k34_key
            )
        elif qtip and getattr(layer, "cuda_k1_key", None) is not None:
            from .yaqa_cuda_k1 import cuda_k1_gemv_rows

            result = cuda_k1_gemv_rows(source, weight, lut, m, n, layer.cuda_k1_key)
        elif qtip and getattr(layer, "cuda_k2_key", None) is not None:
            from .yaqa_cuda import cuda_k2_gemv_rows

            result = cuda_k2_gemv_rows(source, weight, lut, m, n, layer.cuda_k2_key)
        elif qtip:
            packed_lut = layer.tlut.dtype == torch.bfloat16
            pair_lut = layer.tlut.view(torch.int32) if packed_lut else layer.tlut
            pair_rows = 4 if n <= 2048 else 1
            pair_block = (
                min(512, triton.next_power_of_2(n // 2))
                if n <= 2048
                else triton.next_power_of_2(n // 2)
            )
            _yaqa_pair_rows[(triton.cdiv(m, pair_rows), batch)](
                source,
                weight,
                pair_lut,
                scales,
                result,
                **kwargs,
                ROWS=pair_rows,
                BK=pair_block,
                PACKED_LUT=packed_lut,
                num_warps=8 if pair_block > 1024 else 4,
            )
        else:
            vectorized = n % 256 == 0
            block_k = 256 if vectorized else (128 if n % 128 == 0 else 16)
            _yaqa_scalar_rows[(triton.cdiv(m, 4), batch)](
                *args,
                **kwargs,
                ROWS=4,
                BK=block_k,
                VECTORIZED=vectorized,
                num_warps=4,
            )
    elif transient_tile is not None:
        result = transient_prefill_linear(source, layer, transient_tile, output=result)
    elif qtip and batch == 1 and getattr(layer, "cuda_k1_key", None) is not None:
        from .yaqa_cuda_k1 import cuda_k1_gemv

        result = cuda_k1_gemv(source, weight, lut, m, n, layer.cuda_k1_key)
    elif qtip and batch == 1 and getattr(layer, "cuda_k2_key", None) is not None:
        from .yaqa_cuda import cuda_k2_gemv

        result = cuda_k2_gemv(source, weight, lut, m, n, layer.cuda_k2_key)
    elif qtip and batch == 1 and getattr(layer, "cuda_k34_key", None) is not None:
        from .yaqa_cuda_k34 import cuda_k34_gemv

        result = cuda_k34_gemv(
            source, weight, lut, m, n, int(layer.bits), layer.cuda_k34_key
        )
    elif qtip and batch == 1:
        packed_lut = layer.tlut.dtype == torch.bfloat16
        pair_lut = layer.tlut.view(torch.int32) if packed_lut else layer.tlut
        pair_rows = 4 if n <= 2048 else 1
        pair_block = (
            min(512, triton.next_power_of_2(n // 2))
            if n <= 2048
            else triton.next_power_of_2(n // 2)
        )
        _yaqa_gemv_pairs[(triton.cdiv(m, pair_rows),)](
            source,
            weight,
            pair_lut,
            scales,
            result,
            **kwargs,
            ROWS=pair_rows,
            BK=pair_block,
            PACKED_LUT=packed_lut,
            num_warps=8 if pair_block > 1024 else 4,
        )
    elif batch <= 8 and not small_gemm:
        # Static batch dimension must be a power of two for Triton layouts.
        if batch & (batch - 1):
            padded = triton.next_power_of_2(batch)
            source = torch.nn.functional.pad(source, (0, 0, 0, padded - batch))
            result = torch.empty((padded, m), dtype=torch.float32, device=x.device)
            args = (source, weight, lut, scales, result)
            kwargs["BATCH"] = padded
        vectorized = batch <= 2 and n % 256 == 0
        gemv = _yaqa_gemv_vectorized if vectorized else _yaqa_gemv
        block_k = 256 if vectorized else (128 if n % 128 == 0 else 16)
        gemv[(triton.cdiv(m, 4),)](
            *args,
            **kwargs,
            ROWS=4,
            BK=block_k,
            num_warps=4,
        )
        result = result[:batch]
    else:
        block_m, block_n = (64, 32) if batch >= 64 else (16, 32 if small_gemm else 16)
        native_fp16 = qtip and getattr(layer, "native_fp16_gemm", False)
        residual_selected = False
        if native_fp16:
            from .inactive_padding import mask_for_rows

            padding = mask_for_rows(source, batch, role="body")
            if getattr(layer, "cuda_residual_mma_key", None) is not None:
                from .yaqa_residual_mma import selected as select_residual_mma

                residual_selected = select_residual_mma(x, layer)
            if residual_selected:
                from .yaqa_residual_mma import prepare as prepare_residual_mma

                if prepared_residual is not None:
                    half_source, low_source, safe_rows = prepared_residual
                else:
                    half_source, low_source, safe_rows = prepare_residual_mma(
                        source, padding
                    )
            else:
                if prepared_residual is not None:
                    raise ValueError(
                        "Prepared residual no longer selected by original guard"
                    )
                half_source = torch.empty_like(source, dtype=torch.float16)
                safe_rows = torch.empty(
                    (batch,), dtype=torch.int32, device=source.device
                )
                prepare = (
                    _prepare_half_input
                    if padding is None
                    else _prepare_half_input_with_padding
                )
                pointers = (source, half_source, safe_rows)
                if padding is not None:
                    pointers += (padding,)
                prepare[(batch,)](
                    *pointers,
                    N=n,
                    BLOCK=triton.next_power_of_2(n),
                    num_warps=4,
                )
            args = (source, half_source, safe_rows, weight, lut, scales, result)
            gemm = _yaqa_gemm_fp16_prepared
            block_n = 32
            block_k = 64
        else:
            gemm = _yaqa_gemm
            block_k = 32
        if (
            native_fp16
            and (small_register_mma or (sm120 and 3 <= batch <= 8))
            and getattr(layer, "cuda_mma_key", None) is not None
        ):
            from .yaqa_index40 import selected as select_index40

            if select_index40(x, layer):
                from .yaqa_index40 import linear as index40_linear

                result = index40_linear(
                    source,
                    half_source,
                    safe_rows,
                    weight,
                    lut,
                    layer.mma_lut,
                    layer.index40_codes,
                    m,
                    n,
                    int(layer.bits),
                    layer.cuda_index40_key,
                )
            elif residual_selected:
                from .yaqa_residual_mma import residual_mma_linear

                result = residual_mma_linear(
                    source,
                    half_source,
                    low_source,
                    safe_rows,
                    weight,
                    lut,
                    layer.mma_lut,
                    m,
                    n,
                    int(layer.bits),
                    layer.cuda_mma_key,
                    layer.cuda_residual_mma_key,
                )
            else:
                from .yaqa_mma import register_mma_linear

                result = register_mma_linear(
                    source,
                    half_source,
                    safe_rows,
                    weight,
                    lut,
                    layer.mma_lut,
                    m,
                    n,
                    int(layer.bits),
                    layer.cuda_mma_key,
                )
        else:
            gemm[(triton.cdiv(batch, block_m), triton.cdiv(m, block_n))](
                *args,
                **kwargs,
                BM=block_m,
                BN=block_n,
                BK=block_k,
                num_warps=4,
                num_stages=2,
            )
    if qtip and layer.hadamard_output.shape[0] > 1:
        from .yaqa_hadamard import hadamard_candidate

        output = hadamard_candidate(
            result,
            layer.hadamard_output,
            layer.output_scale,
            pre=False,
            global_scale=layer.Wscale,
            output_dtype=x.dtype,
            block_cols=32,
        )
    elif (
        qtip
        and 1 <= batch <= 8
        and x.dtype == torch.bfloat16
        and getattr(layer, "fuse_k1_post", False)
        and getattr(layer, "k1_post_power_num_warps", None) == 16
    ):
        from .yaqa_hadamard import hadamard_candidate

        output = hadamard_candidate(
            result,
            layer.hadamard_output,
            layer.output_scale,
            pre=False,
            global_scale=layer.Wscale,
            output_dtype=x.dtype,
            block_cols=32,
            power_num_warps=16,
        )
    elif (
        qtip
        and getattr(layer, "fuse_k1_post", False)
        and k1_post_eligible(batch, m, x.dtype, layer.Wscale.dtype)
    ):
        from .yaqa_hadamard import hadamard_candidate

        output = hadamard_candidate(
            result,
            layer.hadamard_output,
            layer.output_scale,
            pre=False,
            global_scale=layer.Wscale,
            output_dtype=x.dtype,
            block_cols=32,
        )
    elif qtip:
        result = _hadamard(result, layer.hadamard_output, layer.output_scale, pre=False)
        output = torch.empty_like(result, dtype=x.dtype)
        _scale_output[(triton.cdiv(result.numel(), 256),)](
            result,
            layer.output_scale,
            layer.Wscale,
            output,
            N=m,
            TOTAL=result.numel(),
            BLOCK=256,
        )
    else:
        output = result.to(x.dtype)
    return output.reshape(*x.shape[:-1], m)
