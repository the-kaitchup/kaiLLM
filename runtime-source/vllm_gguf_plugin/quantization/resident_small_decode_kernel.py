# SPDX-License-Identifier: Apache-2.0
"""Unselected SM86 scalar-F32 GEMV over an existing canonical BF16 matrix.

The M1 body is byte-for-byte the qualified portable SM120 arithmetic. The
four-row body adds only row tiling/masks and accepts any positive physical M.
Neither entry point constructs or copies a weight matrix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cached_gemv(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, K: tl.constexpr
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, K)
    x = tl.load(X + cols, cols < N, 0).to(tl.float32)
    weight = tl.load(
        W + rows[:, None] * N + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        0,
    ).to(tl.float32)
    output = tl.sum(weight * x[None, :], 1)
    tl.store(Y + rows, output, rows < M)


@triton.jit(do_not_specialize=["BATCH"])
def _cached_masked(
    X, W, Y, BATCH, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, K: tl.constexpr
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    columns = tl.arange(0, K)
    weight = tl.load(
        W + rows[:, None] * N + columns[None, :],
        (rows[:, None] < M) & (columns[None, :] < N),
        0,
    ).to(tl.float32)
    for part in tl.static_range(4):
        token = tl.program_id(1) * 4 + part
        x = tl.load(X + token * N + columns, (columns < N) & (token < BATCH), 0).to(
            tl.float32
        )
        result = tl.sum(weight * x[None, :], 1)
        tl.store(Y + token * M + rows, result, (rows < M) & (token < BATCH))


def validate_geometry(rows, n, k):
    if (
        not isinstance(rows, int)
        or min(rows, n, k) <= 0
        or rows * k > 2**31 - 1
        or rows * n > 2**31 - 1
        or n * k > 2**31 - 1
    ):
        raise ValueError(
            "Positive physical rows and bounded contiguous matrix indices required"
        )


@torch.library.custom_op("gguf_yaqa::resident_small_decode_mm", mutates_args=())
def cached_linear(
    x: torch.Tensor, weight: torch.Tensor, output_fp32: bool
) -> torch.Tensor:
    if (
        not x.is_cuda
        or torch.cuda.get_device_capability(x.device) != (8, 6)
        or x.dtype not in (torch.float32, torch.bfloat16)
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or x.device != weight.device
        or x.shape[1] != weight.shape[1]
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Existing contiguous SM86 BF16 matrix and raw F32/BF16 input required"
        )
    rows, k = x.shape
    n = weight.shape[0]
    validate_geometry(rows, n, k)
    out = torch.empty(
        (rows, n), device=x.device, dtype=torch.float32 if output_fp32 else x.dtype
    )
    if rows == 1:
        _cached_gemv[(triton.cdiv(n, 4),)](
            x,
            weight,
            out,
            M=n,
            N=k,
            R=4,
            K=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        r = 1 if n <= 16 else 4
        _cached_masked[(triton.cdiv(n, r), triton.cdiv(rows, 4))](
            x,
            weight,
            out,
            rows,
            M=n,
            N=k,
            R=r,
            K=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@cached_linear.register_fake
def _fake(x, weight, output_fp32):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Matrix dimensions differ")
    validate_geometry(x.shape[0], weight.shape[0], weight.shape[1])
    return x.new_empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32 if output_fp32 else x.dtype
    )
