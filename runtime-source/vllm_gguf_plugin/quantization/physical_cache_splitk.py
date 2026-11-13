# SPDX-License-Identifier: Apache-2.0
"""Canonical BF16 long-K cache with bounded ordered Split-K accumulation."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["BATCH"], do_not_specialize_on_alignment=["BATCH"])
def _partial(
    X,
    W,
    Partial,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    columns = tl.arange(0, BK)
    chunks: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLITS)
    acc = tl.zeros((BM, BN), tl.float32)
    for chunk in range(chunks):
        c = (split * chunks + chunk) * BK + columns
        x = tl.load(
            X + batch[:, None] * K + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < K),
            0,
        ).to(tl.bfloat16)
        w = tl.load(
            W + rows[None, :] * K + c[:, None],
            (rows[None, :] < N) & (c[:, None] < K),
            0,
        ).to(tl.bfloat16)
        block = tl.dot(x, w)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=(acc, block),
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Partial + split * BATCH * N + batch[:, None] * N + rows[None, :],
        acc,
        (batch[:, None] < BATCH) & (rows[None, :] < N),
    )


@triton.jit(do_not_specialize=["BATCH"], do_not_specialize_on_alignment=["BATCH"])
def _ordered_reduce(
    Partial, Y, N: tl.constexpr, BATCH, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < BATCH * N
    value = tl.load(Partial + index, valid, 0)
    for split in tl.static_range(1, SPLITS):
        value = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=(value, tl.load(Partial + split * BATCH * N + index, valid, 0)),
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(Y + index, value, valid)


def _launch(x, weight, output, partial, splits):
    rows, k = x.shape
    n = weight.shape[0]
    if (
        splits not in (1, 2, 4, 8)
        or weight.shape[1] != k
        or tuple(output.shape) != (rows, n)
        or tuple(partial.shape) != (splits, rows, n)
    ):
        raise ValueError(
            "Matching positive tensor geometry and bounded splits required"
        )
    _partial[(triton.cdiv(rows, 16), triton.cdiv(n, 16), splits)](
        x,
        weight,
        partial,
        N=n,
        K=k,
        BATCH=rows,
        BM=16,
        BN=16,
        BK=64,
        SPLITS=splits,
        num_warps=4,
        num_stages=3,
    )
    _ordered_reduce[(triton.cdiv(rows * n, 256),)](
        partial,
        output,
        N=n,
        BATCH=rows,
        SPLITS=splits,
        BLOCK=256,
        num_warps=4,
    )
    return output


MODE_ENV = "VLLM_GGUF_YAQA_SM120_LONG_K_SPLITK"
MAX_PARTIAL_BYTES = 4 << 20


def mode():
    import os

    value = os.getenv(MODE_ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(MODE_ENV + " requires none or auto")
    return value


def split_count(rows, n, k, multiprocessors):
    if (
        not 2 <= rows <= 8
        or n < 2048
        or k <= n
        or n % 16
        or k % 16
        or multiprocessors <= 0
    ):
        return 0
    blocks = triton.cdiv(rows, 16) * triton.cdiv(n, 16)
    splits = 1
    while splits < 8 and blocks * splits < 2 * multiprocessors:
        splits *= 2
    while splits > 1 and splits * rows * n * 4 > MAX_PARTIAL_BYTES:
        splits //= 2
    return splits


@lru_cache(maxsize=8)
def device_geometry(device):
    return (
        torch.cuda.get_device_capability(device),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )


@torch.library.custom_op("gguf_yaqa::physical_sm120_long_k_split_mm", mutates_args=())
def physical_sm120_long_k_split_mm(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.shape[-1] != weight.shape[1]
    ):
        raise ValueError("Require matching SM120 BF16 canonical cache operands")
    capability, processors = device_geometry(x.device)
    rows = x.numel() // x.shape[-1]
    n, k = weight.shape
    splits = split_count(rows, n, k, processors)
    if capability != (12, 0) or not splits:
        raise ValueError("Require actual SM120 and aligned long-K geometry")
    if splits == 1:
        from .yaqa_prefill_cache import physical_sm120_long_k_decode_mm

        return physical_sm120_long_k_decode_mm(x, weight)
    source = x.reshape(rows, k).contiguous()
    output = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    partial = torch.empty((splits, rows, n), device=x.device, dtype=torch.float32)
    _launch(source, weight, output, partial, splits)
    return output.reshape(*x.shape[:-1], n)


@physical_sm120_long_k_split_mm.register_fake
def _fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
