# SPDX-License-Identifier: Apache-2.0
"""Default-off merged cached prefills; static product dispatch with no hooks."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=8)
def _sm86(device):
    return torch.cuda.get_device_capability(device) == (8, 6)


@triton.jit(do_not_specialize=["BATCH"], do_not_specialize_on_alignment=["BATCH"])
def _cached_gemm_out(
    X,
    W,
    Y,
    ROW_STRIDE: tl.constexpr,
    COLUMN_OFFSET: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(N, BK)):
        c = start * BK + columns
        x = tl.load(
            X + batch[:, None] * N + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < N),
            0,
        ).to(tl.bfloat16)
        w = tl.load(
            W + rows[None, :] * N + c[:, None],
            (rows[None, :] < M) & (c[:, None] < N),
            0,
        ).to(tl.bfloat16)
        partial = tl.dot(x, w)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=(acc, partial),
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Y + batch[:, None] * ROW_STRIDE + COLUMN_OFFSET + rows[None, :],
        acc,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


@torch.library.custom_op(
    "gguf_yaqa::physical_prefill_cache_mm_out", mutates_args=("out",)
)
def physical_mm_out(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, column_offset: int
) -> None:
    from .yaqa_prefill_cache import row_supported

    if (
        not x.is_cuda
        or not _sm86(x.device)
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or out.ndim != 2
        or weight.dtype != torch.bfloat16
        or out.dtype != x.dtype
        or weight.device != x.device
        or out.device != x.device
        or not weight.is_contiguous()
        or not out.is_contiguous()
        or x.shape[1] != weight.shape[1]
        or out.shape[0] != x.shape[0]
        or not row_supported(x.shape[0])
        or column_offset < 0
        or column_offset + weight.shape[0] > out.shape[1]
    ):
        raise ValueError(
            "Require SM86 BF16 prefill and disjoint columns in contiguous final output"
        )
    m, n = weight.shape
    source = x.contiguous()
    bn = 128 if source.shape[0] == 2048 and m >= 2048 else 64
    _cached_gemm_out[(triton.cdiv(source.shape[0], 64), triton.cdiv(m, bn))](
        source,
        weight,
        out,
        ROW_STRIDE=out.shape[1],
        COLUMN_OFFSET=column_offset,
        M=m,
        N=n,
        BATCH=source.shape[0],
        BM=64,
        BN=bn,
        BK=64,
        num_warps=4,
        num_stages=3,
    )


@physical_mm_out.register_fake
def _fake(x, weight, out, column_offset):
    return None


ENV = "VLLM_GGUF_YAQA_PREFILL_CACHE_MERGED_OUT"


def enabled():
    value = os.getenv(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be0or1")
    return value == "1"


def _cache_selected(x, part):
    from .yaqa_prefill_cache import selected as cache_selected

    return cache_selected(x, part)


def selected(x, parts):
    from .yaqa_prefill_cache import row_supported

    if (
        len(parts) <= 1
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
        or not _sm86(x.device)
    ):
        return False
    if not row_supported(x.numel() // x.shape[-1]):
        return False
    plan = getattr(parts[0], "prefill_cache_plan", None)
    if plan is None or not plan.merged_out:
        return False
    if any(
        getattr(part, "prefill_cache_plan", None) is not plan
        or getattr(part, "prefill_physical_weight", None) is None
        or not _cache_selected(x, part)
        for part in parts
    ):
        return False
    plan.ensure_complete()
    return True


def apply(x, parts):
    if not selected(x, parts):
        raise ValueError(
            "Merged cache requires all selected parts from one complete plan"
        )
    source = x.reshape(-1, x.shape[-1]).contiguous()
    total_columns = sum(part.shape[0] for part in parts)
    out = torch.empty((source.shape[0], total_columns), device=x.device, dtype=x.dtype)
    offset = 0
    for part in parts:
        physical_mm_out(source, part.prefill_physical_weight, out, offset)
        offset += part.shape[0]
    return out.reshape(*x.shape[:-1], total_columns)


def runtime_marker():
    from .yaqa_prefill_cache import GENERIC_MIN_ROWS, rows_mode

    path = Path(__file__).resolve()
    return {
        "contract": "merged-physical-BF16-prefill-RN-out-v1",
        "enabled": enabled(),
        "SM": 86,
        "input_dtype": "bfloat16",
        "rows": [512, 2048] if rows_mode() == "legacy" else None,
        "activation_row_policy": rows_mode(),
        "generic_minimum_rows": GENERIC_MIN_ROWS
        if rows_mode() == "generic-prefill"
        else None,
        "minimum_parts": 2,
        "one_complete_cache_plan_required": True,
        "part_order_and_bias_unchanged": True,
        "declared_mutated_argument": "out",
        "fake_result": None,
        "persistent_extra_weight_bytes": 0,
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
        "compiled_mode3_support": "unqualified; currentmode0only",
    }
