# SPDX-License-Identifier: Apache-2.0
"""Default-off SM86 BF16 Q8/Q6 fused guard; exact qualified component math."""

import hashlib
import os
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import native_body_extended as baseline
from .compact_decode import load_gguf

LIBRARY_SHA = "8fb97dc6bf54cd0d6a161250e87678a9a809dc40a9b4b6a49be723b0091b3801"
ENV = "VLLM_GGUF_NATIVE_BF16_Q8Q6_POWER2"
WARM_ROWS = (1, 4, 16, 32)
# Power-of-two row scaling is exact for every GGML Q8_1-activation format: the int8
# activations are unchanged and the half d/s scales shift by an exact exponent.
# IQ formats need codebook tables in the repair kernel and keep their own guard.
POWER2_TYPES = (2, 3, 6, 7, 8, 12, 13, 14)


def power2_type(qtype, excluded=()):
    return qtype in POWER2_TYPES and qtype not in excluded


def enabled():
    value = os.getenv(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(ENV + " must be0or1")
    return value == "1"


def supported_geometry(qtype, shape, rows, dtype, capability):
    return (
        type(rows) is int
        and 0 < rows <= 32
        and dtype == torch.bfloat16
        and tuple(capability) == (8, 6)
        and qtype in POWER2_TYPES
        and len(shape) == 2
        and geometry_ok(qtype, shape[0], shape[1])
    )


def geometry_ok(qtype, n, k):
    """Body matrices and vocabulary projections with whole GGML blocks."""
    from .native_body_extended import FORMATS

    return 128 <= n <= 262144 and 0 < k <= 16384 and k % FORMATS[qtype][0] == 0


def supports(prepared, rows):
    return supported_geometry(
        int(prepared.qtype),
        (int(prepared.n), int(prepared.k)),
        rows,
        torch.bfloat16,
        torch.cuda.get_device_capability(prepared.packed.device),
    )


@cache
def _pin(library):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare BF16 decode power2 before capture")
    if hashlib.sha256(Path(library).read_bytes()).hexdigest() != LIBRARY_SHA:
        raise ValueError("BF16 decode power2 requires the pinned original SM86 ABI")
    return True


@triton.jit
def _sanitize_scaled(
    X, Raw, Clean, Unsafe, Inverse, K: tl.constexpr, BLOCKS: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCKS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    tl.store(Raw + row * K + cols, x, cols < K)
    absolute = tl.abs(x)
    finite_blocks = tl.min((absolute <= 2047.0).to(tl.int32), axis=1)
    bounded = tl.min(finite_blocks, axis=0) != 0
    block_amax = tl.max(absolute, axis=1)
    amax = tl.max(block_amax, axis=0)
    old_scale_ok = (block_amax == 0) | (block_amax >= 0.0078125)
    old_safe = bounded & (tl.min(old_scale_ok.to(tl.int32), axis=0) != 0)
    # Extract an exact binary exponent. No log approximation is involved.
    exponent = ((amax.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    can_scale = bounded & (amax >= 2.0**-80) & ~old_safe
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


def layout_policy(layout):
    if layout == 1:
        return {"minimum": 2.0**-80, "power2": False, "name": "D4F32"}
    if layout in (0, 2):
        return {"minimum": 2.0**-7, "power2": True, "name": "HalfDS"}
    raise ValueError("Unqualified native RHS layout")


@triton.jit
def _sanitize_d4(X, Raw, Clean, Unsafe, Inverse, K: tl.constexpr, BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCKS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    tl.store(Raw + row * K + cols, x, cols < K)
    absolute = tl.abs(x)
    bounded = tl.min(tl.min((absolute <= 2047.0).to(tl.int32), axis=1), axis=0) != 0
    maximum = tl.max(absolute, axis=1)
    blocks_ok = (maximum == 0) | (maximum >= 2.0**-80)
    safe = bounded & (tl.min(blocks_ok.to(tl.int32), axis=0) != 0)
    tl.store(Clean + row * K + cols, tl.where(safe, x, 1.0), cols < K)
    tl.store(Unsafe + row, (~safe).to(tl.int32))
    tl.store(Inverse + row, 1.0)


def sanitize(x, layout):
    policy = layout_policy(layout)
    raw = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    clean = torch.empty_like(raw)
    unsafe = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    inverse = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    kernel = _sanitize_scaled if policy["power2"] else _sanitize_d4
    kernel[(x.shape[0],)](
        x,
        raw,
        clean,
        unsafe,
        inverse,
        K=x.shape[1],
        BLOCKS=triton.next_power_of_2(triton.cdiv(x.shape[1], 32)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return raw, clean, unsafe, inverse


@torch.library.custom_op("gguf_yaqa::bf16_q8q6_power2_product", mutates_args=())
def projection(
    x: torch.Tensor,
    packed: torch.Tensor,
    values: torch.Tensor,
    qtype: int,
    k: int,
    library: str,
) -> torch.Tensor:
    if (
        not x.is_cuda
        or torch.cuda.get_device_capability(x.device) != (8, 6)
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_contiguous()
        or x.shape[0] <= 0
        or x.shape[0] > 2147483647
        or x.shape[1] != k
        or qtype not in POWER2_TYPES
        or not geometry_ok(qtype, packed.shape[0], k)
        or packed.device != x.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
    ):
        raise ValueError(
            "Require typed BF16 original-body Q8/Q6 positive-M SM86 inputs"
        )
    _pin(library)
    baseline.validate_layout(
        qtype,
        packed.shape,
        k,
        packed.data_ptr(),
        packed.untyped_storage().nbytes() - packed.storage_offset(),
    )
    # Query the actual pinned ABI. No layout or native row admission is inferred
    # from checkpoint, logical batch, or qtype alone.
    layout = baseline.rhs_layout(library, x.device.index, qtype, x.shape[0])
    raw, clean, unsafe, inverse = sanitize(x, layout)
    output = baseline.native_raw(clean, packed, qtype, k, library)
    rows, n = output.shape
    if rows <= 8:
        _repair_scaled_gemv[(triton.cdiv(n, 4),)](
            raw,
            packed,
            values,
            output,
            unsafe,
            inverse,
            M=rows,
            N=n,
            K=k,
            QTYPE=qtype,
            BLOCK_M=triton.next_power_of_2(rows),
            BLOCK_N=4,
            BLOCK_K=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _repair_scaled_gemm[(triton.cdiv(rows, 16), triton.cdiv(n, 32))](
            raw,
            packed,
            values,
            output,
            unsafe,
            inverse,
            M=rows,
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


@projection.register_fake
def _fake(x, packed, values, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


def apply(x, prepared):
    return projection(
        x,
        prepared.packed,
        prepared.values,
        int(prepared.qtype),
        int(prepared.k),
        str(prepared.library),
    )


def warm(prepared):
    _pin(str(prepared.library))
    with torch.inference_mode():
        for rows in WARM_ROWS:
            if supports(prepared, rows):
                apply(
                    torch.ones(
                        rows,
                        prepared.k,
                        device=prepared.packed.device,
                        dtype=torch.bfloat16,
                    ),
                    prepared,
                )


def marker():
    paths = [
        Path(__file__).resolve(),
        Path(baseline.__file__).resolve(),
        Path(__file__).with_name("compact_decode.py").resolve(),
    ]
    return {
        "contract": "SM86-BF16-Q8Q6-actual-RHS-fused-power2-v1",
        "enabled": enabled(),
        "physical_row_cost_limit": 32,
        "types": [8, 14],
        "warm_rows": list(WARM_ROWS),
        "layout1_minimum": 2.0**-80,
        "layout1_power2": False,
        "HalfDS_minimum": 2.0**-7,
        "HalfDS_power2_minimum_row_amax": 2.0**-80,
        "required_library_sha256": LIBRARY_SHA,
        "checkpoint_or_logical_batch_rules": False,
        "extra_weight_bytes": 0,
        "whole_model_EAR_required": True,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
