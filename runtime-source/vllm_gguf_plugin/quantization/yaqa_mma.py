# SPDX-License-Identifier: Apache-2.0
"""Optional SM86/SM120 register MMA over original QTIP1--4 circular bitstreams.

The caller prepares exact-half codebook values and per-row input safety flags.
Unsafe input groups are overwritten on device with the existing prepared
kernel's original BK32 mixed-precision fallback. No weight matrix is expanded.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from ..native.cuda_architecture import (
    bind_library_architecture,
    device_architecture,
    resolve_architecture,
    validate_library_device,
)
from .yaqa_cuda import build_cuda_library
from .yaqa_kernels import _yaqa_gemm_fp16_prepared

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_register_mma.cu"
BUILD_CONTRACT = (
    "sm86-qtip1-4-register-mma-small8-attention-split4-8-cap4mib-"
    "prepared-bk32-unsafe-overwrite-v2"
)
_LIBRARIES = {}
_QUERY4_KEYS = set()
_QUERY4_POLICIES = {
    (2, 7680, 2048): (4, 4, 1, 8),
    (2, 2048, 7680): (4, 4, 1, 32),
    (2, 6144, 2048): (4, 4, 1, 16),
    (3, 7680, 2048): (4, 4, 1, 4),
    (3, 2048, 7680): (4, 4, 1, 16),
    (3, 6144, 2048): (4, 4, 1, 16),
    (4, 7680, 2048): (4, 4, 1, 8),
    (4, 2048, 7680): (4, 4, 1, 32),
    (4, 6144, 2048): (4, 4, 1, 16),
}
_PARTIAL_BUDGET = 4 << 20


@lru_cache(maxsize=2)
def _build_library(architecture=None):
    result = build_cuda_library(
        SOURCE, BUILD_CONTRACT, "yaqa_register_mma", architecture=architecture
    )
    mode = os.getenv("VLLM_GGUF_SM120_VERIFY_POLICY", "off")
    if mode not in ("off", "measured-query4-v1"):
        raise ValueError("Invalid SM120 verification policy")
    return {
        **result,
        "internal_query4_measured": mode != "off"
        and resolve_architecture(architecture) == (12, 0),
        "host_policy_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def register_mma_runtime_marker(mode=None):
    mode = os.getenv("VLLM_GGUF_YAQA_GEMM", "tf32x3") if mode is None else mode
    if mode not in ("tf32x3", "native-fp16", "native-fp16-mma"):
        raise ValueError("Unsupported VLLM_GGUF_YAQA_GEMM mode")
    identity = _build_library() if mode == "native-fp16-mma" else {}
    from .inactive_padding import runtime_marker

    return json.dumps(
        {"mode": mode, **identity, "inactive_padding": runtime_marker()},
        sort_keys=True,
    )


def load_register_mma(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = _build_library(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        launch = library.yaqa_register_mma_run
        launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch)
        if identity["internal_query4_measured"]:
            _QUERY4_KEYS.add(key)
    return key


def _launch_policy_sm86(m, n, batch):
    # Actual packed K2/K3/K4 gate/conv projections measured at M4/M8/M16.
    # Retain W4/R4/shared1; only the existing bounded SplitK reduction changes.
    if n == 2048 and m in (6144, 7680) and batch in (4, 8, 16):
        splits = 4 if m == 6144 and batch in (4, 8) else 2
        return 4, 4, 1, splits
    # M16 square/KV uses the independently measured M8 launch parameters.
    if batch == 16 and n == 2048:
        if m == 2048:
            return 4, 8, 1, 4
        if m == 512:
            return 4, 4, 1, 8
    if batch <= 8:
        if n >= 6144:
            warps, ring, splits = 4, 4, 4
        elif n == 2048 and m >= 6144:
            warps, ring, splits = 4, 4, 1
        elif n == 2048 and m >= 2048:
            warps, ring, splits = 4, 8, 4
        elif n == 2048 and m >= 512:
            warps, ring, splits = 4, 4, 8
        else:
            warps, ring, splits = 4, 4, 1
        splits = min(splits, max(1, _PARTIAL_BUDGET // (batch * m * 4)))
        return warps, ring, 1, splits
    warps, splits = (4, 1) if n == 2048 and m >= 6144 else (8, 1)
    if n >= 6144:
        warps = 8
        splits = min(8, max(1, _PARTIAL_BUDGET // (batch * m * 4)))
    return warps, 4, 1, splits


def _occupancy_split_k(m, n, batch, warps, sm_count, budget):
    # Target32 resident warps/SM, with >=256 K-values per partial and the
    # existing partial-buffer ceiling. Every bound uses actual tensor geometry.
    tiles = ((m + 16 * warps - 1) // (16 * warps)) * ((batch + 15) // 16)
    target = sm_count * max(1, 32 // warps)
    limit = min(8, max(1, n // 256), max(1, budget // (batch * m * 4)))
    splits = 1
    while splits * 2 <= limit and tiles * splits < target:
        splits *= 2
    return splits


def _launch_policy_sm120(m, n, batch, multiprocessors=None):
    if batch <= 8:
        if n >= 6144:
            warps, ring, splits = 4, 4, 4
        elif n == 2048 and m >= 6144:
            warps, ring, splits = 4, 4, 1
        elif n == 2048 and m >= 2048:
            warps, ring, splits = 4, 8, 4
        elif n == 2048 and m >= 512:
            warps, ring, splits = 4, 4, 8
        else:
            warps, ring, splits = 4, 4, 1
        splits = min(splits, max(1, _PARTIAL_BUDGET // (batch * m * 4)))
        return warps, ring, 1, splits
    warps, splits = (4, 1) if n == 2048 and m >= 6144 else (8, 1)
    if n >= 6144:
        warps = 8
        splits = min(8, max(1, _PARTIAL_BUDGET // (batch * m * 4)))
    if multiprocessors is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        multiprocessors = props.multi_processor_count
    splits = _occupancy_split_k(m, n, batch, warps, multiprocessors, _PARTIAL_BUDGET)
    return warps, 4, 1, splits


def _launch_policy_for_arch(m, n, batch, capability, multiprocessors=None):
    capability = tuple(capability)
    if capability == (8, 6):
        return _launch_policy_sm86(m, n, batch)
    if capability == (12, 0):
        return _launch_policy_sm120(m, n, batch, multiprocessors)
    raise ValueError("Native MMA policy requires a qualified SM86 or SM120 target")


def _launch_policy(m, n, batch):
    return _launch_policy_for_arch(m, n, batch, resolve_architecture())


@triton.jit
def _overwrite_unsafe(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    safe = tl.load(SafeRows + row, row < BATCH, 1)
    if tl.min(safe, 0) == 0:
        _yaqa_gemm_fp16_prepared(
            X,
            HalfX,
            SafeRows,
            W,
            Lut,
            Lut,  # QTIP ignores the generic codec scale pointer.
            Y,
            M=M,
            N=N,
            BATCH=BATCH,
            CODEC=0,
            BITS=BITS,
            PERMUTE=True,
            BF16=True,
            BM=BM,
            BN=BN,
            BK=BK,
        )


@torch.library.custom_op("vllm_gguf::yaqa_register_mma", mutates_args=())
def register_mma_linear(
    source: torch.Tensor,
    half: torch.Tensor,
    safe_rows: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    signed_half_lut: torch.Tensor,
    m: int,
    n: int,
    bits: int,
    library_key: str,
) -> torch.Tensor:
    """Apply the original permuted QTIP matrix to transformed F32 token rows.

    signed_half_lut must contain the caller-validated exact-half original BF16
    codebook and its first-component sign-flipped copy. safe_rows uses the
    existing _prepare_half_input contract. Both are prepared before this call.
    """
    batch = source.shape[0] if source.ndim == 2 else 0
    tensors = (half, safe_rows, trellis, lut, signed_half_lut)
    if (
        source.device.type != "cuda"
        or source.dtype != torch.float32
        or source.ndim != 2
        or source.shape != (batch, n)
        or batch < 1
        or m <= 0
        or n <= 0
        or m % 16
        or n % 16
        or bits not in (1, 2, 3, 4)
        or batch * m > 2**31 - 256
        or half.dtype != torch.float16
        or half.shape != source.shape
        or safe_rows.dtype != torch.int32
        or safe_rows.shape != (batch,)
        or trellis.dtype != torch.int16
        or trellis.shape != (m * n // 256, 16 * bits)
        or lut.dtype != torch.bfloat16
        or lut.shape != (512, 2)
        or signed_half_lut.dtype != torch.float16
        or signed_half_lut.shape != (1024, 2)
        or not source.is_contiguous()
        or any(t.device != source.device or not t.is_contiguous() for t in tensors)
    ):
        raise ValueError("Invalid original QTIP register-MMA inputs")
    if library_key not in _LIBRARIES:
        raise RuntimeError("Register MMA must load before graph capture")
    validate_library_device(
        library_key, torch.cuda.get_device_capability(source.device)
    )
    # Native packed pair loads require four-byte alignment even for public views.
    if half.data_ptr() % 4:
        half = half.clone()
    if trellis.data_ptr() % 4:
        trellis = trellis.clone()
    if signed_half_lut.data_ptr() % 4:
        signed_half_lut = signed_half_lut.clone()
    warps, ring, shared, splits = _launch_policy(m, n, batch)
    if batch == 4 and library_key in _QUERY4_KEYS:
        measured = _QUERY4_POLICIES.get((bits, m, n))
        if measured is not None:
            warps, ring, shared, splits = measured
    output = torch.empty((batch, m), device=source.device, dtype=torch.float32)
    partial = (
        torch.empty((splits, batch, m), device=source.device, dtype=torch.float32)
        if splits > 1
        else output
    )
    error = _LIBRARIES[library_key][1](
        half.data_ptr(),
        trellis.data_ptr(),
        signed_half_lut.data_ptr(),
        partial.data_ptr(),
        output.data_ptr(),
        m,
        n,
        batch,
        bits,
        warps,
        ring,
        shared,
        splits,
        torch.cuda.current_stream(source.device).cuda_stream,
    )
    if error:
        raise RuntimeError(f"Register MMA launch failed with CUDA error {error}")
    bm = 64 if batch >= 64 else 16
    _overwrite_unsafe[(triton.cdiv(batch, bm), triton.cdiv(m, 32))](
        source,
        half,
        safe_rows,
        trellis,
        lut,
        output,
        M=m,
        N=n,
        BATCH=batch,
        BITS=bits,
        BM=bm,
        BN=32,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    return output


@register_mma_linear.register_fake
def _register_mma_fake(
    source, half, safe_rows, trellis, lut, signed_half_lut, m, n, bits, library_key
):
    return source.new_empty((source.shape[0], m), dtype=torch.float32)
