# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM86 GGML body over unchanged Q8_0/Q6_K wire bytes.

Safe rows use native Q8_1 activation quantization. Unsafe rows retain the
original compact FP32 arithmetic. This deliberately changes safe-row arithmetic.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import native_head
from .compact_decode import load_gguf

FORMATS = {8: (32, 34), 14: (256, 210)}
SELECTED_ROWS = (1, 2, 8, 16, 512)
MAX_ABS = 2047.0
MIN_NONZERO_BLOCK_AMAX = 2**-7
_WARMED = set()


def native_body_mode():
    mode = os.getenv("VLLM_GGUF_NATIVE_BODY", "none")
    if mode not in ("none", "ggml-q8_0-q6_k", "ggml-extended-b1"):
        raise ValueError(
            "VLLM_GGUF_NATIVE_BODY must be none, ggml-q8_0-q6_k, or ggml-extended-b1"
        )
    return mode


def native_body_eligible(mode, dtype, qtype, shape, is_cuda):
    return (
        mode == "ggml-q8_0-q6_k"
        and dtype == torch.float32
        and qtype in FORMATS
        and is_cuda
        and len(shape) == 2
        and 128 <= shape[0] <= 7680
        and shape[1] in (2048, 7680)
    )


@lru_cache(maxsize=8)
def _is_sm86(device):
    return torch.cuda.get_device_capability(device) == (8, 6)


def validate_layout(qtype, shape, k, pointer, available_bytes):
    if qtype not in FORMATS or len(shape) != 2:
        raise ValueError("Candidate requires original Q8_0 or Q6_K wire bytes")
    block, size = FORMATS[qtype]
    n, row_bytes = shape
    if k <= 0 or k % block or row_bytes != k // block * size or n <= 0:
        raise ValueError("Packed GGUF shape does not match K/type")
    # Restrict this experiment to the actual LFM3 body; vocabulary stays unchanged.
    if n > 7680 or k not in (2048, 7680):
        raise ValueError("Shape is outside the initial LFM3 body experiment")
    tail = (((-k) % 512) + ((-n) % 4) * k) // block * size
    if pointer % 128 or available_bytes < n * row_bytes + tail:
        raise ValueError("Packed storage needs GGML 128B alignment and tail padding")
    return tail


@triton.jit
def _sanitize_rows(
    x_ptr,
    clean_ptr,
    unsafe_ptr,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    LIMIT: tl.constexpr,
    MIN_AMAX: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCKS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(x_ptr + row * K + cols, cols < K, other=0)
    absolute = tl.abs(x)
    # <= also rejects NaN/Inf; a row maximum alone misses tiny nonzero blocks.
    finite_bounded = tl.min((absolute <= LIMIT).to(tl.int32), axis=1)
    block_amax = tl.max(absolute, axis=1)
    valid_scale = (block_amax == 0) | (block_amax >= MIN_AMAX)
    safe = tl.min(finite_bounded * valid_scale.to(tl.int32), axis=0) != 0
    tl.store(unsafe_ptr + row, (~safe).to(tl.int32))
    # Native kernels never see unsafe values. Ones are finite and quantize safely.
    tl.store(clean_ptr + row * K + cols, tl.where(safe, x, 1.0), cols < K)


@triton.jit
def _repair_gemv(
    x_ptr,
    w_ptr,
    y_ptr,
    unsafe_ptr,
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
                w_ptr,
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
def _repair_gemm(
    x_ptr,
    w_ptr,
    y_ptr,
    unsafe_ptr,
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
                w_ptr,
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


def sanitize(x):
    clean = torch.empty_like(x)
    unsafe = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    _sanitize_rows[(x.shape[0],)](
        x,
        clean,
        unsafe,
        K=x.shape[1],
        BLOCKS=triton.next_power_of_2(triton.cdiv(x.shape[1], 32)),
        LIMIT=MAX_ABS,
        MIN_AMAX=MIN_NONZERO_BLOCK_AMAX,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return clean, unsafe


def native_raw(x, packed, qtype, k, library):
    """Existing C ABI, no copy/quantization reimplementation or synchronization."""
    lib = native_head._library(library)
    rows, n = x.shape[0], packed.shape[0]
    output = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        args = (x.device.index, qtype, n, k, rows)
        size = lib.current_direct_workspace(*args)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        status = lib.current_direct_run(
            packed.data_ptr(),
            x.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            size,
            *args,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if status:
            raise RuntimeError(lib.current_direct_error().decode())
    return output


@torch.library.custom_op("gguf_yaqa::native_body_linear", mutates_args=())
def guarded_native(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int, library: str
) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or x.shape[0] not in SELECTED_ROWS
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
    ):
        raise ValueError(
            "Guarded native body expects CUDA F32 selected rows and packed bytes"
        )
    validate_layout(
        qtype,
        packed.shape,
        k,
        packed.data_ptr(),
        packed.untyped_storage().nbytes() - packed.storage_offset(),
    )
    if not _is_sm86(x.device):
        raise ValueError("Guarded native body requires SM86")
    clean, unsafe = sanitize(x)
    output = native_raw(clean, packed, qtype, k, library)
    m, n = output.shape
    if m <= 8:
        _repair_gemv[(triton.cdiv(n, 4),)](
            x,
            packed,
            output,
            unsafe,
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
        _repair_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            x,
            packed,
            output,
            unsafe,
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


@guarded_native.register_fake
def _fake(x, packed, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


@lru_cache(maxsize=1)
def _enabled_marker():
    identity = native_head.native_head_library_identity()
    metadata = Path(identity["path"]).with_suffix(".build.json")
    files = [
        Path(__file__),
        Path(__file__).with_name("compact.py"),
        Path(__file__).with_name("compact_decode.py"),
        Path(native_head.__file__),
    ]
    return {
        "mode": "ggml-q8_0-q6_k",
        "contract": "original-wire-sm86-guarded-q8_1-v1",
        "library": identity,
        "build_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "compiler_command": json.loads(metadata.read_text()).get("compile"),
        "sources": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "SM": 86,
        "qtypes": sorted(FORMATS),
        "activation_rows": list(SELECTED_ROWS),
        "output_rows": [128, 7680],
        "input_columns": [2048, 7680],
        "guard": {
            "max_abs": MAX_ABS,
            "min_nonzero_32block_amax": MIN_NONZERO_BLOCK_AMAX,
        },
        "persistent_additional_weight_bytes": 0,
        "activation_quantization_changes_arithmetic": True,
    }


def native_body_cache_marker():
    # Default-off must not require a native library, its metadata, or CUDA.
    mode = native_body_mode()
    if mode == "none":
        return {"mode": "none"}
    if mode == "ggml-extended-b1":
        from .native_body_extended import marker

        identity = marker()
        identity["mode"] = mode
        identity["selector_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return identity
    return _enabled_marker()


class PreparedNativeGGUFBody:
    """Prepared reference to existing packed bytes, with no recursive fallback."""

    def __init__(self, packed, qtype, k):
        if (
            not native_body_eligible(
                "ggml-q8_0-q6_k",
                torch.float32,
                qtype,
                (packed.shape[0], k),
                packed.is_cuda,
            )
            or packed.dtype != torch.uint8
        ):
            raise ValueError("Native body requires eligible CUDA Q8_0/Q6_K bytes")
        if not _is_sm86(packed.device):
            raise ValueError("Native body requires SM86")
        validate_layout(
            qtype,
            packed.shape,
            k,
            packed.data_ptr(),
            packed.untyped_storage().nbytes() - packed.storage_offset(),
        )
        self.packed = packed
        self.qtype, self.k, self.n = qtype, k, packed.shape[0]
        self.identity = _enabled_marker()
        self.library = self.identity["library"]["path"]
        lib = native_head._library(self.library)
        if lib.current_direct_prepare(packed.device.index):
            raise RuntimeError(lib.current_direct_error().decode())
        key = (packed.device.index, qtype, self.n, k, self.library)
        if key not in _WARMED:
            # Includes native context/attributes, Triton guard and unsafe repair
            # compilation for every selected row count, before graph capture.
            with torch.inference_mode():
                for rows in SELECTED_ROWS:
                    self(torch.ones(rows, k, dtype=torch.float32, device=packed.device))
            _WARMED.add(key)

    def __call__(self, x):
        if x.shape[-1] != self.k or x.device != self.packed.device:
            raise ValueError("Native body input width/device mismatch")
        return guarded_native(
            x.reshape(-1, self.k).contiguous(),
            self.packed,
            self.qtype,
            self.k,
            self.library,
        )


def prepare_native_body(packed, qtype, shape, dtype, values=None):
    mode = native_body_mode()
    if mode == "ggml-extended-b1":
        from .native_body_extended import prepare_extended_body

        return prepare_extended_body(packed, qtype, shape, dtype, values=values)
    if not native_body_eligible(mode, dtype, qtype, shape, packed.is_cuda):
        return None
    # Other architectures retain the original compact implementation.
    if not _is_sm86(packed.device):
        return None
    return PreparedNativeGGUFBody(packed, qtype, shape[1])
