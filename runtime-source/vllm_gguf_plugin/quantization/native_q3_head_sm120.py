# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM120 Q3_K vocabulary projection over unchanged GGUF bytes.

Safe F32 input rows use native Q8_1 activation quantization. Unsafe rows use
the original compact arithmetic. Embedding lookup and other head modes remain
unchanged. This backend deliberately changes the selected head arithmetic.
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

FORMATS = {11: (256, 110)}
SELECTED_ROWS = (1, 4)
MAX_ABS = 2047.0
MIN_NONZERO_BLOCK_AMAX = 2**-7


@lru_cache(maxsize=8)
def _is_sm120(device):
    return torch.cuda.get_device_capability(device) == (12, 0)


def validate_layout(qtype, shape, k, pointer, available_bytes):
    if qtype != 11 or tuple(shape) != (128000, 880) or k != 2048:
        raise ValueError("Candidate requires original Q3_K [128000,2048] head bytes")
    if pointer % 128 or available_bytes < 128000 * 880:
        raise ValueError(
            "Packed head needs GGML 128B alignment and original wire storage"
        )
    return 0


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


@torch.library.custom_op("vllm_gguf::native_sm120_q3_head", mutates_args=())
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
    if not _is_sm120(x.device):
        raise ValueError("Guarded native body requires SM120")
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


LIBRARY_ENV = "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_LIBRARY"
LIBRARY_SHA_ENV = "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_SHA256"
METADATA_SHA_ENV = "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_METADATA_SHA256"
SOURCE_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"


def _expected_digest(name):
    value = os.getenv(name, "")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"An explicit lowercase SHA256 is required in {name}")
    return value


def native_q3_head_library_identity():
    supplied = os.getenv(LIBRARY_ENV)
    if not supplied:
        raise ValueError("An explicit new SM120 Q3 host-admission library is required")
    path = Path(supplied).absolute()
    metadata = path.with_suffix(".build.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _expected_digest(LIBRARY_SHA_ENV)
    if digest != expected:
        raise ValueError("SM120 Q3 binary differs from the explicitly bound library")
    metadata_digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    if metadata_digest != _expected_digest(METADATA_SHA_ENV):
        raise ValueError("SM120 Q3 metadata differs from the explicitly bound metadata")
    build = json.loads(metadata.read_text())
    if (
        digest not in build.get("sha256", {}).values()
        or build.get("architecture", {}).get("compute_capability") != [12, 0]
        or build.get("source_commit") != SOURCE_COMMIT
        or build.get("device_arithmetic_source_unchanged") is not True
        or build.get("Q3_head_host_admission") not in (
            {"type": 11, "shape": [128000, 2048], "rows": [1, 4]},
            {"type": 11, "shape": [128000, 2048], "rows": "positive-int32-C-ABI-supported"},
        )
        or not build.get("host_patch_sha256")
    ):
        raise ValueError(
            "SM120 Q3 coherent source/architecture/admission proof differs"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "build_metadata_sha256": metadata_digest,
        "source_commit": build["source_commit"],
        "host_patch_sha256": build["host_patch_sha256"],
    }


@lru_cache(maxsize=128)
def direct_rows_supported(library, device, rows):
    # Keep the already-qualified small-row computation. Larger tensors may use
    # the backend's original MMQ selection after real workspace admission.
    if type(rows) is not int or rows <= 8 or rows > 2**31 - 1:
        return False
    metadata = json.loads(Path(library).with_suffix(".build.json").read_text())
    admission = metadata.get("Q3_head_host_admission")
    if admission != {"type": 11, "shape": [128000, 2048], "rows": "positive-int32-C-ABI-supported"}:
        return False
    lib = native_head._library(library)
    return bool(lib.current_direct_workspace(device, 11, 128000, 2048, rows))


def native_q3_head_cache_marker():
    if native_head.native_head_validate_b1():
        raise ValueError("native-q3 does not provide HEAD_VALIDATE_B1 row batching")
    return _cache_marker()


@lru_cache(maxsize=1)
def _cache_marker():
    identity = native_q3_head_library_identity()
    metadata = Path(identity["path"]).with_suffix(".build.json")
    paths = [
        Path(__file__),
        Path(native_head.__file__),
        Path(__file__).with_name("compact.py"),
        Path(__file__).with_name("compact_decode.py"),
        Path(__file__).with_name("vocal_embeds.py"),
    ]
    return {
        "mode": "native-q3",
        "validate_b1": False,
        "contract": "original-wire-sm120-guarded-q3-head-v1",
        "library": identity,
        "build_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "compiler_command": json.loads(metadata.read_text()).get("compile"),
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "SM": 120,
        "activation_dtype": "float32",
        "qtype": 11,
        "head_shape": [128000, 2048],
        "activation_rows": list(SELECTED_ROWS),
        "guard": {
            "max_abs": MAX_ABS,
            "min_nonzero_32block_amax": MIN_NONZERO_BLOCK_AMAX,
        },
        "persistent_additional_weight_bytes": 0,
        "activation_quantization_changes_arithmetic": True,
    }


class PreparedNativeQ3Head:
    """Original packed pointer; all selected rows warm before graph capture."""

    def __init__(self, packed, qtype, k):
        if not packed.is_cuda or packed.dtype != torch.uint8 or torch.version.hip:
            raise ValueError("Native Q3 head requires original NVIDIA CUDA wire bytes")
        if not _is_sm120(packed.device):
            raise ValueError("Native Q3 head requires SM120")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare native Q3 head before CUDA graph capture")
        validate_layout(
            qtype,
            packed.shape,
            k,
            packed.data_ptr(),
            packed.untyped_storage().nbytes() - packed.storage_offset(),
        )
        self.packed, self.qtype, self.k, self.n = packed, qtype, k, 128000
        self.mode = "native-q3"
        self.identity = native_q3_head_cache_marker()
        self.library = self.identity["library"]["path"]
        lib = native_head._library(self.library)
        with torch.cuda.device(packed.device), torch.inference_mode():
            if lib.current_direct_prepare(packed.device.index):
                raise RuntimeError(lib.current_direct_error().decode())
            for rows in SELECTED_ROWS:
                self(torch.ones(rows, k, dtype=torch.float32, device=packed.device))
            torch.cuda.synchronize(packed.device)

    def __call__(self, x):
        if x.shape[-1] != self.k or x.device != self.packed.device:
            raise ValueError("Native Q3 head input width/device mismatch")
        return guarded_native(
            x.reshape(-1, self.k).contiguous(),
            self.packed,
            self.qtype,
            self.k,
            self.library,
        )


def prepare_native_q3_head(packed, qtype, shape, dtype):
    if (
        dtype != torch.float32
        or qtype != 11
        or tuple(shape) != (128000, 2048)
        or not packed.is_cuda
        or torch.version.hip is not None
    ):
        return None
    if not _is_sm120(packed.device):
        return None
    return PreparedNativeQ3Head(packed, qtype, shape[1])
