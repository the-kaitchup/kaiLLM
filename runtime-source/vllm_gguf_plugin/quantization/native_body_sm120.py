# SPDX-License-Identifier: Apache-2.0
"""Isolated SM120 extended ordinary-GGUF body; original packed bytes remain resident.

Native Q8_1 activation quantization changes arithmetic. This is opt-in research,
pending actual component/graph and whole-model EAR tests. It installs no hooks.
"""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from vllm_gguf_plugin.quantization import compact, native_head
from vllm_gguf_plugin.quantization.compact_decode import load_gguf

from .native_geometry import geometry_supported as _geometry_supported
from .native_geometry import warm_rows

FORMATS = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 23: (256, 136)}
SELECTED_ROWS = warm_rows()  # Warm families only; never tensor-row admission.
KERNEL_ROWS = (1, 2, 4, 8, 512, 1024, 2048)  # Historical metadata only.


def geometry_supported(qtype, n, k, m=1):
    return _geometry_supported(qtype, n, k, m, FORMATS)


@lru_cache(maxsize=8192)
def _native_rows_supported(library, device_index, qtype, n, k, rows):
    if not geometry_supported(qtype, n, k, rows):
        return False
    lib = native_head._library(library)
    return lib.current_direct_workspace(device_index, qtype, n, k, rows) > 0


MAX_ABS = 2047.0
MIN_NONZERO_BLOCK_AMAX = 2**-7
F32_SCALE_MINIMUM_AMAX = 2**-80


@lru_cache(maxsize=8)
def _is_sm120(device):
    return torch.cuda.get_device_capability(device) == (12, 0)


def minimum_for_layout(layout):
    if layout not in (0, 1, 2):
        raise ValueError("Unknown native RHS layout")
    return F32_SCALE_MINIMUM_AMAX if layout == 1 else MIN_NONZERO_BLOCK_AMAX


@lru_cache(maxsize=128)
def rhs_layout(library, device_index, qtype, rows):
    """C ABI metadata only; constructors warm/cache every selected combination."""
    lib = native_head._library(library)
    query = lib.current_shared_layout
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    query.restype = ctypes.c_int
    result = query(device_index, qtype, rows)
    minimum_for_layout(result)
    if result == 1 and sha(library) != DEFAULT_LIBRARY_SHA256:
        raise ValueError(
            "Float-scale guard requires the source-verified R3 SM120 binary"
        )
    return result


def validate_layout(qtype, shape, k, pointer, available_bytes):
    if qtype not in FORMATS or len(shape) != 2:
        raise ValueError("Extended body requires supported original GGUF wire bytes")
    block, size = FORMATS[qtype]
    n, row_bytes = shape
    if k <= 0 or k % block or row_bytes != k // block * size or n <= 0:
        raise ValueError("Packed GGUF shape does not match K/type")
    if not geometry_supported(qtype, n, k):
        raise ValueError("Geometry exceeds native C ABI integer/block capabilities")
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


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _repair_gemv(
    x_ptr,
    w_ptr,
    values_ptr,
    y_ptr,
    unsafe_ptr,
    M,
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


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _repair_gemm(
    x_ptr,
    w_ptr,
    values_ptr,
    y_ptr,
    unsafe_ptr,
    M,
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


def sanitize(x, minimum=MIN_NONZERO_BLOCK_AMAX):
    clean = torch.empty_like(x)
    unsafe = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    _sanitize_rows[(x.shape[0],)](
        x,
        clean,
        unsafe,
        K=x.shape[1],
        BLOCKS=triton.next_power_of_2(triton.cdiv(x.shape[1], 32)),
        LIMIT=MAX_ABS,
        MIN_AMAX=minimum,
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


@torch.library.custom_op("vllm_gguf::native_sm120_body", mutates_args=())
def guarded_native(
    x: torch.Tensor,
    packed: torch.Tensor,
    values: torch.Tensor,
    qtype: int,
    k: int,
    library: str,
) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not geometry_supported(qtype, packed.shape[0], k, x.shape[0])
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
        or values.device != packed.device
        or (qtype == 23 and (values.dtype != torch.int8 or values.numel() != 16))
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
    layout = rhs_layout(library, x.device.index, qtype, x.shape[0])
    clean, unsafe = sanitize(x, minimum=minimum_for_layout(layout))
    output = native_raw(clean, packed, qtype, k, library)
    m, n = output.shape
    if m <= 8:
        _repair_gemv[(triton.cdiv(n, 4),)](
            x,
            packed,
            values,
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
            values,
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
def _fake(x, packed, values, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


MODE_ENV = "VLLM_GGUF_NATIVE_SM120_B1"
BODY_LIBRARY_ENV = "VLLM_GGUF_NATIVE_SM120_LIBRARY"
DEFAULT_LIBRARY_SHA256 = (
    "fc120c9bb36f262f054956ba7ff7a81242cd0e2a20801699e4b9376fe8a0980c"
)
SOURCE_MANIFEST_SHA256 = (
    "8796bbd00c4cfd3419bbef1186f13d31ef898cd907972222eea9c2dd6d71bcdb"
)
SOURCE_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"
_WARMED = set()


def mode():
    value = os.getenv(MODE_ENV, "none")
    if value not in ("none", "body", "body-q6"):
        raise ValueError("Explicit SM120 candidate mode must be none, body or body-q6")
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_build(build, digest):
    if (
        digest != DEFAULT_LIBRARY_SHA256
        or digest not in build.get("sha256", {}).values()
    ):
        raise ValueError(
            "SM120 candidate requires the exact coherent R3 direct library"
        )
    if build.get("architecture", {}).get("compute_capability") != [12, 0]:
        raise ValueError("Explicit SM120 build architecture is required")
    if (
        build.get("source_commit") != SOURCE_COMMIT
        or build.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "SM120 candidate requires the pinned coherent GGML source closure"
        )
    command = build.get("compile", [])
    if not any(
        "compute_120" in argument and "sm_120" in argument for argument in command
    ):
        raise ValueError("SM120 target is absent from the recorded compile command")
    expected = "97b09694f8757440a897695dad0df083d32e49c1ea72f1b23585511c43bed1b8"
    if (
        build.get("source_closure_sha256", {}).get("kernels/current_ggml_direct.cu")
        != expected
    ):
        raise ValueError("Current direct/shared-layout C ABI source differs from R3")


def library_identity():
    supplied = os.getenv(BODY_LIBRARY_ENV) or os.getenv("VLLM_GGUF_NATIVE_HEAD_LIBRARY")
    if not supplied:
        raise ValueError("An explicit SM120 candidate library path is required")
    path = Path(supplied)
    digest = sha(path)
    metadata = path.with_suffix(".build.json")
    build = json.loads(metadata.read_text())
    validate_build(build, digest)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "build_metadata_sha256": sha(metadata),
        "compiler_command": build["compile"],
        "architecture": build["architecture"],
        "source_manifest_sha256": build["source_manifest_sha256"],
    }


def marker():
    library = library_identity()
    files = [
        Path(__file__),
        Path(__file__).with_name("native_geometry.py"),
        Path(compact.__file__),
        Path(compact.__file__).with_name("compact_decode.py"),
        Path(native_head.__file__),
    ]
    return {
        "contract": "extended-ordinary-body-sm120-abi-layout-minimum-candidate-v1",
        "mode": mode(),
        "library": library,
        "body_library_environment": BODY_LIBRARY_ENV,
        "sources": {str(p.resolve()): sha(p) for p in files},
        "SM": 120,
        "qtypes": sorted(FORMATS),
        "activation_rows": "generic-positive-int32-geometry",
        "constructor_warm_rows": list(warm_rows()),
        "kernel_activation_rows": "generic-positive-int32-geometry",
        "public_F32_automatic_rows": "real C ABI capability; no row table",
        "request_batch_scope": "independent of logical request batch",
        "guard": {
            "max_abs": MAX_ABS,
            "min_nonzero_32block_amax": MIN_NONZERO_BLOCK_AMAX,
            "relaxed_for_D4": True,
            "D4_F32_scale_minimum_amax": F32_SCALE_MINIMUM_AMAX,
            "selection": "pinned current_shared_layout ABI; layout1 only",
            "required_D4_binary_sha256": DEFAULT_LIBRARY_SHA256,
        },
        "MMQ_layout": {
            "Q4_K": "DS4-half",
            "Q5_K": "DS4-half",
            "Q8_0": "D4-F32",
            "Q6_K": "D4-F32",
            "IQ4_XS": "D4-F32",
        },
        "persistent_additional_weight_bytes": 0,
        "IQ4_XS_repair_values": "unchanged original shared int8 nonlinear LUT",
        "selected": False,
        "generic_geometry_source_stage": True,
        "repair_M_runtime": True,
        "partial_geometry_GPU_component_sha256": (
            "68e6faa6f368646f823abc7e6dfea5e9038c3ac3f7c7e723dc49effab5c614db"
        ),
        "actual_geometry_capability": (
            "positive int32 M/N/K; qtype block-aligned K; C ABI layout query"
        ),
        "whole_model_EAR_required": True,
    }


def eligible(qtype, shape, dtype, is_cuda):
    return (
        dtype == torch.float32
        and is_cuda
        and qtype in FORMATS
        and len(shape) == 2
        and geometry_supported(qtype, int(shape[0]), int(shape[1]))
    )


class PreparedNativeGGUFBody:
    """Zero-copy body object; compile and warm all selected rows before capture."""

    SELECTED_ROWS = SELECTED_ROWS

    def __init__(self, packed, qtype, k, values=None):
        if mode() == "none":
            raise ValueError("SM120 candidate is default off; explicit opt-in required")
        if not eligible(qtype, (packed.shape[0], k), torch.float32, packed.is_cuda):
            raise ValueError("Extended body requires eligible original CUDA bytes")
        if packed.dtype != torch.uint8 or not packed.is_contiguous():
            raise ValueError("Extended body requires contiguous uint8 wire storage")
        if not _is_sm120(packed.device):
            raise ValueError("Extended body currently requires SM120")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the extended body before graph capture")
        validate_layout(
            qtype,
            packed.shape,
            k,
            packed.data_ptr(),
            packed.untyped_storage().nbytes() - packed.storage_offset(),
        )
        self.packed, self.qtype, self.k, self.n = packed, qtype, k, packed.shape[0]
        self.values = (
            values
            if values is not None
            else compact.get_iq_table_tensors(packed.device)["kvalues_iq4nl"]
            if qtype == 23
            else packed.reshape(-1)[:1]
        )
        self.identity = marker()
        self.library = self.identity["library"]["path"]
        lib = native_head._library(self.library)
        if lib.current_direct_prepare(packed.device.index):
            raise RuntimeError(lib.current_direct_error().decode())
        key = (packed.device.index, qtype, self.n, k, self.library)
        if key not in _WARMED:
            with torch.inference_mode():
                for rows in warm_rows():
                    if self.supports_rows(rows):
                        self(
                            torch.ones(
                                rows, k, dtype=torch.float32, device=packed.device
                            )
                        )
            torch.cuda.synchronize(packed.device)
            _WARMED.add(key)

    def supports_rows(self, rows):
        return _native_rows_supported(
            self.library, self.packed.device.index, self.qtype, self.n, self.k, rows
        )

    def __call__(self, x):
        if x.shape[-1] != self.k or x.device != self.packed.device:
            raise ValueError("Extended body activation width/device mismatch")
        return guarded_native(
            x.reshape(-1, self.k).contiguous(),
            self.packed,
            self.values,
            self.qtype,
            self.k,
            self.library,
        )


class PreparedExtendedLinear:
    """Component-only wrapper: unsupported rows use the original layer directly."""

    def __init__(self, original):
        if not eligible(
            original.weight_type,
            original.shape,
            original.dtype,
            original.weight.is_cuda,
        ):
            raise ValueError("Original compact layer is outside the extended scope")
        self.original = original
        self.prepared = PreparedNativeGGUFBody(
            original.weight, original.weight_type, original.shape[1], original.values
        )
        self.packed = self.prepared.packed
        self.library = self.prepared.library

    def supports_rows(self, rows):
        return self.prepared.supports_rows(rows)

    def __call__(self, x, bias=None, *, output_dtype=None):
        if (
            x.dtype != torch.float32
            or not self.supports_rows(x.numel() // x.shape[-1])
            or output_dtype not in (None, torch.float32)
        ):
            return self.original(x, bias, output_dtype=output_dtype)
        output = self.prepared(x).reshape(*x.shape[:-1], self.original.shape[0])
        return output if bias is None else output + bias


def prepare_extended_body(packed, qtype, shape, dtype, values=None):
    if mode() == "none":
        return None
    if not eligible(qtype, shape, dtype, packed.is_cuda) or not _is_sm120(
        packed.device
    ):
        return None
    return PreparedNativeGGUFBody(packed, qtype, shape[1], values=values)
