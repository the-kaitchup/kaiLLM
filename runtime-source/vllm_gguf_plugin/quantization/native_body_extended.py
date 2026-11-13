# SPDX-License-Identifier: Apache-2.0
"""Isolated SM86 extended ordinary-GGUF body; original packed bytes remain resident.

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

# GGML wire formats with native MMVQ/MMQ kernels (legacy 32-element formats included).
FORMATS = {2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
           8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
           18: (256, 98), 21: (256, 110), 22: (256, 82), 23: (256, 136)}
SELECTED_ROWS = warm_rows()  # Constructor warmup only, never eligibility.
MAX_ABS = 2047.0
MIN_NONZERO_BLOCK_AMAX = 2**-7
F32_SCALE_MINIMUM_AMAX = 2**-80


@lru_cache(maxsize=8)
def _is_sm86(device):
    return torch.cuda.get_device_capability(device) == (8, 6)


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
        raise ValueError("Float-scale guard requires the source-verified R6 binary")
    return result


def geometry_supported(qtype, n, k, rows=1):
    return _geometry_supported(qtype, n, k, rows, FORMATS)


@lru_cache(maxsize=8192)
def _native_rows_supported(library, device_index, qtype, n, k, rows):
    if not geometry_supported(qtype, n, k, rows):
        return False
    # The identity/source-qualified backend is prepared before this host-only
    # query. It chooses the real typed MMVQ/MMQ capability; no D2H token read.
    lib = native_head._library(library)
    return lib.current_direct_workspace(device_index, qtype, n, k, rows) > 0


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
    grid_ptr,
    sign_ptr,
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
                grid_ptr,
                sign_ptr,
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
    grid_ptr,
    sign_ptr,
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
                grid_ptr,
                sign_ptr,
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


@torch.library.custom_op("gguf_body_d4_guard_probe::guarded_native", mutates_args=())
def guarded_native(
    x: torch.Tensor,
    packed: torch.Tensor,
    values: torch.Tensor,
    qtype: int,
    k: int,
    library: str,
    grid: torch.Tensor | None = None,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    if qtype in (18, 21, 22) and (grid is None or signs is None):
        raise ValueError("Prepare IQ grid/sign tables before guarded native execution")
    grid = packed.reshape(-1)[:1] if grid is None else grid
    signs = packed.reshape(-1)[:1] if signs is None else signs
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
        or grid.device != packed.device or signs.device != packed.device
        or grid.dtype != torch.uint8 or signs.dtype != torch.uint8
        or not grid.is_contiguous() or not signs.is_contiguous()
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
    if not _is_sm86(x.device):
        raise ValueError("Guarded native body requires SM86")
    layout = rhs_layout(library, x.device.index, qtype, x.shape[0])
    clean, unsafe = sanitize(x, minimum=minimum_for_layout(layout))
    output = native_raw(clean, packed, qtype, k, library)
    m, n = output.shape
    if m <= 8:
        _repair_gemv[(triton.cdiv(n, 4),)](
            x,
            packed,
            values,
            grid,
            signs,
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
            grid,
            signs,
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
def _fake(x, packed, values, qtype, k, library, grid=None, signs=None):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


BODY_LIBRARY_ENV = "VLLM_GGUF_NATIVE_BODY_LIBRARY"
DEFAULT_LIBRARY_SHA256 = (
    "8fb97dc6bf54cd0d6a161250e87678a9a809dc40a9b4b6a49be723b0091b3801"
)
DEFAULT_LIBRARY = (Path(__file__).resolve().parents[1] / "native/libraries/sm86/libcurrent_ggml_direct.so")
_WARMED = set()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_identity():
    # This selector is deliberately independent from every vocabulary-head
    # library selector. Extending body types must not change head arithmetic.
    supplied = os.getenv(BODY_LIBRARY_ENV)
    packaged = (
        Path(native_head.__file__).parents[1] / "native/lib/libcurrent_ggml_direct.so"
    )
    path = (
        Path(supplied)
        if supplied
        else packaged
        if packaged.is_file()
        else DEFAULT_LIBRARY
    )
    digest = sha(path)
    metadata = path.with_suffix(".build.json")
    build = json.loads(metadata.read_text())
    if digest not in build.get("sha256", {}).values():
        raise ValueError("Body library is absent from its build provenance")
    if not supplied and digest != DEFAULT_LIBRARY_SHA256:
        raise ValueError(
            "Default extended body requires the unchanged pinned R6 binary"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "build_metadata_sha256": sha(metadata),
        "compiler_command": build.get("compile"),
    }


def marker():
    library = library_identity()
    files = [
        Path(__file__),
        Path(__file__).with_name("native_geometry.py"),
        Path(__file__).with_name("native_wire.py"),
        Path(compact.__file__),
        Path(compact.__file__).with_name("compact_decode.py"),
        Path(native_head.__file__),
    ]
    return {
        "contract": "extended-ordinary-body-sm86-abi-layout-minimum-candidate-v1",
        "library": library,
        "body_library_environment": BODY_LIBRARY_ENV,
        "sources": {str(p.resolve()): sha(p) for p in files},
        "SM": 86,
        "qtypes": sorted(FORMATS),
        "constructor_warm_rows": list(warm_rows()),
        "activation_rows": "positive int32 typed rows accepted by actual C ABI",
        "request_batch_scope": "B1 decode/speculative queries and B1 prompt tokens",
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


def _repair_tables(packed, qtype, grid=None, signs=None):
    # Reuse the same cached codebooks as the original compact projection.
    names = {18: "iq3xxs_grid", 21: "iq3s_grid", 22: "iq2s_grid"}
    dummy = packed.reshape(-1)[:1]
    if qtype not in names:
        return (dummy if grid is None else grid,
                dummy if signs is None else signs)
    tables = compact.get_iq_table_tensors(packed.device)
    expected_grid, expected_signs = tables[names[qtype]], tables["ksigns_iq2xs"]
    if grid is not None and grid is not expected_grid:
        raise ValueError("IQ body repair requires the original compact grid")
    if signs is not None and signs is not expected_signs:
        raise ValueError("IQ body repair requires the original compact sign table")
    return expected_grid, expected_signs


class PreparedNativeGGUFBody:
    """Zero-copy body object; compile and warm all selected rows before capture."""

    SELECTED_ROWS = SELECTED_ROWS

    def __init__(self, packed, qtype, k, values=None, *, grid=None, signs=None):
        if not eligible(qtype, (packed.shape[0], k), torch.float32, packed.is_cuda):
            raise ValueError("Extended body requires eligible original CUDA bytes")
        if packed.dtype != torch.uint8 or not packed.is_contiguous():
            raise ValueError("Extended body requires contiguous uint8 wire storage")
        if not _is_sm86(packed.device):
            raise ValueError("Extended body currently requires SM86")
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
        self.grid, self.signs = _repair_tables(packed, qtype, grid, signs)
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
            grid=self.grid,
            signs=self.signs,
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
            original.weight, original.weight_type, original.shape[1], original.values,
            grid=original.grid, signs=original.signs
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


def prepare_extended_body(packed, qtype, shape, dtype, values=None, *, grid=None, signs=None):
    if not eligible(qtype, shape, dtype, packed.is_cuda) or not _is_sm86(packed.device):
        return None
    return PreparedNativeGGUFBody(packed, qtype, shape[1], values=values,
                                  grid=grid, signs=signs)
