# SPDX-License-Identifier: Apache-2.0
"""Opt-in, pinned GGML Q5_K vocabulary head over the original GGUF bytes."""

import ctypes
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger("vllm.gguf.native_head")
NATIVE_HEAD_COMPUTE_CAPABILITIES = ((8, 6), (12, 0))
_DEFAULT_LIBRARY = (Path(__file__).resolve().parents[1] / "native/libraries/sm86/libcurrent_ggml_direct.so")
_DEFAULT_SHA256 = "8fb97dc6bf54cd0d6a161250e87678a9a809dc40a9b4b6a49be723b0091b3801"
_DEFAULT_MMA_LIBRARY = (Path(__file__).resolve().parents[1] / "native/libraries/sm86/libcurrent_ggml_affine_mma.so")
_DEFAULT_MMA_SHA256 = "e7c6f5b20ad87809d1d081e7d2699ce48053476e97410e37e20eb559ad48419b"


_SM120_SOURCE_MANIFEST = (
    "8796bbd00c4cfd3419bbef1186f13d31ef898cd907972222eea9c2dd6d71bcdb"
)


_SM120_AFFINE_HOST_MANIFEST = (
    "88a0eedec2faa9ac64c04b0c8f5c1c982463e8a8963d809b3d866989debb2435"
)
_SM120_AFFINE_HOST_SOURCE = (
    "aabe151a62bde3fbeec548cfdb969930a5e9c3e1c22fee9d53ac16cada5466be"
)


def _validate_head_architecture(identity, capability):
    """Keep pinned SM86 behavior; SM120 requires the coherent R6 build proof."""
    actual = tuple(capability)
    if actual not in ((8, 6), (12, 0)):
        raise ValueError("Native GGUF head supports only SM86 or experimental SM120")
    path = Path(identity["path"])
    metadata = path.with_suffix(".build.json")
    build = json.loads(metadata.read_text()) if metadata.is_file() else {}
    declared = build.get("architecture", {}).get("compute_capability")
    if declared is not None and tuple(declared) != actual:
        raise ValueError("Native head library architecture differs from tensor GPU")
    if actual == (8, 6):
        return
    if declared != [12, 0]:
        raise ValueError("SM120 native head requires explicit SM120 build provenance")
    if build.get("source_commit") != "0c1e57098bba43ac29e6e3b677cdceebdd22334f":
        raise ValueError(
            "SM120 native head requires the coherent pinned R6 GGML source"
        )
    manifest = build.get("source_manifest_sha256")
    if manifest not in (_SM120_SOURCE_MANIFEST, _SM120_AFFINE_HOST_MANIFEST):
        raise ValueError(
            "SM120 native head source closure differs from the reviewed R6 build"
        )
    if manifest == _SM120_AFFINE_HOST_MANIFEST and (
        build.get("host_affine_sm120_override") is not True
        or build.get("parent_source_manifest_sha256") != _SM120_SOURCE_MANIFEST
        or build.get("source_closure_sha256", {}).get(
            "kernels/current_ggml_body_mma.cu"
        )
        != _SM120_AFFINE_HOST_SOURCE
    ):
        raise ValueError(
            "SM120 affine host-policy override lacks its exact source proof"
        )
    if identity["sha256"] not in build.get("sha256", {}).values():
        raise ValueError(
            "SM120 native head binary hash is absent from build provenance"
        )
    command = build.get("compile", build.get("command", []))
    if not any("compute_120" in arg and "sm_120" in arg for arg in command):
        raise ValueError("SM120 native head compile command lacks the required target")


def library_compute_capability(identity):
    """Resolve the selected library target from provenance, without a GPU query."""
    if identity.get("sha256") == _DEFAULT_SHA256:
        return (8, 6)
    architecture = identity.get("architecture", {})
    declared = (
        architecture.get("compute_capability")
        if isinstance(architecture, dict)
        else None
    )
    if declared is None:
        metadata = Path(identity["path"]).with_suffix(".build.json")
        if metadata.is_file():
            declared = (
                json.loads(metadata.read_text())
                .get("architecture", {})
                .get("compute_capability")
            )
    if declared is None or tuple(declared) not in NATIVE_HEAD_COMPUTE_CAPABILITIES:
        raise ValueError("Selected native library has no supported target provenance")
    return tuple(declared)


def native_head_mode():
    mode = os.getenv("VLLM_GGUF_LFM3_HEAD", "compact")
    if mode not in ("compact", "native", "native-mma", "native-q6", "auto-native"):
        raise ValueError(
            "LFM3_HEAD must be compact, native, native-mma, native-q6, or auto-native"
        )
    return mode


def native_head_validate_b1():
    value = os.getenv("VLLM_GGUF_LFM3_HEAD_VALIDATE_B1", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_GGUF_LFM3_HEAD_VALIDATE_B1 must be 0 or 1")
    return value == "1"


def native_head_library_identity():
    supplied = os.getenv("VLLM_GGUF_NATIVE_HEAD_LIBRARY")
    packaged = Path(__file__).parents[1] / "native/lib/libcurrent_ggml_direct.so"
    path = (
        Path(supplied)
        if supplied
        else packaged
        if packaged.is_file()
        else _DEFAULT_LIBRARY
    )
    if not path.is_file():
        raise FileNotFoundError(f"Native GGUF head library is unavailable: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = path.with_suffix(".build.json")
    if not supplied and digest != _DEFAULT_SHA256:
        raise ValueError("Default native GGUF head library does not match pinned R6")
    if supplied:
        if not metadata.is_file():
            raise ValueError(
                "An explicit native head library requires .build.json provenance"
            )
        build = json.loads(metadata.read_text())
        if digest not in build.get("sha256", {}).values():
            raise ValueError(
                "Native head library hash is absent from its build metadata"
            )
    result = {"path": str(path.resolve()), "sha256": digest}
    if supplied:
        result["build_metadata_sha256"] = hashlib.sha256(
            metadata.read_bytes()
        ).hexdigest()
        if "architecture" in build:
            result["architecture"] = build["architecture"]
            result["source_manifest_sha256"] = build.get("source_manifest_sha256")
    return result


def native_head_cache_marker():
    mode = native_head_mode()
    if mode == "auto-native":
        from .auto_head import runtime_marker

        return runtime_marker()
    if mode == "native-q6":
        if library_compute_capability(native_head_library_identity()) == (12, 0):
            from .native_q6_head_sm120 import native_q6_head_cache_marker
        else:
            from .native_q6_head import native_q6_head_cache_marker
        return native_q6_head_cache_marker()
    marker = {
        "mode": mode,
        "validate_b1": native_head_validate_b1(),
        "library": native_head_library_identity() if mode != "compact" else None,
    }
    if mode == "native-mma":
        marker["mma_library"] = native_head_mma_library_identity()
        marker["large_rows_helper_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("native_head_large.py").read_bytes()
        ).hexdigest()
        marker["helper_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    return marker


def native_head_mma_library_identity():
    supplied = os.getenv("VLLM_GGUF_NATIVE_HEAD_MMA_LIBRARY")
    packaged = Path(__file__).parents[1] / "native/lib/libcurrent_ggml_affine_mma.so"
    path = (
        Path(supplied)
        if supplied
        else (packaged if packaged.is_file() else _DEFAULT_MMA_LIBRARY)
    )
    if not path.is_file():
        raise FileNotFoundError(f"Native MMA head library is unavailable: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = path.with_suffix(".build.json")
    if not metadata.is_file():
        raise ValueError("Native MMA head requires .build.json provenance")
    build = json.loads(metadata.read_text())
    if build.get("affine_mma") is not True:
        raise ValueError("Native MMA head requires affine_mma build provenance")
    if digest != _DEFAULT_MMA_SHA256 and (
        not supplied or digest not in build.get("sha256", {}).values()
    ):
        raise ValueError("Native MMA head library hash does not match pinned build")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "build_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }


@lru_cache(maxsize=4)
def _library(path):
    lib = ctypes.CDLL(path)
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.current_direct_prepare.argtypes = [ctypes.c_int]
    lib.current_direct_prepare.restype = ctypes.c_int
    shape = [ctypes.c_int, ctypes.c_int] + [ctypes.c_int64] * 3
    lib.current_direct_workspace.argtypes = shape
    lib.current_direct_workspace.restype = ctypes.c_size_t
    lib.current_direct_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + shape + [ctypes.c_size_t]
    )
    lib.current_direct_run.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=4)
def _channels_library(path):
    lib = _library(path)
    shape = [ctypes.c_int, ctypes.c_int] + [ctypes.c_int64] * 4
    lib.current_channels_workspace.argtypes = shape
    lib.current_channels_workspace.restype = ctypes.c_size_t
    lib.current_channels_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + shape + [ctypes.c_size_t]
    )
    lib.current_channels_run.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=4)
def _mma_library(path):
    lib = ctypes.CDLL(path)
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.current_direct_prepare.argtypes = [ctypes.c_int]
    lib.current_direct_prepare.restype = ctypes.c_int
    lib.current_body_workspace.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.current_body_workspace.restype = ctypes.c_size_t
    lib.current_body_run.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_size_t,
    ]
    lib.current_body_run.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_yaqa::native_mma_head", mutates_args=())
def _native_mma_head_op(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int, library: str
) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.shape != (8, 2048)
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or packed.shape != (128000, 1408)
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
        or packed.data_ptr() % 128
        or qtype != 13
        or k != 2048
    ):
        raise ValueError("Native MMA head requires CUDA F32 M8 and original LFM3 Q5_K")
    output = torch.empty((8, 128000), device=x.device, dtype=torch.float32)
    lib = _mma_library(library)
    types = (ctypes.c_int * 1)(qtype)
    rows = (ctypes.c_int64 * 1)(128000)
    pointers = (ctypes.c_void_p * 1)(packed.data_ptr())
    with torch.cuda.device(x.device):
        size = lib.current_body_workspace(x.device.index, types, rows, 1, k, 8)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        if lib.current_body_run(
            pointers,
            types,
            rows,
            1,
            x.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            size,
            x.device.index,
            k,
            8,
            torch.cuda.current_stream(x.device).cuda_stream,
        ):
            raise RuntimeError(lib.current_direct_error().decode())
    return output


@_native_mma_head_op.register_fake
def _fake_native_mma_head(x, packed, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


@torch.library.custom_op("gguf_yaqa::native_head", mutates_args=())
def _native_head_op(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int, library: str
) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not 1 <= x.shape[0] <= 8
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.device != x.device
        or qtype != 13
        or k % 256
        or packed.shape[1] != k // 256 * 176
    ):
        raise ValueError(
            "Native head requires contiguous CUDA F32 input and original Q5_K bytes"
        )
    tail = (((-k) % 512) + ((-packed.shape[0]) % 4) * k) // 256 * 176
    if (
        packed.data_ptr() % 128
        or packed.untyped_storage().nbytes() - packed.storage_offset()
        < packed.numel() + tail
    ):
        raise ValueError("Native head packed storage lacks GGML alignment/tail padding")
    lib = _library(library)
    output = torch.empty(
        (x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32
    )
    with torch.cuda.device(x.device):
        args = x.device.index, qtype, packed.shape[0], k, x.shape[0]
        size = lib.current_direct_workspace(*args)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, device=x.device, dtype=torch.uint8)
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


@_native_head_op.register_fake
def _fake_native_head(x, packed, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


@torch.library.custom_op("gguf_yaqa::native_head_b1_rows", mutates_args=())
def _native_head_b1_rows_op(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int, library: str
) -> torch.Tensor:
    """Validation-only independent [channels, one token, K] head evaluation."""
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not 1 <= x.shape[0] <= 8
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.device != x.device
        or qtype != 13
        or k % 256
        or packed.shape[1] != k // 256 * 176
    ):
        raise ValueError(
            "Native B1 head validation requires CUDA F32 rows and original Q5_K bytes"
        )
    tail = (((-k) % 512) + ((-packed.shape[0]) % 4) * k) // 256 * 176
    if (
        packed.data_ptr() % 128
        or packed.untyped_storage().nbytes() - packed.storage_offset()
        < packed.numel() + tail
    ):
        raise ValueError("Native head packed storage lacks GGML alignment/tail padding")
    if x.shape[0] == 1:
        return _native_head_op(x, packed, qtype, k, library)
    lib = _channels_library(library)
    output = torch.empty(
        (x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32
    )
    with torch.cuda.device(x.device):
        # Weights broadcast over channels; dispatch/reductions see T=1.
        args = x.device.index, qtype, packed.shape[0], k, 1, x.shape[0]
        size = lib.current_channels_workspace(*args)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, device=x.device, dtype=torch.uint8)
        status = lib.current_channels_run(
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


@_native_head_b1_rows_op.register_fake
def _fake_native_head_b1_rows(x, packed, qtype, k, library):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


class PreparedNativeGGUFHead:
    """Capture one library identity and warm all supported decode batch sizes.

    The parent compact module owns the sole packed-weight buffer. This helper
    retains only scalar metadata and never allocates a second weight matrix.
    """

    def __init__(self, packed, qtype, k, activation_dtype, mode="native"):
        if not packed.is_cuda or torch.version.hip is not None:
            raise ValueError("Native GGUF head requires NVIDIA CUDA")
        capability = tuple(torch.cuda.get_device_capability(packed.device))
        if capability not in ((8, 6), (12, 0)):
            raise ValueError("Native head requires NVIDIA SM86 or experimental SM120")
        if qtype != 13 or k % 512 or packed.shape[0] % 4:
            raise ValueError(
                "Native head requires Q5_K with K divisible by512 and rows by4"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare native GGUF head before CUDA graph capture")
        self.identity = native_head_library_identity()
        _validate_head_architecture(self.identity, capability)
        self.validate_b1 = native_head_validate_b1()
        if mode not in ("native", "native-mma"):
            raise ValueError("Prepared native head mode must be native or native-mma")
        self.mode = mode
        self.mma_identity = (
            native_head_mma_library_identity()
            if mode == "native-mma"
            and not self.validate_b1
            and k == 2048
            and packed.shape[0] == 128000
            else None
        )
        if self.mma_identity is not None:
            _validate_head_architecture(self.mma_identity, capability)
        self.qtype, self.k = int(qtype), int(k)
        lib = _library(self.identity["path"])
        with torch.cuda.device(packed.device):
            if lib.current_direct_prepare(packed.device.index):
                raise RuntimeError(lib.current_direct_error().decode())
            if self.mma_identity is not None:
                mma = _mma_library(self.mma_identity["path"])
                if mma.current_direct_prepare(packed.device.index):
                    raise RuntimeError(mma.current_direct_error().decode())
            for batch in range(1, 9):
                x = torch.zeros(
                    (batch, k), device=packed.device, dtype=activation_dtype
                )
                self(x, packed)
            torch.cuda.synchronize(packed.device)
        logger.info(
            "Native Q5_K head enabled at batches1..8; F32/Q8_1 input, "
            "F32 output; validate_b1=%s library=%s sha256=%s mode=%s mma=%s",
            self.validate_b1,
            self.identity["path"],
            self.identity["sha256"],
            self.mode,
            self.mma_identity,
        )

    def __call__(self, x, packed):
        if self.mma_identity is not None and 2 <= x.shape[0] <= 8:
            source = torch.nn.functional.pad(x.float(), (0, 0, 0, 8 - x.shape[0]))
            return _native_mma_head_op(
                source, packed, self.qtype, self.k, self.mma_identity["path"]
            )[: x.shape[0]]
        op = _native_head_b1_rows_op if self.validate_b1 else _native_head_op
        return op(
            x.float().contiguous(), packed, self.qtype, self.k, self.identity["path"]
        )

    def supports_large_rows(self, rows):
        return (
            type(rows) is int
            and rows > 16
            and rows * 128000 <= 2**31 - 1
            and self.qtype == 13
            and self.k == 2048
            and self.mode == "native-mma"
            and not self.validate_b1
            and bool(self.mma_identity)
        )

    def project_large_rows(
        self,
        inputs,
        packed,
        grid,
        signs,
        values,
        *,
        canonical_rounding,
        canonical_dtype,
    ):
        from .native_head_large import apply

        return apply(
            self,
            inputs,
            packed,
            grid,
            signs,
            values,
            canonical_rounding,
            canonical_dtype,
        )

    def selects_large_rows(self, rows, input_dtype, canonical_rounding):
        # Canonical Half/BF16 Compact already uses a tensor-core GEMM.
        # F32 or unrounded operands take the slower TF32x3 path there.
        return self.supports_large_rows(rows) and not (
            canonical_rounding and input_dtype in (torch.float16, torch.bfloat16)
        )
