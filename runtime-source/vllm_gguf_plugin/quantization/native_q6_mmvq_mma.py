# SPDX-License-Identifier: Apache-2.0
"""Explicit SM86 M4 Q6 head using the original MMVQ activation quantizer."""

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton

from . import native_q6_head as baseline

ENV = "VLLM_GGUF_NATIVE_Q6_MMA_ROWS"


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "m4", "m4-m8", "m4-m8-m16"):
        raise ValueError(ENV + " must be none, m4 or m4-m8")
    return value


def _library_path():
    supplied = os.getenv("VLLM_GGUF_NATIVE_Q6_MMA_LIBRARY")
    return (
        Path(supplied).absolute()
        if supplied
        else Path(__file__).parents[1] / "native/libraries/libq6_mmvq_mma.so"
    )


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(_library_path()))
    lib.q6_mma_error.restype = ctypes.c_char_p
    lib.q6_mma_prepare.argtypes = [ctypes.c_int]
    lib.q6_mma_prepare.restype = ctypes.c_int
    shape = [ctypes.c_int] + [ctypes.c_int64] * 3
    lib.q6_mma_workspace.argtypes = shape
    lib.q6_mma_workspace.restype = ctypes.c_size_t
    lib.q6_mma_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + shape + [ctypes.c_size_t]
    )
    lib.q6_mma_run.restype = ctypes.c_int
    lib.q6_mma_pack.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int]
        + [ctypes.c_int64] * 2
        + [ctypes.c_size_t] * 2
    )
    lib.q6_mma_pack.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_yaqa::q6_mmvq_mma_raw", mutates_args=())
def q6_mma(x: torch.Tensor, packed: torch.Tensor, k: int) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or k <= 0
        or k % 256
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
        or packed.shape[1] != k // 256 * 210
    ):
        raise ValueError(
            "Expected CUDA FP32 activations and original packed Q6_K matrix"
        )
    m, n = x.shape[0], packed.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if not m:
        return y
    lib = _library()
    with torch.cuda.device(x.device):
        shape = (x.device.index, n, k, m)
        size = lib.q6_mma_workspace(*shape)
        if not size:
            raise RuntimeError(lib.q6_mma_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        if lib.q6_mma_run(
            packed.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            size,
            *shape,
            torch.cuda.current_stream(x.device).cuda_stream,
        ):
            raise RuntimeError(lib.q6_mma_error().decode())
    return y


@q6_mma.register_fake
def _fake_q6_mma(x, packed, k):
    return torch.empty(
        (x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32
    )


@lru_cache(maxsize=8)
def prepare(device):
    if torch.cuda.get_device_capability(device) != (8, 6):
        raise ValueError("Qualified Q6 M4 MMA requires SM86")
    status = _library().q6_mma_prepare(device.index)
    if status:
        raise RuntimeError(_library().q6_mma_error().decode())


@torch.library.custom_op("gguf_yaqa::q6_mmvq_mma_guarded", mutates_args=())
def guarded(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.shape[0] not in (4, 8, 16)
        or x.shape[1] != 2048
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or packed.device != x.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (8, 6)
    ):
        raise ValueError("Only original SM86 Q6 [128000,2048] head atM4/M8")
    baseline.validate_layout(
        14,
        packed.shape,
        2048,
        packed.data_ptr(),
        packed.untyped_storage().nbytes() - packed.storage_offset(),
    )
    m = x.shape[0]
    clean, unsafe = baseline.sanitize(x)
    output = q6_mma(clean, packed, 2048)
    if m <= 8:
        baseline._repair_gemv[(triton.cdiv(128000, 4),)](
            x,
            packed,
            output,
            unsafe,
            M=m,
            N=128000,
            K=2048,
            QTYPE=14,
            BLOCK_M=triton.next_power_of_2(m),
            BLOCK_N=4,
            BLOCK_K=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        baseline._repair_gemm[(triton.cdiv(m, 16), triton.cdiv(128000, 32))](
            x,
            packed,
            output,
            unsafe,
            M=m,
            N=128000,
            K=2048,
            QTYPE=14,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return output


@guarded.register_fake
def _fake_guarded(x, packed):
    return x.new_empty((x.shape[0], 128000))


@lru_cache(maxsize=1)
def marker():
    supplied = os.getenv("VLLM_GGUF_NATIVE_Q6_MMA_DEPENDENCY_DIR")
    if supplied:
        dependency_dir = Path(supplied).absolute()
    else:
        names = ("libggml-cuda.so.0", "libggml-base.so.0")
        candidates = [_library_path().parent, Path(__file__).parents[1] / "native/lib"]
        candidates += [
            Path(p) for p in os.getenv("LD_LIBRARY_PATH", "").split(":") if p
        ]
        dependency_dir = next(
            (p.absolute() for p in candidates if all((p / n).is_file() for n in names)),
            None,
        )
        if dependency_dir is None:
            raise ValueError(
                "Set VLLM_GGUF_NATIVE_Q6_MMA_DEPENDENCY_DIR to the exact GGML dependencies"
            )
    files = [
        Path(__file__).resolve(),
        _library_path(),
        Path(__file__).parents[1] / "native/kernels/q6_mmvq_mma_qualified.cu",
    ]
    files += [
        (dependency_dir / "libggml-cuda.so.0"),
        (dependency_dir / "libggml-base.so.0"),
    ]

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    return {
        "mode": mode(),
        "scope": "Q6headM4M8SM86",
        "activation_rows": [4, 8, 16]
        if mode() == "m4-m8-m16"
        else ([4, 8] if mode() == "m4-m8" else [4]),
        "activation_quantization": "originalMMVQ_Q8_1",
        "changed_F32_reduction_order": True,
        "extra_persistent_weight_bytes": 0,
        "whole_model_EAR_required": True,
        "sources": {str(p): sha(p) for p in files},
    }
