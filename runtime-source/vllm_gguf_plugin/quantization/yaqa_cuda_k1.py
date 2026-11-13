# SPDX-License-Identifier: Apache-2.0
"""Optional SM86/SM120 QTIP1 coalesced GEMV and identical row-batched oracle."""

import ctypes
import json
from pathlib import Path

import torch

from ..native.cuda_architecture import bind_library_architecture, device_architecture
from .yaqa_cuda import (
    _LIBRARIES,
    _cuda_k2_gemv_fake,
    _cuda_k2_gemv_rows_fake,
    _cuda_k2_launch,
    build_cuda_library,
)

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_k1_gemv.cu"
CONTRACT = "sm86-fp32-k1-row-pair-prefetch4-warps4-shared-batch-oracle-v1"


def cuda_k1_identity(architecture=None):
    return build_cuda_library(
        SOURCE, CONTRACT, "yaqa_k1_gemv", architecture=architecture
    )


def load_cuda_k1(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = cuda_k1_identity(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        launch = library.yaqa_k1_gemv
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        rows = library.yaqa_k1_gemv_rows
        rows.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        rows.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch, rows)
    return key


@torch.library.custom_op("vllm_gguf::yaqa_cuda_k1_gemv", mutates_args=())
def cuda_k1_gemv(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
) -> torch.Tensor:
    return _cuda_k2_launch(source, trellis, lut, m, n, library_key, bits=1)


@torch.library.custom_op("vllm_gguf::yaqa_cuda_k1_gemv_rows", mutates_args=())
def cuda_k1_gemv_rows(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
) -> torch.Tensor:
    return _cuda_k2_launch(
        source, trellis, lut, m, n, library_key, batched=True, bits=1
    )


cuda_k1_gemv.register_fake(_cuda_k2_gemv_fake)
cuda_k1_gemv_rows.register_fake(_cuda_k2_gemv_rows_fake)
