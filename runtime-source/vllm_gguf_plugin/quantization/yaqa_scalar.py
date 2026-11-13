# SPDX-License-Identifier: Apache-2.0
"""Optional packed ternary CUDA with original BF16 weights and activations."""

import ctypes
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from ..native.cuda_architecture import (
    bind_library_architecture,
    device_architecture,
    validate_library_device,
)
from .yaqa_cuda import build_cuda_library

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_ternary_gemv.cu"
CONTRACT = "sm86-original-five-trits-bf16-input-weight-fp32-fma-v1"
_LIBRARIES = {}


@lru_cache(maxsize=2)
def _identity(architecture=None):
    return build_cuda_library(
        SOURCE, CONTRACT, "yaqa_ternary_gemv", architecture=architecture
    )


def scalar_runtime_marker():
    mode = os.getenv("VLLM_GGUF_YAQA_SCALAR", "triton")
    if mode not in ("triton", "cuda"):
        raise ValueError("VLLM_GGUF_YAQA_SCALAR must be triton or cuda")
    return {"mode": mode, **(_identity() if mode == "cuda" else {})}


def load_cuda_ternary(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = _identity(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        function = library.yaqa_ternary_run
        function.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        )
        function.restype = ctypes.c_int
        _LIBRARIES[key] = (library, function)
    return key


@torch.library.custom_op("vllm_gguf::yaqa_ternary", mutates_args=())
def cuda_ternary(
    source: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
) -> torch.Tensor:
    batch = source.shape[0] if source.ndim == 2 else 0
    if (
        source.device.type != "cuda"
        or source.dtype != torch.bfloat16
        or source.ndim != 2
        or source.shape != (batch, n)
        or not 1 <= batch <= 8
        or m < 1
        or n not in (2048, 7680)
        or m * n > 2**31 - 1
        or packed.dtype != torch.int8
        or packed.numel() != (m * n + 4) // 5
        or scales.dtype != torch.float32
        or scales.shape != (m,)
        or any(
            t.device != source.device or not t.is_contiguous()
            for t in (source, packed, scales)
        )
    ):
        raise ValueError("Invalid original ternary CUDA inputs")
    if library_key not in _LIBRARIES:
        raise RuntimeError("Ternary CUDA must load before graph capture")
    validate_library_device(
        library_key, torch.cuda.get_device_capability(source.device)
    )
    output = torch.empty((batch, m), device=source.device, dtype=torch.float32)
    # Each thread visits its bytes in ascending order for every prefetch choice.
    prefetch = 4 if batch == 1 else 1
    error = _LIBRARIES[library_key][1](
        source.data_ptr(),
        packed.data_ptr(),
        scales.data_ptr(),
        output.data_ptr(),
        m,
        n,
        batch,
        4,
        prefetch,
        torch.cuda.current_stream(source.device).cuda_stream,
    )
    if error:
        raise RuntimeError(f"Ternary CUDA failed with code {error}")
    return output


@cuda_ternary.register_fake
def _cuda_ternary_fake(source, packed, scales, m, n, library_key):
    return source.new_empty((source.shape[0], m), dtype=torch.float32)
