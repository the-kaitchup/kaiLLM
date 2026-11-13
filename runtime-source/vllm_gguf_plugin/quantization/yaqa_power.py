# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM86/SM120 power-input transform with the current explicit FMA contract."""

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton

from ..native.cuda_architecture import (
    bind_library_architecture,
    device_architecture,
    validate_library_device,
)
from .yaqa_cuda import build_cuda_library

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_power_hadamard.cu"
CONTRACT = "sm86-power2048-input-shuffle5-explicit-level0-fma-threads256-v1"
_LIBRARIES = {}
_DTYPES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def power_hadamard_mode():
    mode = os.getenv("VLLM_GGUF_YAQA_HADAMARD", "triton")
    if mode not in ("triton", "cuda"):
        raise ValueError("VLLM_GGUF_YAQA_HADAMARD must be triton or cuda")
    return mode


@lru_cache(maxsize=2)
def _build(architecture=None):
    return build_cuda_library(
        SOURCE, CONTRACT, "yaqa_power_hadamard", architecture=architecture
    )


def _power_marker_sm86():
    mode = power_hadamard_mode()
    marker = {
        "mode": mode,
        "wrapper_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "triton": triton.__version__,
        "prefill_factor_source_sha256": hashlib.sha256(
            Path(__file__).with_name("yaqa_hadamard.py").read_bytes()
        ).hexdigest(),
        "prefill_factor_policy": "post-only-real6144x12-7680x60-at512or2048",
        "prefill_input_factor_storage": "exactF32-shadow-7680k60-at512or2048",
        "prefill_input_factor_loader_sha256": hashlib.sha256(
            Path(__file__).with_name("yaqa.py").read_bytes()
        ).hexdigest(),
        "prefill_input_factor_dispatch_sha256": hashlib.sha256(
            Path(__file__).with_name("yaqa_kernels.py").read_bytes()
        ).hexdigest(),
    }
    if mode == "cuda":
        marker.update(_build())
    return marker

def _power_marker_sm120():
    mode = power_hadamard_mode()
    from .yaqa_sm120_b1_policy import runtime_marker
    marker = {"opaque_dispatch": os.getenv("VLLM_GGUF_YAQA_OPAQUE", "0"), "opaque_source_sha256": hashlib.sha256(Path(__file__).with_name("yaqa_opaque.py").read_bytes()).hexdigest(), "SM120_measured": runtime_marker(),
        "mode": mode,
        "wrapper_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "triton": triton.__version__,
    }
    if mode == "cuda":
        marker.update(_build())
    return marker

def power_hadamard_runtime_marker():
    marker = _power_marker_sm120()
    old = _power_marker_sm86()
    for key, value in old.items():
        if key.startswith("prefill_"):
            marker[key] = value
    marker["prefill_factor_policy_qualified_capabilities"] = [[8, 6]]
    return marker


def load_cuda_power(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = _build(architecture)
    key = identity["library_sha256"]
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        lib = ctypes.CDLL(identity["library"])
        lib.power_hadamard_run.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 9
            + [
                ctypes.c_float,
                ctypes.c_size_t,
            ]
        )
        lib.power_hadamard_run.restype = ctypes.c_int
        lib.power_hadamard_error.argtypes = [ctypes.c_int]
        lib.power_hadamard_error.restype = ctypes.c_char_p
        _LIBRARIES[key] = lib
    return key


@torch.library.custom_op("gguf_yaqa::cuda_power_input", mutates_args=())
def cuda_power_input(x: torch.Tensor, scale: torch.Tensor, key: str) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.shape[1] != 2048
        or not x.is_cuda
        or x.dtype not in _DTYPES
        or not x.is_contiguous()
        or scale.shape != (2048,)
        or scale.dtype != torch.float32
        or scale.device != x.device
        or not scale.is_contiguous()
    ):
        raise ValueError("CUDA power input requires contiguous CUDA rows of width2048")
    validate_library_device(key, torch.cuda.get_device_capability(x.device))
    output = torch.empty_like(x, dtype=torch.float32)
    if x.data_ptr() % 16 or scale.data_ptr() % 16:
        from .yaqa_kernels import _power_hadamard

        _power_hadamard[(x.shape[0], 1)](
            x, scale, output, N=2048, P=2048, LOG_P=11, PRE=True, BLOCK=2048
        )
        return output
    lib = _LIBRARIES[key]
    with torch.cuda.device(x.device):
        status = lib.power_hadamard_run(
            x.data_ptr(),
            scale.data_ptr(),
            scale.data_ptr(),
            output.data_ptr(),
            2048,
            2048,
            x.shape[0],
            _DTYPES[x.dtype],
            0,
            0,
            256,
            2,
            0,
            2048**-0.5,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(lib.power_hadamard_error(status).decode())
    return output


@cuda_power_input.register_fake
def _fake_cuda_power_input(x, scale, key):
    return torch.empty_like(x, dtype=torch.float32)
