# SPDX-License-Identifier: Apache-2.0
"""Exact SM86/SM120 latent decoding into a bounded, temporary F16 matrix."""

import ctypes
import hashlib
import json
from pathlib import Path

import torch

from ..native.cuda_architecture import (
    bind_library_architecture,
    device_architecture,
    resolve_architecture,
    validate_library_device,
)
from .yaqa_cuda import build_cuda_library

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_latent_decode.cu"
CONTRACT = "sm86-qtip2-4-latent-half-vector-original-wire-v1"
MAX_WEIGHT_BYTES = 32 * 1024**2
POLICY = {
    ((7680, 2048), 2): 4,
    ((2048, 7680), 2): 8,
    ((6144, 2048), 2): 8,
    ((7680, 2048), 3): 4,
    ((2048, 7680), 3): 8,
    ((6144, 2048), 3): 4,
    ((7680, 2048), 4): 8,
    ((2048, 7680), 4): 8,
    ((6144, 2048), 4): 4,
}
_LIBRARIES = {}


def runtime_identity(architecture=None):
    architecture = resolve_architecture(architecture)
    return {
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library": build_cuda_library(
            SOURCE, CONTRACT, "yaqa_latent_decode", architecture=architecture
        ),
        "policy": {f"{m}x{n}:K{bits}": w for ((m, n), bits), w in POLICY.items()},
        "max_weight_bytes": MAX_WEIGHT_BYTES,
        "minimum_batch": 512,
        "resident_weight_expansion_bytes": 0,
        "device_capability": list(architecture),
    }


def load_decoder(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = runtime_identity(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"]["library"])
        launch = library.yaqa_latent_cuda_run
        launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch)
    return key


def decoder_warps(layer):
    if (
        getattr(layer, "codec", None) != "qtip"
        or not getattr(layer, "permutation", False)
        or not getattr(layer, "native_fp16_gemm", False)
        or not hasattr(layer, "mma_lut")
        or layer.mma_lut.dtype != torch.float16
        or layer.mma_lut.shape != (1024, 2)
        or layer.shape[0] * layer.shape[1] * 2 > MAX_WEIGHT_BYTES
    ):
        return None
    return POLICY.get((tuple(layer.shape), int(layer.bits)))


@torch.library.custom_op("vllm_gguf::yaqa_decode_latent_half_cuda", mutates_args=())
def decode_half(
    trellis: torch.Tensor,
    signed: torch.Tensor,
    m: int,
    n: int,
    bits: int,
    warps: int,
    key: str,
) -> torch.Tensor:
    if (
        key not in _LIBRARIES
        or trellis.device.type != "cuda"
        or trellis.dtype != torch.int16
        or trellis.shape != (m * n // 256, 16 * bits)
        or not trellis.is_contiguous()
        or signed.dtype != torch.float16
        or signed.shape != (1024, 2)
        or not signed.is_contiguous()
        or signed.device != trellis.device
        or bits not in (2, 3, 4)
        or m <= 0
        or n <= 0
        or m % 16
        or n % 16
        or m * n * 2 > MAX_WEIGHT_BYTES
        or warps not in (4, 8)
    ):
        raise ValueError("Invalid loaded CUDA latent decoder inputs")
    validate_library_device(key, torch.cuda.get_device_capability(trellis.device))
    if trellis.data_ptr() % 4:
        trellis = trellis.clone()
    if signed.data_ptr() % 4:
        signed = signed.clone()
    result = torch.empty((m, n), device=trellis.device, dtype=torch.float16)
    _, launch = _LIBRARIES[key]
    with torch.cuda.device(trellis.device):
        error = launch(
            trellis.data_ptr(),
            signed.data_ptr(),
            result.data_ptr(),
            m,
            n,
            bits,
            warps,
            1,
            0,
            torch.cuda.current_stream(trellis.device).cuda_stream,
        )
    if error:
        raise RuntimeError(f"CUDA latent decoder failed with CUDA error {error}")
    return result


@decode_half.register_fake
def _fake_decode_half(trellis, signed, m, n, bits, warps, key):
    del signed, bits, warps, key
    return trellis.new_empty((m, n), dtype=torch.float16)
