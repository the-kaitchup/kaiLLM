# SPDX-License-Identifier: Apache-2.0
"""Optional SM86/SM120 K3/K4 GEMV using original wires and FP32 activations."""

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
from .yaqa_sm120_b1_policy import gemv_policy, measured_gemv_enabled, runtime_marker

SOURCES = Path(__file__).parents[1] / "native/kernels"
KINDS = {3: "tile", 4: "tile"}
MAX_WORKSPACE_BYTES = 4 * 1024**2
_FUNNEL_SHAPES = {(6144, 2048), (7680, 2048), (2048, 7680), (2048, 2048)}
_LIBRARIES = {}
_MEASURED_KEYS = set()


def cuda_k34_identity(bits, architecture=None):
    if bits not in KINDS:
        raise ValueError("The CUDA K3/K4 backend requires three or four bits")
    kind = KINDS[bits]
    measured = measured_gemv_enabled() and resolve_architecture(architecture) == (12, 0)
    contract = f"sm86-fp32-k{bits}-{kind}-warps4-prefetch4-shared-rows-v1"
    identity = {
        "bits": bits,
        "kind": kind,
        "measured_b1_policy": measured,
        "measured_policy_marker": runtime_marker(),
        **build_cuda_library(
            SOURCES / (f"yaqa_sm120_k{bits}_split32.cu" if measured else f"yaqa_k{bits}_gemv.cu"),
            contract + ("-sm120-measured-S32-v1" if measured else ""),
            f"yaqa_k{bits}_gemv",
            architecture=architecture,
        ),
    }
    if bits == 3:
        identity["policy_contract"] = "sm86-k3-exact-funnel-four-large-B1-shapes-v2"
        identity["funnel"] = build_cuda_library(
            SOURCES / "yaqa_k3_funnel.cu",
            "sm86-fp32-k3-funnel-warps4-prefetch4-shared-rows-v1",
            "yaqa_k3_funnel",
            architecture=architecture,
        )
    return identity


def cuda_k34_runtime_identity():
    return {
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workspace_limit_bytes": MAX_WORKSPACE_BYTES,
        "kernels": [cuda_k34_identity(bits) for bits in (3, 4)],
    }


def load_cuda_k34(device, bits):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = cuda_k34_identity(bits, architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        tile = identity["kind"] == "tile"
        symbol = f"yaqa_k{bits}_{'tile' if tile else 'gemv'}_rows"
        launch = getattr(library, symbol)
        launch.argtypes = (
            [ctypes.c_void_p] * (5 if tile else 4)
            + [ctypes.c_int] * (7 if tile else 6)
            + [ctypes.c_void_p]
        )
        launch.restype = ctypes.c_int
        libraries = library
        if bits == 3 and not identity["measured_b1_policy"]:
            funnel_library = ctypes.CDLL(identity["funnel"]["library"])
            funnel = funnel_library.yaqa_k3_tile_rows
            funnel.argtypes = launch.argtypes
            funnel.restype = ctypes.c_int
            libraries = (library, funnel_library)
            launch = _dispatch_funnel(launch, funnel)
        _LIBRARIES[key] = (bits, tile, libraries, launch)
        if identity["measured_b1_policy"]:
            _MEASURED_KEYS.add(key)
    return key


def _dispatch_funnel(original, funnel):
    def launch(*arguments):
        if (
            len(arguments) == 13
            and tuple(arguments[5:7]) in _FUNNEL_SHAPES
            and arguments[7] == 1
            and tuple(arguments[8:10]) == (4, 4)
            and arguments[11] == 1
        ):
            return funnel(*arguments)
        return original(*arguments)

    return launch


def tile_splits(m, n):
    """Choose B1 arithmetic independently of the number of validation rows."""
    if m < 128:
        splits = 1
    elif n >= 4096 or m < 4096:
        splits = 8
    elif m >= 7680:
        splits = 2
    else:
        splits = 4
    while splits > 1 and splits * m * 4 > MAX_WORKSPACE_BYTES:
        splits //= 2
    return splits


def _launch(source, trellis, lut, m, n, bits, library_key):
    if (
        source.device.type != "cuda"
        or source.dtype != torch.float32
        or source.ndim != 2
        or source.shape[1] != n
        or source.shape[0] < 1
        or not source.is_contiguous()
        or bits not in (3, 4)
        or m <= 0
        or n <= 0
        or m % 16
        or n % 16
        or trellis.dtype != torch.int16
        or trellis.shape != (m * n // 256, bits * 16)
        or not trellis.is_contiguous()
        or lut.dtype != torch.bfloat16
        or lut.shape != (512, 2)
        or not lut.is_contiguous()
        or trellis.device != source.device
        or lut.device != source.device
    ):
        raise ValueError("Invalid original-wire K3/K4 CUDA GEMV inputs")
    if library_key not in _LIBRARIES:
        raise RuntimeError("K3/K4 CUDA library must load before graph capture")
    validate_library_device(
        library_key, torch.cuda.get_device_capability(source.device)
    )
    loaded_bits, tile, _, launch = _LIBRARIES[library_key]
    if bits != loaded_bits:
        raise ValueError("K3/K4 library identity disagrees with the tensor bits")
    if source.data_ptr() % 8:
        source = source.clone()
    if trellis.data_ptr() % 4:
        trellis = trellis.clone()
    if lut.data_ptr() % 4:
        lut = lut.clone()
    batch = source.shape[0]
    result = torch.empty((batch, m), dtype=torch.float32, device=source.device)
    splits = tile_splits(m, n) if tile else 1
    warps, prefetch, shared = 4, 4, 1
    measured = gemv_policy(bits, m, n) if library_key in _MEASURED_KEYS and batch == 1 else None
    if measured is not None:
        warps, prefetch, splits, shared = measured
    rows_per_chunk = (
        max(1, MAX_WORKSPACE_BYTES // (splits * m * 4)) if splits > 1 else batch
    )
    partial = (
        torch.empty(
            (min(batch, rows_per_chunk), splits, m),
            dtype=torch.float32,
            device=source.device,
        )
        if splits > 1
        else result
    )
    with torch.cuda.device(source.device):
        for begin in range(0, batch, rows_per_chunk):
            rows = min(batch - begin, rows_per_chunk)
            prefix = (
                source.data_ptr() + begin * n * 4,
                trellis.data_ptr(),
                lut.data_ptr(),
            )
            if tile:
                prefix += (partial.data_ptr(),)
            prefix += (result.data_ptr() + begin * m * 4, m, n, rows, warps, prefetch)
            if tile:
                prefix += (splits,)
            error = launch(
                *prefix, shared, torch.cuda.current_stream(source.device).cuda_stream
            )
            if error:
                raise RuntimeError(f"K3/K4 CUDA GEMV failed with CUDA error {error}")
    return result


@torch.library.custom_op("vllm_gguf::yaqa_cuda_k34_gemv", mutates_args=())
def cuda_k34_gemv(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    bits: int,
    library_key: str,
) -> torch.Tensor:
    """Batch independent B1 cores with at most 4 MiB of transient partials."""
    return _launch(source, trellis, lut, m, n, bits, library_key)


@cuda_k34_gemv.register_fake
def _fake(source, trellis, lut, m, n, bits, library_key):
    del trellis, lut, n, bits, library_key
    return source.new_empty((source.shape[0], m), dtype=torch.float32)
