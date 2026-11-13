# SPDX-License-Identifier: Apache-2.0
"""Finite-test wrapper for independent exact power transforms in one launch."""

import ctypes
import hashlib
from pathlib import Path

_LIBRARY = None
_DTYPES = {"torch.float32": 0, "torch.bfloat16": 1, "torch.float16": 2}


def load(path, expected_sha256):
    global _LIBRARY
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Grouped input library differs from its source-bound identity")
    library = ctypes.CDLL(str(path))
    call = library.power_hadamard_grouped_input_run
    call.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_size_t]
    )
    call.restype = ctypes.c_int
    _LIBRARY = library, call


def geometry(groups, rows):
    return (
        type(groups) is int
        and type(rows) is int
        and groups > 0
        and rows > 0
        and groups <= 65535
        and groups * rows * 2048 * 4 <= 4 << 20
    )


def apply(x, grouped_scale):
    import torch

    if (
        _LIBRARY is None
        or x.ndim != 2
        or grouped_scale.ndim != 2
        or x.shape[1] != 2048
        or grouped_scale.shape[1] != 2048
        or str(x.dtype) not in _DTYPES
        or grouped_scale.dtype != torch.float32
        or not x.is_cuda
        or grouped_scale.device != x.device
        or tuple(torch.cuda.get_device_capability(x.device)) != (8, 6)
        or not x.is_contiguous()
        or not grouped_scale.is_contiguous()
        or x.data_ptr() % 16
        or grouped_scale.data_ptr() % 16
        or not geometry(grouped_scale.shape[0], x.shape[0])
    ):
        raise ValueError("Aligned original SM86 X[B,2048]/SU[G,2048] required")
    groups, batch = grouped_scale.shape[0], x.shape[0]
    result = torch.empty((groups, batch, 2048), device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _LIBRARY[1](
            x.data_ptr(),
            grouped_scale.data_ptr(),
            result.data_ptr(),
            groups,
            batch,
            _DTYPES[str(x.dtype)],
            2048**-0.5,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Grouped independent power input failed:{status}")
    return result
