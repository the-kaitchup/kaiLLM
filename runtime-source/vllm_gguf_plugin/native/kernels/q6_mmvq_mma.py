# SPDX-License-Identifier: Apache-2.0
"""Experimental Q6_K INT-MMA with exact current MMVQ Q8_1 activations.

The current native quantizer produces signed Q8 codes and FP16 scales. A small
kernel rearranges those bytes into MMQ's transposed D4 layout, widening scales
exactly to FP32. The current Q6_K integer tensor-core kernel then consumes them.
This preserves activation quantization, but changes floating reduction order;
output equality with MMVQ must be measured. No serving integration is enabled.

Build against the same local GGML library/compiler as current_ggml_direct:
    PYTHONPATH=benchmarks python -m kernels.q6_mmvq_mma --build
"""

import argparse
import ctypes
import json
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths
from .current_ggml_direct import CurrentGGMLDirectLinear


def _library_path():
    return Path(
        os.environ.get("GGUF_Q6_MMA_LIBRARY", str(Path(__file__).resolve().parents[1] / "libraries/sm86/libq6_mmvq_mma.so"))
    )


def build():
    _, root, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entries = json.loads((root / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/ggml-cuda.cu"))
    command = shlex.split(entry["command"])
    library = _library_path()
    obj = library.with_suffix(".o")
    command[command.index(entry["file"])] = str(Path(__file__).with_suffix(".cu"))
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=entry["directory"], check=True)
    subprocess.run(
        [
            "g++",
            "-shared",
            str(obj),
            f"-L{root / 'bin'}",
            f"-L{cuda_library}",
            f"-Wl,-rpath,{root / 'bin'}",
            f"-Wl,-rpath,{cuda_library}",
            "-lggml-cuda",
            "-lggml-base",
            "-l:libcudart.so.13",
            "-l:libcublas.so.13",
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


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


@torch.library.custom_op("gguf_q6_mma::linear", mutates_args=())
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


def activation_blocks(x):
    """Return native MMVQ blocks and converted MMQ blocks for byte-level checks."""
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA FP32 input")
    m, k = x.shape
    if not m or not k or k % 256:
        raise ValueError("Expected nonempty rows and Q6_K-compatible width")
    padded = (k + 511) // 512 * 512
    q8 = torch.empty((m, padded // 32, 36), device=x.device, dtype=torch.uint8)
    mmq = torch.empty(
        (padded // 128 * m + 128, 144), device=x.device, dtype=torch.uint8
    )
    lib = _library()
    with torch.cuda.device(x.device):
        if lib.q6_mma_pack(
            x.data_ptr(),
            q8.data_ptr(),
            mmq.data_ptr(),
            x.device.index,
            k,
            m,
            mmq.numel(),
            torch.cuda.current_stream(x.device).cuda_stream,
        ):
            raise RuntimeError(lib.q6_mma_error().decode())
    return q8, mmq


class Q6MMVQMMA(CurrentGGMLDirectLinear):
    """Original Q6 matrix with MMVQ quantization and MMQ integer accumulation."""

    def __init__(self, data, k, *, device=None):
        super().__init__(data, 14, k, device=device)
        lib = _library()
        if lib.q6_mma_prepare(self._device.index):
            raise RuntimeError(lib.q6_mma_error().decode())

    def forward(self, x, bias=None):
        if (
            x.dtype != torch.float32
            or x.shape[-1] != self.k
            or x.device != self._device
        ):
            raise ValueError("Expected FP32 input on the matrix device with matching K")
        packed = self.packed_storage[: self.shape[0] * self.row_bytes].view(
            self.shape[0], self.row_bytes
        )
        y = q6_mma(x.reshape(-1, self.k).contiguous(), packed, self.k)
        y = y.reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())
