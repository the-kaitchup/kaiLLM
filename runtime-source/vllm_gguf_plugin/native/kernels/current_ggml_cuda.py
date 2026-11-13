# SPDX-License-Identifier: Apache-2.0
"""Persistent diagnostic bridge to the current local llama.cpp CUDA backend.

No quantization arithmetic is reimplemented. Inputs and outputs stay FP32;
GGML selects its current MMVQ/MMQ kernels and owns persistent packed weights.
Forward performs two device-to-device copies and synchronizes the Torch and
GGML streams. It is deliberately not a serving or CUDA-graph implementation.
Backend allocations are outside Torch's allocator/memory accounting.

Build once, without GPU execution:
    python current_ggml_cuda.py --build

Override local defaults with GGUF_CURRENT_GGML_SOURCE, GGUF_CURRENT_GGML_BUILD,
CUDA_HOME, or GGUF_CURRENT_GGML_BRIDGE. The shared library uses the exact headers
and shared backend from those paths; rebuild after changing llama.cpp.
"""

import argparse
import ctypes
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import torch


def _cuda_library_dir(cuda: Path) -> Path:
    """Find CUDA 13 runtime libraries in pip and native toolkit layouts."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    candidates = (
        cuda / "lib",
        cuda / "lib64",
        cuda / "targets" / f"{machine}-linux" / "lib",
    )
    for directory in candidates:
        if (directory / "libcudart.so.13").is_file():
            return directory.resolve()
    searched = ", ".join(str(directory) for directory in candidates)
    raise FileNotFoundError(
        f"CUDA 13 runtime libcudart.so.13 was not found; searched: {searched}"
    )


def _paths():
    source = Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_SOURCE",
            str(Path.cwd() / "llama.cpp"),
        )
    )
    build = Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_BUILD", str(Path.cwd() / "llama.cpp/build")
        )
    )
    cuda = Path(
        os.environ.get(
            "CUDA_HOME",
            str(Path(__import__("sysconfig").get_path("purelib")) / "nvidia/cu13"),
        )
    )
    library = Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_BRIDGE",
            str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_ggml_bridge.so"),
        )
    )
    return source, build, cuda, library


def build_bridge() -> Path:
    source, build, cuda, library = _paths()
    cuda_library = _cuda_library_dir(cuda)
    library.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        str(Path(__file__).with_suffix(".cpp")),
        f"-I{source / 'ggml/include'}",
        f"-I{cuda / 'include'}",
        f"-L{build / 'bin'}",
        f"-L{cuda_library}",
        f"-Wl,-rpath,{build / 'bin'}",
        f"-Wl,-rpath,{cuda_library}",
        "-lggml-cuda",
        "-lggml-base",
        "-l:libcudart.so.13",
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    return library


@lru_cache(maxsize=1)
def _library():
    path = _paths()[3]
    if not path.is_file():
        raise FileNotFoundError(f"Build the current GGML bridge first: {path}")
    library = ctypes.CDLL(str(path))
    library.current_ggml_error.restype = ctypes.c_char_p
    library.current_ggml_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_size_t,
    ]
    library.current_ggml_create.restype = ctypes.c_void_p
    library.current_ggml_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_size_t,
    ]
    library.current_ggml_forward.restype = ctypes.c_int
    library.current_ggml_forward_channels.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_size_t,
    ]
    library.current_ggml_forward_channels.restype = ctypes.c_int
    library.current_ggml_destroy.argtypes = [ctypes.c_void_p]
    library.current_ggml_destroy.restype = None
    return library


class CurrentGGMLLinear(torch.nn.Module):
    """One fixed-device GGUF matrix, with GGML graphs cached by token count.

    ``data`` is contiguous uint8 shaped (N, wire_bytes_per_row), on CPU or CUDA;
    qtype=0/1 instead accepts original F32/F16 weights shaped (N, K).
    CPU bytes upload directly into GGML's buffer; CUDA bytes use a D2D copy.
    The source tensor is not retained after synchronous construction. The
    module cannot be migrated with .to(); construct it on the intended device.
    """

    def __init__(self, data: torch.Tensor, qtype: int, k: int, *, device=None):
        super().__init__()
        expected_dtype = {0: torch.float32, 1: torch.float16}.get(qtype, torch.uint8)
        if data.dtype != expected_dtype or data.ndim != 2 or not data.is_contiguous():
            raise ValueError("Expected contiguous GGUF bytes or original float matrix")
        if qtype not in (0, 1, 8, 12, 13, 14, 18, 21, 22, 23):
            raise ValueError(f"Unsupported current GGML oracle type: {qtype}")
        if k <= 0 or data.shape[0] <= 0:
            raise ValueError("Matrix dimensions must be positive")
        block, size = {
            0: (1, 4),
            1: (1, 2),
            8: (32, 34),
            12: (256, 144),
            13: (256, 176),
            14: (256, 210),
            18: (256, 98),
            21: (256, 110),
            22: (256, 82),
            23: (256, 136),
        }[qtype]
        if k % block or data.shape[1] * data.element_size() != k // block * size:
            raise ValueError(
                "GGUF row byte count does not match quantization type/width"
            )
        selected = data.device if data.is_cuda else device
        self.device = torch.device(
            selected if selected is not None else f"cuda:{torch.cuda.current_device()}"
        )
        if self.device.type != "cuda":
            raise ValueError("This bridge requires a CUDA output device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.n, self.k, self.qtype = int(data.shape[0]), int(k), int(qtype)
        self._lib = _library()
        self._handle = self._lib.current_ggml_create(
            data.data_ptr(),
            data.numel() * data.element_size(),
            self.qtype,
            self.n,
            self.k,
            self.device.index,
            data.is_cuda,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if not self._handle:
            raise RuntimeError(self._lib.current_ggml_error().decode())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32 or x.device != self.device:
            raise ValueError(
                "Current GGML oracle requires FP32 input on its CUDA device"
            )
        if x.ndim < 2 or x.shape[-1] != self.k:
            raise ValueError("Input's final dimension must match the matrix width")
        if not self._handle:
            raise RuntimeError("Current GGML linear has been closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Synchronous GGML diagnostic cannot be CUDA-graph captured"
            )
        flat = x.reshape(-1, self.k).contiguous()
        output = torch.empty(
            (*x.shape[:-1], self.n), device=x.device, dtype=torch.float32
        )
        if flat.shape[0] and self._lib.current_ggml_forward(
            self._handle,
            flat.data_ptr(),
            output.data_ptr(),
            flat.shape[0],
            torch.cuda.current_stream(self.device).cuda_stream,
        ):
            raise RuntimeError(self._lib.current_ggml_error().decode())
        return output

    def forward_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve native [K,T,B,1] MUL_MAT geometry for input [B,T,K]."""
        if x.dtype != torch.float32 or x.device != self.device:
            raise ValueError(
                "Current GGML oracle requires FP32 input on its CUDA device"
            )
        if x.ndim != 3 or x.shape[-1] != self.k:
            raise ValueError("Channel input must have shape [B,T,K]")
        if not self._handle:
            raise RuntimeError("Current GGML linear has been closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Synchronous GGML diagnostic cannot be CUDA-graph captured"
            )
        channels, tokens, _ = x.shape
        contiguous = x.contiguous()
        output = torch.empty(
            (channels, tokens, self.n), device=x.device, dtype=torch.float32
        )
        if channels and tokens and self._lib.current_ggml_forward_channels(
            self._handle,
            contiguous.data_ptr(),
            output.data_ptr(),
            tokens,
            channels,
            torch.cuda.current_stream(self.device).cuda_stream,
        ):
            raise RuntimeError(self._lib.current_ggml_error().decode())
        return output

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.current_ggml_destroy(handle)
            self._handle = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build_bridge())
