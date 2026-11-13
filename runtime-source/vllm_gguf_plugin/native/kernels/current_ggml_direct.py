# SPDX-License-Identifier: Apache-2.0
"""Experimental graph-compatible current llama.cpp MMVQ/MMQ Torch operator.

Uses the exact local libggml-cuda arithmetic with FP32 input and output. Torch
owns packed weights, output, scratch, and the CUDA stream. There are no device
copies or stream synchronizations in the operator. Each original GGUF matrix
keeps its row count and native multiplication. The optional grouped entry point
shares activation quantization across compatible layouts and output storage.

This depends on private GGML C++ ABI. Build from the *same* source, compiler,
headers, and definitions as the linked library, using its compile_commands.json:
    PYTHONPATH=benchmarks python -m kernels.current_ggml_direct --build
    PYTHONPATH=benchmarks python -m kernels.current_ggml_direct --build-shared

No external CUDA-graph capture is supported before eager warmup of the shapes
and formats being captured (GGML initializes kernel shared-memory attributes).
This is an experimental benchmark helper, not an installed serving backend.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths

_FORMATS = {
    8: (32, 34),  # Q8_0
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    18: (256, 98),  # IQ3_XXS
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
}


def _library_path():
    return Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_DIRECT",
            str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_ggml_direct.so"),
        )
    )


def _shared_library_path():
    return Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_SHARED",
            str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_ggml_shared.so"),
        )
    )


def build_direct(*, shared=False) -> Path:
    _, build, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(entry for entry in entries if entry["file"].endswith("/ggml-cuda.cu"))
    command = shlex.split(entry["command"])
    source = str(Path(__file__).with_suffix(".cu"))
    library = _shared_library_path() if shared else _library_path()
    library.parent.mkdir(parents=True, exist_ok=True)
    obj = library.with_suffix(".o")
    source_index = command.index(entry["file"])
    command[source_index] = source
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, check=True, cwd=entry["directory"])
    link = [
        "g++",
        "-shared",
        str(obj),
        f"-L{build / 'bin'}",
        f"-L{cuda_library}",
        f"-Wl,-rpath,{build / 'bin'}",
        f"-Wl,-rpath,{cuda_library}",
        "-lggml-cuda",
        "-lggml-base",
        "-l:libcudart.so.13",
        "-l:libcublas.so.13",
        "-o",
        str(library),
    ]
    subprocess.run(link, check=True)
    # Keep exact included kernel revisions next to each independently built
    # direct/shared library. The original libggml-cuda remains unchanged.
    files = [Path(source)] + [
        Path(source).with_name(name)
        for name in (
            "ssm_channel_interleave.cuh",
            "iq_native_m8.cuh",
            "native_iq_m1.cuh",
            "native_q4_m1.cuh",
            "native_ssm_fused_channels.cuh",
            "native_head_m8.cuh",
            "native_head_m1.cuh",
            "native_body_m8.cuh",
        )
    ]
    files.extend([build / "bin/libggml-cuda.so", library])
    library.with_suffix(".build.json").write_text(
        json.dumps(
            {
                "compile": command,
                "link": link,
                "sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files
                },
            },
            indent=2,
        )
        + "\n"
    )
    return library


@lru_cache(maxsize=1)
def _library():
    path = _library_path()
    if not path.is_file():
        raise FileNotFoundError(f"Build the direct GGML helper first: {path}")
    lib = ctypes.CDLL(str(path))
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.current_direct_prepare.argtypes = [ctypes.c_int]
    lib.current_direct_prepare.restype = ctypes.c_int
    shape_args = [ctypes.c_int, ctypes.c_int] + [ctypes.c_int64] * 3
    lib.current_direct_workspace.argtypes = shape_args
    lib.current_direct_workspace.restype = ctypes.c_size_t
    lib.current_direct_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + shape_args + [ctypes.c_size_t]
    )
    lib.current_direct_run.restype = ctypes.c_int
    channel_shape = shape_args + [ctypes.c_int64]
    lib.current_channels_workspace.argtypes = channel_shape
    lib.current_channels_workspace.restype = ctypes.c_size_t
    lib.current_channels_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + channel_shape + [ctypes.c_size_t]
    )
    lib.current_channels_run.restype = ctypes.c_int
    if hasattr(lib, "current_prequant_run"):
        lib.current_prequant_run.argtypes = (
            [ctypes.c_void_p] * 3 + shape_args + [ctypes.c_int, ctypes.c_size_t]
        )
        lib.current_prequant_run.restype = ctypes.c_int
    lib.current_float_prepare.argtypes = [ctypes.c_int]
    lib.current_float_prepare.restype = ctypes.c_int
    float_shape = [ctypes.c_int] + [ctypes.c_int64] * 3
    lib.current_float_route.argtypes = float_shape
    lib.current_float_route.restype = ctypes.c_int
    lib.current_float_workspace.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.current_float_workspace.restype = ctypes.c_size_t
    lib.current_float_run.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + float_shape + [ctypes.c_size_t]
    )
    lib.current_float_run.restype = ctypes.c_int
    lib.current_half_route.argtypes = float_shape
    lib.current_half_route.restype = ctypes.c_int
    lib.current_half_workspace.argtypes = float_shape
    lib.current_half_workspace.restype = ctypes.c_size_t
    lib.current_half_run.argtypes = lib.current_float_run.argtypes
    lib.current_half_run.restype = ctypes.c_int
    return lib


def has_prequant():
    """Frozen reference libraries remain usable without the new entry point."""
    return hasattr(_library(), "current_prequant_run")


@torch.library.custom_op("gguf_current::prequant_linear", mutates_args=())
def current_ggml_prequant(
    q8: torch.Tensor,
    packed: torch.Tensor,
    qtype: int,
    k: int,
    channels: bool = False,
) -> torch.Tensor:
    """Consume exact native Q8_1 bytes for a flat or T1-channel projection."""
    if (
        q8.ndim != 2
        or q8.dtype != torch.uint8
        or not q8.is_cuda
        or not q8.is_contiguous()
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or packed.device != q8.device
        or not packed.is_contiguous()
        or qtype not in _FORMATS
        or k <= 0
    ):
        raise ValueError("Expected contiguous CUDA Q8_1 bytes and GGUF matrix")
    block, block_bytes = _FORMATS[qtype]
    if (
        k % block
        or packed.shape[1] != k // block * block_bytes
        or q8.shape[1] != ((k + 511) // 512 * 512) // 32 * 36
        or not 1 <= q8.shape[0] <= 8
    ):
        raise ValueError("Prequantized shape does not match format/K/M")
    tail = (((-k) % 512) + ((-packed.shape[0]) % 4) * k) // block * block_bytes
    available = packed.untyped_storage().nbytes() - packed.storage_offset()
    if (
        available < packed.numel() + tail
        or packed.data_ptr() % 128
        or q8.data_ptr() % 4
    ):
        raise ValueError("Expected aligned packed storage with native tail padding")
    lib = _library()
    if not has_prequant():
        raise RuntimeError("Native prequantized helper has not been built")
    output = torch.empty(
        (q8.shape[0], packed.shape[0]), device=q8.device, dtype=torch.float32
    )
    with torch.cuda.device(q8.device):
        status = lib.current_prequant_run(
            packed.data_ptr(),
            q8.data_ptr(),
            output.data_ptr(),
            q8.device.index,
            qtype,
            packed.shape[0],
            k,
            q8.shape[0],
            int(channels),
            torch.cuda.current_stream(q8.device).cuda_stream,
        )
        if status:
            raise RuntimeError(lib.current_direct_error().decode())
    return output


@current_ggml_prequant.register_fake
def _fake_prequant(q8, packed, qtype, k, channels=False):
    return torch.empty(
        (q8.shape[0], packed.shape[0]), device=q8.device, dtype=torch.float32
    )


@torch.library.custom_op("gguf_current::direct_linear", mutates_args=())
def current_ggml_direct(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int
) -> torch.Tensor:
    """Dense FP32 matrix multiply using current packed GGML CUDA arithmetic."""
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA FP32 X and uint8 packed matrix")
    if qtype not in _FORMATS:
        raise ValueError(f"Unsupported direct GGML quantization type: {qtype}")
    block, block_bytes = _FORMATS[qtype]
    if k <= 0 or k % block or packed.shape[1] != k // block * block_bytes:
        raise ValueError("Packed matrix width does not match format/K")
    # Current NVIDIA MMVQ can read four rows per block before masking its
    # output stores. Guard incomplete row groups as well as the final K tail.
    tail_bytes = (((-k) % 512) + ((-packed.shape[0]) % 4) * k) // block * block_bytes
    available = packed.untyped_storage().nbytes() - packed.storage_offset()
    if available < packed.numel() + tail_bytes or packed.data_ptr() % 128:
        raise ValueError("Packed matrix needs aligned storage with GGML tail padding")
    n, m = packed.shape[0], x.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if not m:
        return y
    lib = _library()
    with torch.cuda.device(x.device):
        args = (x.device.index, qtype, n, k, m)
        size = lib.current_direct_workspace(*args)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        status = lib.current_direct_run(
            packed.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            size,
            *args,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if status:
            raise RuntimeError(lib.current_direct_error().decode())
    return y


@current_ggml_direct.register_fake
def _fake_current_ggml_direct(x, packed, qtype, k):
    return torch.empty(
        (x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32
    )


class CurrentGGMLDirectLinear(torch.nn.Module):
    """One original GGUF matrix with Torch-owned padded packed storage.

    Supports Q8_0, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_S, IQ3_XXS, and IQ4_XS.
    The final packed row has
    GGML's required zero tail up to a 512-value boundary and guards incomplete
    four-row MMVQ groups. Inputs and outputs remain FP32. Construct a module on
    the device where it will execute; moving it between devices is unsupported.
    """

    def __init__(self, data: torch.Tensor, qtype: int, k: int, *, device=None):
        super().__init__()
        if data.dtype != torch.uint8 or data.ndim != 2 or not data.is_contiguous():
            raise ValueError("Expected contiguous two-dimensional GGUF bytes")
        if qtype not in _FORMATS:
            raise ValueError(f"Unsupported direct GGML quantization type: {qtype}")
        block, size = _FORMATS[qtype]
        if k <= 0 or k % block or data.shape[1] != k // block * size:
            raise ValueError("GGUF row byte count does not match format/width")
        if data.shape[0] <= 0:
            raise ValueError("Matrix must contain at least one row")
        selected = data.device if data.is_cuda else device
        selected = torch.device(
            selected if selected is not None else f"cuda:{torch.cuda.current_device()}"
        )
        if selected.type != "cuda":
            raise ValueError("Current GGML direct kernels require CUDA")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(selected):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Initialize GGML modules before graph capture")
            lib = _library()
            if lib.current_direct_prepare(selected.index):
                raise RuntimeError(lib.current_direct_error().decode())
            tail_bytes = (((-k) % 512) + ((-data.shape[0]) % 4) * k) // block * size
            storage = torch.zeros(
                data.numel() + tail_bytes, dtype=torch.uint8, device=selected
            )
            packed = storage[: data.numel()].view(data.shape)
            packed.copy_(data)
        # Retain padding in the registered buffer, including state_dict saves.
        self.register_buffer("packed_storage", storage)
        self.row_bytes = data.shape[1]
        self._device = selected
        self.qtype = int(qtype)
        self.k = int(k)
        self.shape = (data.shape[0], self.k)

    def forward(self, x: torch.Tensor, bias=None):
        if x.dtype != torch.float32 or x.shape[-1] != self.k:
            raise ValueError("Current GGML direct inputs must be FP32 with matching K")
        if x.device != self._device:
            raise ValueError("Construct the GGML direct module on the input's device")
        packed = self.packed_storage[: self.shape[0] * self.row_bytes].view(
            self.shape[0], self.row_bytes
        )
        y = current_ggml_direct(
            x.reshape(-1, self.k).contiguous(), packed, self.qtype, self.k
        ).reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


@torch.library.custom_op("gguf_current::channels_linear", mutates_args=())
def current_ggml_channels(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int
) -> torch.Tensor:
    """Native GGML [K,T,B] multiplication; Torch input/output are [B,T,K/N]."""
    if (
        x.ndim != 3
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[2] != k
        or packed.ndim != 2
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
        or qtype not in _FORMATS
    ):
        raise ValueError("Expected CUDA FP32 [B,T,K] and contiguous GGUF bytes")
    block, block_bytes = _FORMATS[qtype]
    if k <= 0 or k % block or packed.shape[1] != k // block * block_bytes:
        raise ValueError("Packed matrix width does not match format/K")
    tail = (((-k) % 512) + ((-packed.shape[0]) % 4) * k) // block * block_bytes
    available = packed.untyped_storage().nbytes() - packed.storage_offset()
    if available < packed.numel() + tail or packed.data_ptr() % 128:
        raise ValueError("Packed matrix needs aligned storage with GGML tail padding")
    b, t, _ = x.shape
    n = packed.shape[0]
    if b == 1:
        # Preserve the existing single-request launch exactly.
        return current_ggml_direct(x.view(t, k), packed, qtype, k).view(1, t, n)
    y = torch.empty((b, t, n), device=x.device, dtype=torch.float32)
    if not b or not t:
        return y
    lib = _library()
    with torch.cuda.device(x.device):
        args = (x.device.index, qtype, n, k, t, b)
        size = lib.current_channels_workspace(*args)
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, device=x.device, dtype=torch.uint8)
        status = lib.current_channels_run(
            packed.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            size,
            *args,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if status:
            raise RuntimeError(lib.current_direct_error().decode())
    return y


@current_ggml_channels.register_fake
def _fake_current_ggml_channels(x, packed, qtype, k):
    return x.new_empty((x.shape[0], x.shape[1], packed.shape[0]))


def ssm_channel_groups(tokens: int, starts: list[int]):
    """Contiguous equal-length request groups, as (first_row, T, B).

    Zero-length/padded requests do not participate. Different query lengths
    never share a GGML channel dimension; a singleton retains its B1 launch.
    """
    if (
        not starts
        or starts[0] != 0
        or starts[-1] > tokens
        or any(type(value) is not int or value < 0 for value in starts)
        or any(right < left for left, right in zip(starts, starts[1:]))
    ):
        raise ValueError("Invalid native SSM query boundaries")
    groups = []
    for begin, end in zip(starts, starts[1:]):
        t = end - begin
        if not t:
            continue
        if groups and groups[-1][1] == t:
            first, length, count = groups[-1]
            groups[-1] = (first, length, count + 1)
        else:
            groups.append((begin, t, 1))
    return groups


def _runtime_ssm_groups(tokens):
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if not is_forward_context_available():
        raise RuntimeError("SSM channel projection requires vLLM forward metadata")
    context = get_forward_context()
    if torch.cuda.is_current_stream_capturing():
        if context.cudagraph_runtime_mode.name != "PIECEWISE" or not getattr(
            CudaGraphManager, "_gguf_ssm_decode_guard", False
        ):
            raise RuntimeError("SSM capture requires the protected decode graph")
        # Capture metadata may describe fresh dummy prompts or be absent.
        # The installed graph-manager guard rejects ALL prefill replays,
        # including B1/T4 versus B4/T1 and fresh one-token requests.
        return [(0, 1, tokens)] if tokens else []
    metadata = context.attn_metadata
    if not metadata:
        # vLLM's memory/profile forward has no real requests.
        return [(0, tokens, 1)] if tokens else []
    if not isinstance(metadata, dict):
        raise ValueError("SSM channels do not support microbatch metadata")
    candidates = [m for m in metadata.values() if hasattr(m, "native_query_starts")]
    if not candidates:
        raise ValueError("Native attention CPU request boundaries are required")
    starts = candidates[0].native_query_starts
    if any(m.native_query_starts != starts for m in candidates):
        raise ValueError("Native attention layers disagree on request boundaries")
    return ssm_channel_groups(tokens, starts)


@torch.library.custom_op("gguf_current::ssm_linear", mutates_args=())
def current_ggml_ssm(
    x: torch.Tensor, packed: torch.Tensor, qtype: int, k: int
) -> torch.Tensor:
    """Read live CPU scheduling metadata inside an opaque compiled boundary."""
    if x.ndim != 2:
        raise ValueError("SSM projection expects flattened [tokens,K] activations")
    groups = _runtime_ssm_groups(x.shape[0])
    if len(groups) == 1:
        begin, t, b = groups[0]
        if begin == 0 and t * b == x.shape[0]:
            return current_ggml_channels(x.view(b, t, k), packed, qtype, k).view(
                x.shape[0], packed.shape[0]
            )
    output = x.new_zeros((x.shape[0], packed.shape[0]))
    for begin, t, b in groups:
        end = begin + t * b
        output[begin:end] = current_ggml_channels(
            x[begin:end].view(b, t, k), packed, qtype, k
        ).view(t * b, packed.shape[0])
    return output


@current_ggml_ssm.register_fake
def _fake_current_ggml_ssm(x, packed, qtype, k):
    return x.new_empty((x.shape[0], packed.shape[0]))


class CurrentGGMLSSMLinear(CurrentGGMLDirectLinear):
    """SSM-out only: retain request channels through native multiplication."""

    def forward_channels(self, x):
        packed = self.packed_storage[: self.shape[0] * self.row_bytes].view(
            self.shape[0], self.row_bytes
        )
        return current_ggml_channels(x.contiguous(), packed, self.qtype, self.k)

    def forward(self, x, bias=None):
        packed = self.packed_storage[: self.shape[0] * self.row_bytes].view(
            self.shape[0], self.row_bytes
        )
        y = current_ggml_ssm(
            x.reshape(-1, self.k).contiguous(), packed, self.qtype, self.k
        ).reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


@lru_cache(maxsize=1)
def _shared_library():
    """Separate library preserves the already-built unshared oracle unchanged."""
    path = _shared_library_path()
    if not path.is_file():
        raise FileNotFoundError(f"Build the shared GGML helper first: {path}")
    lib = ctypes.CDLL(str(path))
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.current_direct_prepare.argtypes = [ctypes.c_int]
    lib.current_direct_prepare.restype = ctypes.c_int
    lib.current_shared_layout.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    lib.current_shared_layout.restype = ctypes.c_int
    lib.current_shared_workspace.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.current_shared_workspace.restype = ctypes.c_size_t
    lib.current_shared_run.argtypes = [
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
    lib.current_shared_run.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_current::shared_linear", mutates_args=())
def current_ggml_shared(
    x: torch.Tensor, packed: list[torch.Tensor], qtypes: list[int], k: int
) -> torch.Tensor:
    """Quantize once per compatible native layout, keeping each matrix's N."""
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not 1 <= len(packed) <= 32
        or len(qtypes) != len(packed)
    ):
        raise ValueError("Expected CUDA FP32 input and 1..32 packed matrix/type pairs")
    for weight, qtype in zip(packed, qtypes):
        if (
            qtype not in _FORMATS
            or weight.ndim != 2
            or weight.dtype != torch.uint8
            or weight.device != x.device
            or not weight.is_contiguous()
            or weight.shape[0] <= 0
        ):
            raise ValueError("Expected contiguous CUDA supported GGML matrices")
        block, size = _FORMATS[qtype]
        if k <= 0 or k % block or weight.shape[1] != k // block * size:
            raise ValueError("Packed matrix width does not match shared input")
        tail = (((-k) % 512) + ((-weight.shape[0]) % 4) * k) // block * size
        available = weight.untyped_storage().nbytes() - weight.storage_offset()
        if available < weight.numel() + tail or weight.data_ptr() % 128:
            raise ValueError("Each original matrix needs aligned GGML tail padding")
    rows = [weight.shape[0] for weight in packed]
    y = torch.empty((x.shape[0], sum(rows)), device=x.device, dtype=torch.float32)
    if not x.shape[0]:
        return y
    count = len(packed)
    types_c = (ctypes.c_int * count)(*qtypes)
    rows_c = (ctypes.c_int64 * count)(*rows)
    weights_c = (ctypes.c_void_p * count)(*(weight.data_ptr() for weight in packed))
    lib = _shared_library()
    with torch.cuda.device(x.device):
        size = lib.current_shared_workspace(
            x.device.index, types_c, rows_c, count, k, x.shape[0]
        )
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        status = lib.current_shared_run(
            weights_c,
            types_c,
            rows_c,
            count,
            x.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            size,
            x.device.index,
            k,
            x.shape[0],
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if status:
            raise RuntimeError(lib.current_direct_error().decode())
    return y


@current_ggml_shared.register_fake
def _fake_current_ggml_shared(x, packed, qtypes, k):
    return torch.empty(
        (x.shape[0], sum(weight.shape[0] for weight in packed)),
        device=x.device,
        dtype=torch.float32,
    )


class CurrentGGMLSharedLinear(torch.nn.Module):
    """Original matrices sharing X, native Q8 math, and one contiguous output.

    MMVQ shares Q8_1 across all supported types. MMQ shares only within D4
    (Q8_0/Q6_K and the supported IQ types) or DS4 (Q4_K/Q5_K);
    the native device/type/M dispatch decides
    independently for every matrix, including mixed MMVQ/MMQ on Blackwell.
    Matrices keep their original row counts and reduction grids. Only output
    strides differ, eliminating the subsequent concatenation copy as well.
    """

    def __init__(self, parts, k: int, *, device=None):
        super().__init__()
        if not 1 <= len(parts) <= 32:
            raise ValueError("Expected 1..32 original GGUF matrices")
        self.parts = torch.nn.ModuleList(
            CurrentGGMLDirectLinear(data, int(qtype), k, device=device)
            for data, qtype in parts
        )
        self.k = int(k)
        self.shape = (sum(part.shape[0] for part in self.parts), self.k)
        self._device = self.parts[0]._device
        if any(part._device != self._device for part in self.parts):
            raise ValueError("All shared matrices must reside on the same CUDA device")
        with torch.cuda.device(self._device):
            lib = _shared_library()
            if lib.current_direct_prepare(self._device.index):
                raise RuntimeError(lib.current_direct_error().decode())

    def activation_layouts(self, m):
        """Inspect native dispatch: 0=MMVQ, 1=MMQ D4, 2=MMQ DS4."""
        lib = _shared_library()
        layouts = [
            lib.current_shared_layout(self._device.index, part.qtype, m)
            for part in self.parts
        ]
        if any(layout < 0 for layout in layouts):
            raise RuntimeError(lib.current_direct_error().decode())
        return layouts

    def forward(self, x: torch.Tensor, bias=None):
        if (
            x.dtype != torch.float32
            or x.device != self._device
            or x.shape[-1] != self.k
        ):
            raise ValueError("Shared GGML inputs must be FP32 with matching device/K")
        packed = [
            part.packed_storage[: part.shape[0] * part.row_bytes].view(
                part.shape[0], part.row_bytes
            )
            for part in self.parts
        ]
        y = current_ggml_shared(
            x.reshape(-1, self.k).contiguous(),
            packed,
            [part.qtype for part in self.parts],
            self.k,
        ).reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


@torch.library.custom_op("gguf_current::float_linear", mutates_args=())
def current_ggml_float(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Current GGML dispatch for original F32/F16 weights and F32 activations."""
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.float32
        or weight.dtype not in (torch.float32, torch.float16)
        or not x.is_cuda
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("Expected CUDA F32 input and original F32/F16 weights")
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if not m:
        return y
    lib = _library()
    with torch.cuda.device(x.device):
        shape = (x.device.index, n, k, m)
        is_half = weight.dtype == torch.float16
        route_fn = lib.current_half_route if is_half else lib.current_float_route
        route = route_fn(*shape)
        if route < 0:
            raise RuntimeError(lib.current_direct_error().decode())
        size = (
            lib.current_half_workspace(*shape)
            if is_half
            else lib.current_float_workspace(x.device.index, route)
        )
        scratch = torch.empty(size, device=x.device, dtype=torch.uint8)
        run = lib.current_half_run if is_half else lib.current_float_run
        if run(
            weight.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            size,
            *shape,
            torch.cuda.current_stream(x.device).cuda_stream,
        ):
            raise RuntimeError(lib.current_direct_error().decode())
    return y


@current_ggml_float.register_fake
def _fake_current_ggml_float(x, weight):
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )


class CurrentGGMLFloatLinear(torch.nn.Module):
    """One original F32/F16 GGUF projection with native MMVF/MMF/cuBLAS dispatch.

    Alpha/beta must remain separate matrices. For the actual 16x1024 matrices
    on SM86, M<=3 selects MMVF and M>=4 selects native-style cuBLAS Sgemm.
    cuBLAS uses the native TF32 math mode and a Torch-owned 4 MiB workspace
    (32 MiB on Hopper or newer). Initialization and eager warmup precede capture.
    F16 weights retain native F16 dispatch and cuBLAS input/output conversions.
    """

    def __init__(self, weight: torch.Tensor, *, device=None):
        super().__init__()
        if (
            weight.ndim != 2
            or weight.dtype not in (torch.float32, torch.float16)
            or min(weight.shape) < 1
        ):
            raise ValueError("Expected an original nonempty F32/F16 matrix")
        selected = weight.device if weight.is_cuda else device
        selected = torch.device(
            selected if selected is not None else f"cuda:{torch.cuda.current_device()}"
        )
        if selected.type != "cuda":
            raise ValueError("Current GGML float kernels require CUDA")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(selected):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Initialize GGML float modules before graph capture")
            lib = _library()
            if lib.current_float_prepare(selected.index):
                raise RuntimeError(lib.current_direct_error().decode())
            self.register_buffer("weight", weight.to(selected).contiguous().clone())
        self.shape = tuple(weight.shape)
        self._device = selected

    def forward(self, x: torch.Tensor, bias=None):
        if x.dtype != torch.float32 or x.device != self._device:
            raise ValueError("GGML float inputs must be FP32 on the module's device")
        if x.shape[-1] != self.shape[1]:
            raise ValueError("Input width does not match the original matrix")
        y = current_ggml_float(
            x.reshape(-1, self.shape[1]).contiguous(), self.weight
        ).reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--build-shared", action="store_true")
    arguments = parser.parse_args()
    print(build_direct(shared=arguments.build_shared))
