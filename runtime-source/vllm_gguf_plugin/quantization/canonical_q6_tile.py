# SPDX-License-Identifier: Apache-2.0
"""CPU-staged packed CUDA Q8/Q6 experiment; never installs a model hook.

Raw F32 or exact BF16-expanded activations, optional canonical BF16 weights,
F32 output. The reduction differs from compact Triton and requires GPU/F64/EAR
qualification. No GPU work occurs on import or during build().
"""

import ctypes
import hashlib
import importlib.util
import json
import subprocess
from functools import lru_cache
from pathlib import Path

FORMATS = {14: (256, 210)}
CUDA_SOURCE = Path(__file__).with_name("canonical_q6_tile.cu")
FLAGS = (
    "-O3",
    "-std=c++17",
    "--shared",
    "-Xcompiler=-fPIC",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--cudart=none",
    "--cudadevrt=none",
    "-gencode",
    "arch=compute_86,code=sm_86",
    "-gencode",
    "arch=compute_86,code=compute_86",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_layout(m, n, k, qtype, available_bytes):
    if qtype not in FORMATS or not 1 <= m <= 32 or not 0 < n <= 2**31 - 1:
        raise ValueError("Requires Q8_0/Q6_K and positive M<=32/N<=INT_MAX")
    block, size = FORMATS[qtype]
    if not 0 < k <= 2**31 - 1 or k % block:
        raise ValueError("K must contain complete original GGUF blocks")
    if available_bytes < n * (k // block) * size:
        raise ValueError("Original packed weight storage is incomplete")


def build(output, nvcc, cuda_home, header_selector):
    """Build offline with stock-first, source-bound private header selection."""
    output, nvcc = Path(output).absolute(), Path(nvcc).absolute()
    selector_path = Path(header_selector).resolve()
    spec = importlib.util.spec_from_file_location(
        "_raw_gemv_cuda_headers", selector_path
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    choice = selector.select_cuda_headers(str(nvcc), str(Path(cuda_home).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(cuda_home).resolve() / "lib"
    runtime_library = runtime / "libcudart.so.13"
    if not runtime_library.is_file():
        raise ValueError("Pinned CUDA13 shared runtime is unavailable")
    command = [
        str(nvcc),
        *FLAGS,
        "-I" + choice["include_directory"],
        "-L" + str(runtime),
        "-Xlinker=-rpath=" + str(runtime),
        str(CUDA_SOURCE),
        "-Xlinker=" + str(runtime_library),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output.with_suffix(".compile.log").write_text(result.stdout + result.stderr)
    receipt = {
        "kind": "raw-packed-CUDA-F32-GEMV-CPU-build",
        "GPU_executed": False,
        "command": command,
        "returncode": result.returncode,
        "source_sha256": {
            str(p.resolve()): sha(p)
            for p in (Path(__file__), CUDA_SOURCE, selector_path)
        },
        "header_choice": choice,
        "compiler_sha256": sha(nvcc),
        "runtime_library": {
            "path": str(runtime_library),
            "sha256": sha(runtime_library),
        },
        "arithmetic": {
            "FMA": (
                "BF16 register MMA partials with explicit FP32 adds between K256 tiles"
            ),
            "FTZ": False,
            "weight_rounding": "optional BF16 RN before product",
            "activation_quantization_or_sanitizer": False,
            "reference_bitwise_claim": False,
            "supported_operands": "BF16 activations and canonical BF16 weights only",
        },
    }
    if result.returncode == 0:
        receipt["library_sha256"] = sha(output)
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(
            "Offline CUDA build failed: " + str(output.with_suffix(".compile.log"))
        )
    return receipt


def decode_cpu(packed, qtype, k):
    """Independent CPU F32 operands, without a resident runtime expansion."""
    import numpy as np

    raw = np.asarray(packed, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("Expected N by packed-row-bytes")
    n = raw.shape[0]
    validate_layout(1, n, k, qtype, raw.nbytes)
    block, size = FORMATS[qtype]
    if raw.shape[1] != k // block * size:
        raise ValueError("Packed row stride differs")
    b = raw.reshape(n, -1, size)
    if qtype == 8:
        scale = b[..., :2].copy().view("<f2").reshape(n, -1, 1).astype(np.float32)
        return (b[..., 2:].view(np.int8).astype(np.float32) * scale).reshape(n, k)
    pos = np.arange(256)
    half, rem = pos // 128, pos % 128
    group, lane = rem // 32, rem % 32
    ql = b[..., 64 * half + (group % 2) * 32 + lane]
    qh = b[..., 128 + 32 * half + lane]
    code = np.where(group < 2, ql & 15, ql >> 4).astype(np.int32)
    code |= ((qh >> (2 * group)) & 3).astype(np.int32) << 4
    scales = b[..., 192 + 8 * half + 2 * group + lane // 16].view(np.int8)
    d = b[..., 208:210].copy().view("<f2").reshape(n, -1, 1).astype(np.float32)
    return ((code - 32).astype(np.float32) * (scales.astype(np.float32) * d)).reshape(
        n, k
    )


def bf16_round_cpu(value):
    """Finite F32 RN-even operand oracle; preserve NaN instead of integer wrap."""
    import numpy as np

    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan, bits | np.uint32(0x00400000), rounded)
    return rounded.astype(np.uint32).view(np.float32)


def split_count(m, n, k, multiprocessors, partial_budget=4 << 20):
    """Fill available warps using actual geometry, bounded K work and scratch."""
    if min(m, n, k, multiprocessors) < 1:
        raise ValueError("Positive geometry and SM count required")
    grid = ((n + 31) // 32) * ((m + 15) // 16)
    wanted = (32 * multiprocessors + 2 * grid - 1) // (2 * grid)
    splits = 1
    while splits < 8 and splits < wanted and k // (splits * 2) >= 256:
        splits *= 2
    while splits > 1 and 4 * m * n * splits > partial_budget:
        splits //= 2
    return splits


@lru_cache(maxsize=4)
def library(path):
    lib = ctypes.CDLL(path)
    lib.gguf_q6_tile_abi.restype = ctypes.c_int
    if lib.gguf_q6_tile_abi() != 2:
        raise ValueError("Q6 tile ABI differs")
    lib.gguf_q6_tile_pack.argtypes = (
        [ctypes.c_void_p] * 2
        + [ctypes.c_int] * 2
        + [ctypes.c_uint64] * 2
        + [ctypes.c_void_p]
    )
    lib.gguf_q6_tile_run.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int] * 5
        + [ctypes.c_uint64] * 2
        + [ctypes.c_void_p]
    )
    lib.gguf_q6_tile_pack.restype = lib.gguf_q6_tile_run.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def registered_op():
    import torch

    @torch.library.custom_op("gguf_yaqa::canonical_q6_body_raw", mutates_args=())
    def op(
        x: torch.Tensor,
        cache: torch.Tensor,
        n: int,
        k: int,
        output_bf16: bool,
        path: str,
        multiprocessors: int,
        explicit_splits: int,
    ) -> torch.Tensor:
        if (
            x.ndim != 2
            or x.shape[1] != k
            or not 1 <= x.shape[0] <= 32
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or cache.dtype != torch.uint8
            or cache.device != x.device
            or not x.is_contiguous()
            or not cache.is_contiguous()
        ):
            raise ValueError("Contiguous CUDA BF16 operands and Q6 tile cache required")
        output = torch.empty(
            (x.shape[0], n),
            device=x.device,
            dtype=torch.bfloat16 if output_bf16 else torch.float32,
        )
        splits = explicit_splits or split_count(x.shape[0], n, k, multiprocessors)
        if splits not in (1, 2, 4, 8):
            raise ValueError("Positive power-of-two split count at most eight required")
        if splits > 1 and (k // splits < 256 or 4 * x.shape[0] * n * splits > 4 << 20):
            raise ValueError("Explicit split exceeds K-work or scratch-byte admission")
        partial = (
            torch.empty((splits, x.shape[0], n), device=x.device, dtype=torch.float32)
            if splits > 1
            else output
        )
        error = library(path).gguf_q6_tile_run(
            x.data_ptr(),
            cache.data_ptr(),
            output.data_ptr(),
            partial.data_ptr(),
            x.shape[0],
            n,
            k,
            int(output_bf16),
            splits,
            cache.numel(),
            partial.numel() * partial.element_size(),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if error:
            raise RuntimeError(f"Q6 tile run failed: {error}")
        return output

    @op.register_fake
    def fake(x, cache, n, k, output_bf16, path, multiprocessors, explicit_splits):
        return torch.empty(
            (x.shape[0], n),
            device=x.device,
            dtype=torch.bfloat16 if output_bf16 else torch.float32,
        )

    return op


class PreparedRawPackedGEMV:
    """Explicit immutable-weight code cache; rebuild after known wire mutations."""

    def __init__(self, packed, qtype, k, library_path, canonical_rounding=False):
        import torch

        if qtype != 14 or not canonical_rounding or packed.ndim != 2:
            raise ValueError("Canonical original Q6_K matrix required")
        self.packed, self.k, self.n = packed, int(k), packed.shape[0]
        self.explicit_splits = 0
        validate_layout(1, self.n, self.k, qtype, packed.numel())
        if packed.shape[1] != k // 256 * 210:
            raise ValueError("Original Q6_K row shape differs")
        self.path = str(Path(library_path).resolve())
        self.multiprocessors = torch.cuda.get_device_properties(
            packed.device
        ).multi_processor_count
        self.cache = torch.empty(
            ((self.n + 15) // 16 * (self.k // 16) * 256,),
            dtype=torch.uint8,
            device=packed.device,
        )
        self.op = registered_op()
        self.rebuild_cache()

    def rebuild_cache(self):
        import torch

        error = library(self.path).gguf_q6_tile_pack(
            self.packed.data_ptr(),
            self.cache.data_ptr(),
            self.n,
            self.k,
            self.packed.numel(),
            self.cache.numel(),
            torch.cuda.current_stream(self.packed.device).cuda_stream,
        )
        if error:
            raise RuntimeError(f"Q6 tile packing failed: {error}")

    def __call__(self, x, bias=None, *, output_dtype=None):
        import torch

        output_dtype = x.dtype if output_dtype is None else output_dtype
        if x.dtype != torch.bfloat16 or output_dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Canonical BF16 input and BF16/F32 output required")
        output = self.op(
            x.reshape(-1, self.k),
            self.cache,
            self.n,
            self.k,
            output_dtype == torch.bfloat16,
            self.path,
            self.multiprocessors,
            self.explicit_splits,
        ).reshape(*x.shape[:-1], self.n)
        return output if bias is None else output + bias
