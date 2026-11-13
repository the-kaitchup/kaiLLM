# SPDX-License-Identifier: Apache-2.0
"""Optional SM86/SM120 QTIP2 GEMV using original packed words and FP32 activations."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from ..native.cuda_architecture import (
    architecture_identity,
    bind_library_architecture,
    device_architecture,
    validate_library_device,
)

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_k2_gemv.cu"
BUILD_CONTRACT = "sm86-fp32-k2-row-pair-prefetch4-warps4-shared-batch-oracle-v2"
_LIBRARIES = {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def build_cuda_library(source, contract, stem, architecture=None):
    """Build a source-identified optional CUDA library without using the GPU."""
    target = architecture_identity(architecture)
    # kaiLLM deployment: reuse qualified native bytes without rewriting identity.
    native_assets = os.getenv("KAILLM_NATIVE_ASSETS")
    if native_assets:
        from kaillm.native_assets import resolve

        effective_contract = contract.replace(
            "sm86-", target["target"].replace("_", "") + "-", 1
        )
        frozen = resolve(native_assets, source, effective_contract, stem, target)
        if frozen is not None:
            return frozen
    nvcc = shutil.which("nvcc")
    if nvcc is None and os.getenv("CUDA_HOME"):
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc")
    if nvcc is None or not Path(nvcc).is_file():
        raise RuntimeError("The opted-in YAQA CUDA backend requires nvcc")
    configured_cuda_home = os.getenv("CUDA_HOME")
    cuda_home = (
        Path(configured_cuda_home)
        if configured_cuda_home
        else Path(nvcc).resolve().parents[1]
    )
    from ..native.cuda_toolchain import select_cuda_headers

    compatibility = select_cuda_headers(str(nvcc), str(cuda_home))
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    identity = dict(
        contract=contract.replace("sm86-", target["target"].replace("_", "") + "-", 1),
        architecture=target,
        source_sha256=_sha256(source),
        compiler=str(Path(nvcc).resolve()),
        compiler_version=version,
        host_header_compatibility=compatibility["identity"],
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = root / "vllm_gguf_plugin/yaqa_cuda" / key
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"lib{stem}.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            overlay = Path(compatibility["include_directory"])
            directories = [
                cuda_home / "lib",
                cuda_home / "lib64",
                cuda_home / "targets/x86_64-linux/lib",
            ]
            runtime = next(
                (
                    candidate
                    for directory in directories
                    for candidate in sorted(directory.glob("libcudart.so*"))
                    if candidate.is_file()
                ),
                None,
            )
            if runtime is None:
                raise RuntimeError("CUDA runtime library was not found")
            link = cache / "libcudart.so"
            if not link.exists():
                # Keep the valid SDK pathname: executable-backed mounts can
                # resolve a readable library to a non-addressable /memfd name.
                link.symlink_to(runtime.absolute())
            temporary = cache / f".library-{os.getpid()}.so"
            command = [
                nvcc,
                "-O3",
                "-std=c++17",
                "-shared",
                "-Xcompiler=-fPIC",
                "--cudart",
                "shared",
                "--cudadevrt",
                "none",
                "-I",
                str(overlay),
                "-L",
                str(cache),
                f"-Xlinker=-rpath={runtime.parent}",
                target["gencode"],
                str(source),
                "-o",
                str(temporary),
            ]
            with (cache / "build.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"YAQA CUDA build failed; see {cache / 'build.log'}")
            temporary.replace(library)
        identity.update(library=str(library), library_sha256=_sha256(library))
        (cache / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


@lru_cache(maxsize=2)
def _build_library(architecture=None):
    return build_cuda_library(
        SOURCE, BUILD_CONTRACT, "yaqa_k2_gemv", architecture=architecture
    )


def cuda_k2_runtime_marker(mode=None):
    """Identity for engine caches; compiling the optional binary needs no GPU."""
    mode = os.getenv("VLLM_GGUF_YAQA_GEMV", "triton") if mode is None else mode
    if mode not in ("triton", "cuda-k2", "cuda-k12", "cuda-k1234"):
        raise ValueError(
            "VLLM_GGUF_YAQA_GEMV must be triton, cuda-k2, cuda-k12 or cuda-k1234"
        )
    if mode == "triton":
        return json.dumps(
            {"mode": mode, "validate_b1": os.getenv("VLLM_GGUF_YAQA_VALIDATE_B1", "0")},
            sort_keys=True,
        )
    identity = {
        "mode": mode,
        "validate_b1": os.getenv("VLLM_GGUF_YAQA_VALIDATE_B1", "0"),
        **_build_library(),
    }
    if mode in ("cuda-k12", "cuda-k1234"):
        from .yaqa_cuda_k1 import cuda_k1_identity

        identity["k1"] = cuda_k1_identity()
    if mode == "cuda-k1234":
        from .yaqa_cuda_k34 import cuda_k34_runtime_identity

        identity["k34"] = cuda_k34_runtime_identity()
    return json.dumps(identity, sort_keys=True)


def load_cuda_k2(device):
    """Load only on the validated Ampere target; other devices keep Triton."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        return None
    identity = _build_library(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        launch = library.yaqa_k2_gemv
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        launch_rows = library.yaqa_k2_gemv_rows
        launch_rows.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        launch_rows.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch, launch_rows)
    return key


def _cuda_k2_launch(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
    batched: bool = False,
    bits: int = 2,
) -> torch.Tensor:
    if (
        source.device.type != "cuda"
        or source.dtype != torch.float32
        or source.ndim != 2
        or source.shape[1] != n
        or source.shape[0] < 1
        or (not batched and source.shape[0] != 1)
        or not source.is_contiguous()
        or trellis.dtype != torch.int16
        or not trellis.is_contiguous()
        or trellis.shape != (m * n // 256, 16 * bits)
        or lut.dtype != torch.bfloat16
        or lut.shape != (512, 2)
        or not lut.is_contiguous()
        or m <= 0
        or n <= 0
        or m % 16
        or n % 16
        or trellis.device != source.device
        or lut.device != source.device
    ):
        raise ValueError(f"Invalid packed QTIP{bits} CUDA GEMV inputs")
    # Public tensor views may be contiguous yet misaligned for float2/u32.
    if source.data_ptr() % 8:
        source = source.clone()
    if trellis.data_ptr() % 4:
        trellis = trellis.clone()
    if lut.data_ptr() % 4:
        lut = lut.clone()
    if library_key not in _LIBRARIES:
        raise RuntimeError("YAQA CUDA GEMV must load before graph capture")
    validate_library_device(
        library_key, torch.cuda.get_device_capability(source.device)
    )
    batch = source.shape[0]
    result = torch.empty((batch, m), dtype=torch.float32, device=source.device)
    extra = (batch,) if batched else ()
    error = _LIBRARIES[library_key][2 if batched else 1](
        source.data_ptr(),
        trellis.data_ptr(),
        lut.data_ptr(),
        result.data_ptr(),
        m,
        n,
        *extra,
        4,
        4,
        1,
        torch.cuda.current_stream(source.device).cuda_stream,
    )
    if error:
        raise RuntimeError(f"YAQA CUDA GEMV launch failed with CUDA error {error}")
    return result


@torch.library.custom_op("vllm_gguf::yaqa_cuda_k2_gemv", mutates_args=())
def cuda_k2_gemv(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
) -> torch.Tensor:
    return _cuda_k2_launch(source, trellis, lut, m, n, library_key)


@torch.library.custom_op("vllm_gguf::yaqa_cuda_k2_gemv_rows", mutates_args=())
def cuda_k2_gemv_rows(
    source: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    m: int,
    n: int,
    library_key: str,
) -> torch.Tensor:
    """Validation-only batching of exactly the same independent B1 row core."""
    return _cuda_k2_launch(source, trellis, lut, m, n, library_key, batched=True)


@cuda_k2_gemv.register_fake
def _cuda_k2_gemv_fake(source, trellis, lut, m, n, library_key):
    del trellis, lut, n, library_key
    return source.new_empty((1, m), dtype=torch.float32)


@cuda_k2_gemv_rows.register_fake
def _cuda_k2_gemv_rows_fake(source, trellis, lut, m, n, library_key):
    del trellis, lut, n, library_key
    return source.new_empty((source.shape[0], m), dtype=torch.float32)
