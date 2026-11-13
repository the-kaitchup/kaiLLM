# SPDX-License-Identifier: Apache-2.0
"""Optional one-launch native Qwen3.5 FP32 post-convolution preparation.

Scope: equal 16 Q/K/V heads of width128, no input mutation. Q/K retain the
native 256-thread reduction tree; gate/unary math uses the local native CUDA
compiler flags. This is an experimental helper, not a serving default.
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


def library_path():
    return Path(
        os.environ.get(
            "GGUF_NATIVE_POSTCONV_FUSED",
            str(Path(__file__).resolve().parents[1] / "libraries/sm86/libnative_postconv_fused.so"),
        )
    )


def build():
    _, directory, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entry = next(
        e
        for e in json.loads((directory / "compile_commands.json").read_text())
        if e["file"].endswith("/norm.cu")
    )
    command = shlex.split(entry["command"])
    source = Path(__file__).with_suffix(".cu")
    library = library_path()
    library.parent.mkdir(parents=True, exist_ok=True)
    obj = library.with_suffix(".o")
    linked = library.with_suffix(".building.so")
    command[command.index(entry["file"])] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, check=True, cwd=entry["directory"])
    subprocess.run(
        [
            "g++",
            "-shared",
            str(obj),
            f"-L{cuda_library}",
            f"-Wl,-rpath,{cuda_library}",
            "-l:libcudart.so.13",
            "-o",
            str(linked),
        ],
        check=True,
    )
    linked.replace(library)
    library.with_suffix(".build.json").write_text(
        json.dumps(
            {
                "command": command,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return library


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(library_path()))
    lib.native_postconv_run.argtypes = (
        [ctypes.c_void_p] * 10
        + [ctypes.c_int64] * 6
        + [ctypes.c_float] * 2
        + [ctypes.c_int] * 2
        + [ctypes.c_size_t]
    )
    lib.native_postconv_run.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_current::postconv_fused", mutates_args=())
def postconv_fused(
    conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    original_a: torch.Tensor,
    dt: torch.Tensor,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if conv.ndim != 2 or conv.shape[1] != 6144 or conv.stride(1) != 1:
        raise ValueError("Fused prep requires [M,6144] row-strided convolution output")
    m = conv.shape[0]
    if (
        a.shape != (m, 16)
        or b.shape != a.shape
        or original_a.shape != (16,)
        or dt.shape != (16,)
    ):
        raise ValueError("Fused prep requires equal16 Q/K/V heads and scalar gates")
    if any(
        t.dtype != torch.float32 or t.device != conv.device or not t.is_cuda
        for t in (conv, a, b, original_a, dt)
    ):
        raise ValueError("Fused prep requires FP32 CUDA inputs on one device")
    if not original_a.is_contiguous() or not dt.is_contiguous():
        raise ValueError("Original A and dt bias must be contiguous")
    outputs = tuple(
        torch.empty(
            (m, 16, 128) if i < 3 else (m, 16), device=conv.device, dtype=torch.float32
        )
        for i in range(5)
    )
    if not m:
        return outputs
    with torch.cuda.device(conv.device):
        status = _library().native_postconv_run(
            *(t.data_ptr() for t in (conv, a, b, original_a, dt, *outputs)),
            m,
            conv.stride(0),
            *a.stride(),
            *b.stride(),
            1e-6 / 128,
            128**-0.5,
            int(apply_l2norm),
            int(output_g_exp),
            torch.cuda.current_stream(conv.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Fused native post-convolution CUDA error {status}")
    return outputs


@postconv_fused.register_fake
def _fake(conv, a, b, original_a, dt, apply_l2norm=True, output_g_exp=False):
    return tuple(
        torch.empty(
            (conv.shape[0], 16, 128) if i < 3 else (conv.shape[0], 16),
            device=conv.device,
            dtype=torch.float32,
        )
        for i in range(5)
    )


def native_post_conv_prep(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
):
    if (num_k_heads, head_k_dim, head_v_dim, A_log.numel()) != (16, 128, 128, 16):
        raise ValueError("Fused native prep is restricted to Qwen H16/D128")
    original_a = getattr(A_log, "_gguf_native_A", None)
    if original_a is None:
        raise RuntimeError("Fused preparation requires original GGUF ssm_a")
    return postconv_fused(
        conv_output, a, b, original_a, dt_bias, apply_l2norm, output_g_exp
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())
