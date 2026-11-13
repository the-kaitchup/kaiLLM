"""Exact native text-only RoPE diagnostic for Qwen3.5-0.8B."""

import argparse
import ctypes
import json
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from types import MethodType

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths


def library_path():
    return Path(
        os.environ.get("GGUF_CURRENT_ROPE", str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_rope.so"))
    )


def build():
    _, directory, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entry = next(
        entry
        for entry in json.loads((directory / "compile_commands.json").read_text())
        if entry["file"].endswith("/ggml-cuda.cu")
    )
    command = shlex.split(entry["command"])
    library = library_path()
    obj = library.with_suffix(".o")
    command[command.index(entry["file"])] = str(Path(__file__).with_suffix(".cu"))
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, check=True, cwd=entry["directory"])
    subprocess.run(
        [
            "g++",
            "-shared",
            str(obj),
            f"-L{directory / 'bin'}",
            f"-L{cuda_library}",
            f"-Wl,-rpath,{directory / 'bin'}",
            f"-Wl,-rpath,{cuda_library}",
            "-lggml-cuda",
            "-lggml-base",
            "-l:libcudart.so.13",
            "-o",
            str(library),
        ],
        check=True,
    )


@lru_cache(maxsize=1)
def _library():
    library = ctypes.CDLL(str(library_path()))
    library.current_norm_error.restype = ctypes.c_char_p
    library.current_norm_prepare.argtypes = [ctypes.c_int]
    library.current_norm_prepare.restype = ctypes.c_int
    library.current_rope_run.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int] * 4
        + [ctypes.c_float]
        + [ctypes.c_int] * 2
        + [ctypes.c_size_t]
    )
    library.current_rope_run.restype = ctypes.c_int
    return library


@torch.library.custom_op("gguf_current::rope", mutates_args=())
def current_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 3
        or x.shape[-1] != 256
        or not x.is_contiguous()
        or x.dtype != torch.float32
        or not x.is_cuda
        or positions.shape != (4, x.shape[0])
        or positions.dtype != torch.int32
        or positions.device != x.device
        or not positions.is_contiguous()
    ):
        raise ValueError("Expected FP32 Qwen3.5 [T,H,256] and I32 [4,T] positions")
    result = torch.empty_like(x)
    if not x.numel():
        return result
    library = _library()
    with torch.cuda.device(x.device):
        status = library.current_rope_run(
            x.data_ptr(),
            positions.data_ptr(),
            result.data_ptr(),
            x.shape[0],
            x.shape[1],
            256,
            64,
            10000000.0,
            262144,
            x.device.index,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_norm_error().decode())
    return result


@current_rope.register_fake
def _fake(x, positions):
    return torch.empty_like(x)


def install_native_rope():
    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    original = Qwen35GGUFAdapter.prepare_model
    if getattr(original, "_native_rope", False):
        return

    def forward(module, positions, query, key):
        if positions.ndim == 1:
            pos = positions.repeat(4, 1)
        elif positions.ndim == 2 and positions.shape[0] == 3:
            pos = torch.cat((positions, positions[:1]), dim=0)
        elif positions.ndim == 2 and positions.shape[0] == 4:
            pos = positions
        else:
            raise ValueError("Expected one text axis or three MRoPE position axes")
        pos = pos.to(torch.int32).contiguous()
        q = current_rope(
            query.reshape(-1, query.shape[-1] // 256, 256).contiguous(), pos
        )
        k = current_rope(key.reshape(-1, key.shape[-1] // 256, 256).contiguous(), pos)
        return q.reshape(query.shape), k.reshape(key.shape)

    def prepare(adapter, model):
        original(adapter, model)
        library = _library()
        if library.current_norm_prepare(torch.cuda.current_device()):
            raise RuntimeError(library.current_norm_error().decode())
        for name, module in model.named_modules():
            if name.endswith(".self_attn.rotary_emb"):
                if (
                    module.head_size != 256
                    or module.rotary_dim != 64
                    or module.base != 10000000
                ):
                    raise ValueError(
                        "Native RoPE diagnostic requires Qwen3.5-0.8B dimensions"
                    )
                module.forward = MethodType(forward, module)

    prepare._native_rope = True
    Qwen35GGUFAdapter.prepare_model = prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    build()
