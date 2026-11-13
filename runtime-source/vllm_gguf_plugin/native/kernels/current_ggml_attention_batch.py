# SPDX-License-Identifier: Apache-2.0
"""True multi-stream native FA for uniform decode and opt-in uniform prefill.

This uses llama.cpp's B-stream dispatch/reductions and a common padded KV
span. It intentionally differs from independent B1 attention calls. Physical
KV capacity is independent of the logical mask width. No production defaults
or base attention files are changed; the caller opts into this operator.
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
            "GGUF_CURRENT_ATTN_BATCH",
            str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_attention_batch.so"),
        )
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_build_manifest(library, source):
    """Reject an ABI-compatible binary with stale physical-mask semantics."""
    manifest_path = library.with_suffix(".build.json")
    rebuild = "Rebuild with python -m kernels.current_ggml_attention_batch --build"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Missing or invalid native attention build manifest {manifest_path}. "
            f"{rebuild}"
        ) from error
    paths = {
        "library_sha256": library,
        "source_sha256": source,
        "included_source_sha256": source.with_name("current_ggml_attention.cu"),
    }
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), str)
        or len(manifest[key]) != 64
        or any(c not in "0123456789abcdef" for c in manifest[key])
        for key in paths
    ):
        raise RuntimeError(
            f"Invalid native attention build manifest {manifest_path}. {rebuild}"
        )
    for key, path in paths.items():
        try:
            actual = _sha256(path)
        except OSError as error:
            raise RuntimeError(
                f"Cannot verify native attention {path}. {rebuild}"
            ) from error
        if manifest[key] != actual:
            raise RuntimeError(f"Native attention {key} mismatch for {path}. {rebuild}")
    return {"path": str(library.resolve()), **{key: manifest[key] for key in paths}}


@lru_cache(maxsize=1)
def library_provenance():
    """Validate once per process without loading CUDA or creating a context."""
    return _validate_build_manifest(library_path(), Path(__file__).with_suffix(".cu"))


def build():
    _, directory, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entries = json.loads((directory / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/ggml-cuda.cu"))
    command = shlex.split(entry["command"])
    source = Path(__file__).with_suffix(".cu")
    source_hash = _sha256(source)
    included_hash = _sha256(source.with_name("current_ggml_attention.cu"))
    library = library_path()
    library.parent.mkdir(parents=True, exist_ok=True)
    obj, linked = library.with_suffix(".o"), library.with_suffix(".building.so")
    command[command.index(entry["file"])] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=entry["directory"], check=True)
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
            str(linked),
        ],
        check=True,
    )
    linked.replace(library)
    manifest_path = library.with_suffix(".build.json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                "command": command,
                "source_sha256": source_hash,
                "included_source_sha256": included_hash,
                "library_sha256": _sha256(library),
            },
            indent=2,
        )
        + "\n"
    )
    temporary_manifest.replace(manifest_path)
    return library


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(library_provenance()["path"])
    lib.current_attention_error.restype = ctypes.c_char_p
    lib.current_attention_batch_run.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_size_t]
        + [ctypes.c_int] * 4
        + [ctypes.c_float, ctypes.c_size_t]
    )
    lib.current_attention_batch_run.restype = ctypes.c_int
    return lib


def _launch_native_batch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    scale: float,
) -> None:
    """Q/out [B,1,8,256], KV [B,capacity,2,256], mask [B,64,logical_K].

    The caller supplies a valid attention mask for every stream. Inactive
    padded streams need one visible zero-valued KV entry; all -inf masks have
    undefined softmax. Active streams remain independent mathematically.
    Shapes are fixed during CUDA graph capture; mask and KV contents are live.
    """
    if (
        q.ndim != 4
        or q.shape[1:] != (1, 8, 256)
        or not 1 <= q.shape[0] <= 65535
        or q.dtype != torch.float32
        or k.ndim != 4
        or k.shape[0] != q.shape[0]
        or k.shape[2:] != (2, 256)
        or k.shape[1] % 256
        or v.shape != k.shape
        or mask.ndim != 3
        or mask.shape[:2] != (q.shape[0], 64)
        or mask.shape[2] <= 0
        or mask.shape[2] > k.shape[1]
        or mask.shape[2] % 256
        or any(t.dtype != torch.float16 for t in (k, v, mask))
        or any(
            not t.is_cuda or not t.is_contiguous() or t.device != q.device
            for t in (q, k, v, mask, output, scratch)
        )
        or output.shape != q.shape
        or output.dtype != torch.float32
        or scratch.dtype != torch.uint8
        or scratch.numel() < 16 * 1024 * 1024
    ):
        raise ValueError("Expected uniform F32 decode Q, F16 KV/mask, padded KV span")
    library = _library()
    with torch.cuda.device(q.device):
        status = library.current_attention_batch_run(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            mask.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            scratch.numel(),
            q.device.index,
            q.shape[0],
            k.shape[1],
            mask.shape[2],
            float(scale),
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_attention_error().decode())


@torch.library.custom_op("gguf_current::attention_batch", mutates_args=())
def native_batch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Allocate outputs for one native call; graph caches use the out variant."""
    output = torch.empty_like(q)
    scratch = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    if mask.ndim == 3 and mask.shape[:2] == q.shape[:2]:
        padded_mask = torch.full(
            (mask.shape[0], 64, mask.shape[2]),
            float("-inf"),
            dtype=mask.dtype,
            device=mask.device,
        )
        padded_mask[:, :1].copy_(mask)
        mask = padded_mask
    _launch_native_batch_attention(q, k, v, mask, output, scratch, scale)
    return output


@native_batch_attention.register_fake
def _fake(q, k, v, mask, scale):
    return torch.empty_like(q)


@torch.library.custom_op(
    "gguf_current::attention_batch_out", mutates_args=("output", "scratch")
)
def native_batch_attention_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    scale: float,
) -> None:
    """Use caller-owned output and scratch, shared serially across graph keys."""
    _launch_native_batch_attention(q, k, v, mask, output, scratch, scale)


@native_batch_attention_out.register_fake
def _fake_out(q, k, v, mask, output, scratch, scale):
    return None


def _prefill_mask_rows(queries: int) -> int:
    if queries <= 1:
        raise ValueError("Batched prefill requires more than one query per request")
    return (queries + 63) // 64 * 64


@lru_cache(maxsize=1)
def _prefill_library():
    library = _library()
    library.current_attention_batch_prefill_run.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_size_t]
        + [ctypes.c_int] * 5
        + [ctypes.c_float, ctypes.c_size_t]
    )
    library.current_attention_batch_prefill_run.restype = ctypes.c_int
    return library


def _validate_prefill_shapes(q, k, v, mask):
    """Shape-only checks also usable on CPU, before loading the CUDA library."""
    if (
        q.ndim != 4
        or not 2 <= q.shape[0] <= 65535
        or q.shape[1] <= 1
        or q.shape[2:] != (8, 256)
        or k.ndim != 4
        or k.shape[0] != q.shape[0]
        or k.shape[2:] != (2, 256)
        or k.shape[1] % 256
        or v.shape != k.shape
        or mask.ndim != 3
        or mask.shape[0] != q.shape[0]
        or mask.shape[1] not in (q.shape[1], _prefill_mask_rows(q.shape[1]))
        or mask.shape[2] < q.shape[1]
        or mask.shape[2] > k.shape[1]
        or mask.shape[2] % 256
    ):
        raise ValueError("Expected uniform prefill Q [B,Q,8,256] and padded KV/mask")


@torch.library.custom_op("gguf_current::attention_batch_prefill", mutates_args=())
def native_batch_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One B-stream native FA call for B>1 and equal query counts Q>1.

    KV capacity may exceed the common logical mask width. An unpadded mask
    [B,Q,K] is accepted; explicitly padded masks must have -inf guard rows.
    B1 and one-token decode deliberately use their existing operators.
    """
    _validate_prefill_shapes(q, k, v, mask)
    if q.dtype != torch.float32 or any(t.dtype != torch.float16 for t in (k, v, mask)):
        raise ValueError("Expected F32 queries and F16 KV/mask")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != q.device
        for t in (q, k, v, mask)
    ):
        raise ValueError("Expected contiguous CUDA tensors on the same device")
    rows = _prefill_mask_rows(q.shape[1])
    if mask.shape[1] != rows:
        padded_mask = torch.full(
            (q.shape[0], rows, mask.shape[2]),
            float("-inf"),
            dtype=mask.dtype,
            device=mask.device,
        )
        padded_mask[:, : q.shape[1]].copy_(mask)
        mask = padded_mask
    output = torch.empty_like(q)
    scratch = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    library = _prefill_library()
    with torch.cuda.device(q.device):
        status = library.current_attention_batch_prefill_run(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            mask.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            scratch.numel(),
            q.device.index,
            q.shape[0],
            q.shape[1],
            k.shape[1],
            mask.shape[2],
            float(scale),
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_attention_error().decode())
    return output


@native_batch_prefill_attention.register_fake
def _fake_prefill(q, k, v, mask, scale):
    return torch.empty_like(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())
