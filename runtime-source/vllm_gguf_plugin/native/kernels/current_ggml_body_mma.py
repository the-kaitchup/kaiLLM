# SPDX-License-Identifier: Apache-2.0
"""Opt-in body MMA for physical-M8 decode, with the native prefill guard.

Original matrices and FP32 outputs are preserved; integer tensor cores change
floating multiplication/reduction ordering. Only Q6_K (N,K)=(6144,1024) or
(1024,3584) at exactly M8 can use MMA; all other shapes/formats and already-
native-MMQ calls retain native arithmetic. The broader Q6/Q8 M2..8 experiment
was narrowed after timings showed regressions elsewhere. Whole-model quality
validation remains required; this is not a default serving implementation.

GGUF_PARITY_AFFINE_MMA=1 selects a separate experimental library adding
Q4_K/Q5_K FP32 affine arithmetic with the original native Q8 activation codes.
Original packed weights, FP16 block factors, and integer group coefficients
are retained; no expanded weight scales are rounded to FP16.

Build: PYTHONPATH=benchmarks python -m kernels.current_ggml_body_mma --build
"""

import argparse
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import replace
from functools import lru_cache, wraps
from pathlib import Path

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths
from .current_ggml_direct import (
    _FORMATS,
    CurrentGGMLSharedLinear,
    current_ggml_shared,
)


def body_mma_shape_policy(
    qtype: int, n: int, k: int, m: int, *, affine: bool | None = None
) -> bool:
    """Measured policy, before the device's native MMVQ/MMQ dispatch check."""
    if affine is None:
        affine = os.environ.get("GGUF_PARITY_AFFINE_MMA") == "1"
    if affine and m == 8 and qtype in (12, 13) and n >= 128:
        return n % 128 == 0 and k > 0 and k % 256 == 0
    return m == 8 and qtype == 14 and (n, k) in ((6144, 1024), (1024, 3584))


def metadata_allows_decode(metadata):
    """Host-only batch policy; None means no metadata during graph profiling."""
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        return False
    recurrent = [m for m in metadata.values() if hasattr(m, "num_spec_decodes")]
    attention = [m for m in metadata.values() if hasattr(m, "native_query_starts")]
    if not recurrent or not attention:
        return False
    expected_count = recurrent[0].num_decodes
    for m in recurrent:
        count = m.num_decodes
        if not (
            1 <= count <= 8
            and count == expected_count
            and m.num_prefills == m.num_prefill_tokens == 0
            and m.num_spec_decodes == m.num_spec_decode_tokens == 0
            and m.num_decode_tokens == m.num_actual_tokens == count
        ):
            return False
    for m in attention:
        starts, lengths = m.native_query_starts, m.native_seq_lens
        if not (
            len(starts) == len(lengths) + 1
            and 1 <= len(lengths) <= 8
            and starts[0] == 0
            and all(b - a == 1 for a, b in zip(starts, starts[1:]))
            and all(length > 1 for length in lengths)
            and starts[-1] == m.num_actual_tokens == expected_count
        ):
            return False
    return True


def _runtime_allows_decode(m):
    if m != 8:
        return False
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    context = get_forward_context()
    # CUDA graphs freeze this host decision. MRV2's M8 dummy capture describes
    # eight fresh length-one requests, even though the protected graph is only
    # replayed for actual decode. Override synthetic/absent capture metadata.
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if (
        getattr(CudaGraphManager, "_gguf_body_decode_guard", False)
        and context.cudagraph_runtime_mode.name == "PIECEWISE"
        and torch.cuda.is_current_stream_capturing()
    ):
        return True
    return metadata_allows_decode(context.attn_metadata) is True


def restrict_body_replay(
    descriptor,
    *,
    num_reqs,
    num_tokens,
    uniform_token_count,
    num_active_loras=0,
    num_ubatches=1,
):
    """Never replay an M8 decode graph for prefill; allow padded5–7 decode."""
    if descriptor.num_tokens != 8 or descriptor.cg_mode.name == "NONE":
        return descriptor
    if (
        1 <= num_reqs <= 8
        and num_tokens == num_reqs
        and uniform_token_count == 1
        and not num_active_loras
        and num_ubatches == 1
    ):
        return descriptor
    return replace(
        descriptor,
        cg_mode=type(descriptor.cg_mode).NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform_token_count=uniform_token_count,
        max_query_len=None,
    )


def install_body_decode_guard():
    """Protect the host policy from being frozen into a prefill graph replay."""
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if getattr(CudaGraphManager, "_gguf_body_decode_guard", False):
        return
    original = CudaGraphManager.dispatch

    @wraps(original)
    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
        num_ubatches=1,
    ):
        result = original(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len,
            num_ubatches,
        )
        additional = self.vllm_config.additional_config
        if isinstance(additional, dict) and additional.get("gguf_parity_body_mma"):
            result = restrict_body_replay(
                result,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_token_count,
                num_active_loras=num_active_loras,
                num_ubatches=num_ubatches,
            )
        return result

    CudaGraphManager.dispatch = dispatch
    CudaGraphManager._gguf_body_decode_guard = True


def _library_path():
    default = (
        str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_ggml_affine_mma.so")
        if os.environ.get("GGUF_PARITY_AFFINE_MMA") == "1"
        else str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_ggml_body_mma.so")
    )
    return Path(
        os.environ.get(
            "GGUF_CURRENT_GGML_BODY_MMA",
            default,
        )
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
    if os.environ.get("GGUF_PARITY_AFFINE_MMA") == "1":
        command.append("-DGGUF_AFFINE_MMA=1")
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
    sources = [
        Path(__file__).with_suffix(".cu"),
        Path(__file__).with_name("current_ggml_direct.cu"),
    ]
    if os.environ.get("GGUF_PARITY_AFFINE_MMA") == "1":
        sources.append(Path(__file__).with_name("q45_affine_mma.cuh"))
    library.with_suffix(".build.json").write_text(
        json.dumps(
            {
                "command": command,
                "affine_mma": os.environ.get("GGUF_PARITY_AFFINE_MMA") == "1",
                "source_sha256": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    return library


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(_library_path()))
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.current_direct_prepare.argtypes = [ctypes.c_int]
    lib.current_direct_prepare.restype = ctypes.c_int
    lib.current_body_mma_enabled.argtypes = [ctypes.c_int, ctypes.c_int] + [
        ctypes.c_int64
    ] * 3
    lib.current_body_mma_enabled.restype = ctypes.c_int
    lib.current_body_shape_policy.argtypes = [ctypes.c_int] + [ctypes.c_int64] * 3
    lib.current_body_shape_policy.restype = ctypes.c_int
    lib.current_body_workspace.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.current_body_workspace.restype = ctypes.c_size_t
    lib.current_body_run.argtypes = [
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
    lib.current_body_run.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_current::body_mma_linear", mutates_args=())
def current_body_mma(
    x: torch.Tensor,
    packed: list[torch.Tensor],
    qtypes: list[int],
    k: int,
    decode_only: bool = False,
) -> torch.Tensor:
    if decode_only and not _runtime_allows_decode(x.shape[0]):
        return current_ggml_shared(x, packed, qtypes, k)
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] != k
        or not 1 <= len(packed) <= 32
        or len(qtypes) != len(packed)
    ):
        raise ValueError("Expected CUDA FP32 input and original matrix/type pairs")
    for weight, qtype in zip(packed, qtypes):
        if (
            qtype not in _FORMATS
            or weight.ndim != 2
            or weight.dtype != torch.uint8
            or weight.device != x.device
            or not weight.is_contiguous()
            or weight.shape[0] <= 0
        ):
            raise ValueError("Expected packed Q8_0/Q4_K/Q5_K/Q6_K matrices")
        block, size = _FORMATS[qtype]
        if k <= 0 or k % block or weight.shape[1] != k // block * size:
            raise ValueError("Packed matrix does not match activation width")
        tail = (((-k) % 512) + ((-weight.shape[0]) % 4) * k) // block * size
        available = weight.untyped_storage().nbytes() - weight.storage_offset()
        if available < weight.numel() + tail or weight.data_ptr() % 128:
            raise ValueError("Each original matrix needs aligned GGML tail padding")
    rows = [weight.shape[0] for weight in packed]
    y = torch.empty((x.shape[0], sum(rows)), dtype=torch.float32, device=x.device)
    if not x.shape[0]:
        return y
    count = len(packed)
    types_c = (ctypes.c_int * count)(*qtypes)
    rows_c = (ctypes.c_int64 * count)(*rows)
    pointers = (ctypes.c_void_p * count)(*(weight.data_ptr() for weight in packed))
    lib = _library()
    with torch.cuda.device(x.device):
        size = lib.current_body_workspace(
            x.device.index, types_c, rows_c, count, k, x.shape[0]
        )
        if not size:
            raise RuntimeError(lib.current_direct_error().decode())
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        if lib.current_body_run(
            pointers,
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
        ):
            raise RuntimeError(lib.current_direct_error().decode())
    return y


@current_body_mma.register_fake
def _fake(x, packed, qtypes, k, decode_only=False):
    return torch.empty(
        (x.shape[0], sum(weight.shape[0] for weight in packed)),
        device=x.device,
        dtype=torch.float32,
    )


class CurrentGGMLBodyMMALinear(CurrentGGMLSharedLinear):
    def __init__(self, parts, k, *, device=None, decode_only=False):
        super().__init__(parts, k, device=device)
        self.decode_only = decode_only
        with torch.cuda.device(self._device):
            lib = _library()
            if lib.current_direct_prepare(self._device.index):
                raise RuntimeError(lib.current_direct_error().decode())

    def mma_enabled(self, m):
        lib = _library()
        flags = [
            lib.current_body_mma_enabled(
                self._device.index, part.qtype, part.shape[0], self.k, m
            )
            for part in self.parts
        ]
        if any(flag < 0 for flag in flags):
            raise RuntimeError(lib.current_direct_error().decode())
        return [bool(flag) for flag in flags]

    def forward(self, x, bias=None):
        if (
            x.dtype != torch.float32
            or x.device != self._device
            or x.shape[-1] != self.k
        ):
            raise ValueError(
                "Body MMA requires FP32 activations with matching device/K"
            )
        packed = [
            part.packed_storage[: part.shape[0] * part.row_bytes].view(
                part.shape[0], part.row_bytes
            )
            for part in self.parts
        ]
        y = current_body_mma(
            x.reshape(-1, self.k).contiguous(),
            packed,
            [part.qtype for part in self.parts],
            self.k,
            self.decode_only,
        ).reshape(*x.shape[:-1], self.shape[0])
        return y if bias is None else y + bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())
