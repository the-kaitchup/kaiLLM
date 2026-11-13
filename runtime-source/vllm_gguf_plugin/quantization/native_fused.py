# SPDX-License-Identifier: Apache-2.0
"""Fused GGML decode projections for 1..8 activation rows.

One activation preparation per projection group (exact power-of-two row
normalization + GGML Q8_1, optionally a second int8 pass over the residual) is
shared by every matrix of a merged projection; each matrix then runs one MMVQ
launch that writes BF16/F32 straight into its slice of the merged output. This
replaces the per-matrix input cast, guard, quantization, repair and output cast
kernels and the concatenation of merged projections.

Numerics: single pass equals GGML/llama.cpp MMVQ arithmetic on the normalized
row (the normalization is an exact binary scaling, so for every row the old
guard admitted the result is bit-identical); the two-pass mode adds a residual
int8 pass (~16-bit activations). Works for every GGML format with a Q8_1
vec_dot and for any CUDA architecture the native library was built for.

Batches above 8 rows (multi-sequence decode, prefill, prompt scoring) use the
fused MMQ group: the same exact normalization, GGML's MMQ activation quantization
(bit-identical on the normalized row), GGML MMQ per matrix into a merged F32 tile
and one scale/cast epilogue (KAILLM_FUSED_MMQ=0 disables it).

KAILLM_FUSED_MMVQ: "0" disabled, "1" single pass (default), "2" two passes.
"""

import ctypes
import os
from functools import lru_cache

import torch

ENV = "KAILLM_FUSED_MMVQ"
MAX_ROWS = 8
TYPES = (2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29)
MMQ_TYPES = (2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23)


def mmq_mode():
    value = os.getenv("KAILLM_FUSED_MMQ", "1")
    if value not in ("0", "1"):
        raise ValueError("KAILLM_FUSED_MMQ must be 0 or 1")
    return value == "1"
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def mode():
    value = os.getenv(ENV, "1")
    if value not in ("0", "1", "2"):
        raise ValueError(ENV + " must be 0, 1 or 2")
    return int(value)


@lru_cache(maxsize=4)
def _lib(path):
    lib = ctypes.CDLL(path)
    if not hasattr(lib, "kai_fused_mmvq"):
        return None
    i64, ptr = ctypes.c_int64, ctypes.c_void_p
    lib.current_direct_error.restype = ctypes.c_char_p
    lib.kai_fused_supported.argtypes = [ctypes.c_int, i64, i64, i64]
    lib.kai_fused_supported.restype = ctypes.c_int
    lib.kai_fused_activation_bytes.argtypes = [i64, i64, ctypes.c_int]
    lib.kai_fused_activation_bytes.restype = ctypes.c_size_t
    lib.kai_fused_prepare.argtypes = [ptr, ctypes.c_int, i64, i64, i64, ptr, ctypes.c_int, ctypes.c_uint64]
    lib.kai_fused_prepare.restype = ctypes.c_int
    lib.kai_fused_mmvq.argtypes = [ptr, ctypes.c_int, i64, i64, i64, ptr, ctypes.c_int, ptr,
                                   ctypes.c_int, i64, ctypes.c_uint64]
    lib.kai_fused_mmvq.restype = ctypes.c_int
    if hasattr(lib, "kai_fused_mmq_run"):
        arr = ctypes.POINTER
        lib.kai_fused_mmq_workspace.argtypes = [ctypes.c_int, arr(ctypes.c_int), arr(i64), ctypes.c_int, i64, i64,
                                                ctypes.c_int]
        lib.kai_fused_mmq_workspace.restype = ctypes.c_size_t
        lib.kai_fused_mmq_run.argtypes = [arr(ptr), arr(ctypes.c_int), arr(i64), ctypes.c_int, ptr, ctypes.c_int,
                                          i64, i64, i64, ptr, ctypes.c_int, i64, ptr, ctypes.c_size_t, ctypes.c_int,
                                          ctypes.c_uint64]
        lib.kai_fused_mmq_run.restype = ctypes.c_int
    if hasattr(lib, "kai_fused_norm_prepare"):
        lib.kai_fused_prepare_view.argtypes = [ptr, ctypes.c_int, i64, i64, i64, ptr, i64, i64, i64, ctypes.c_uint64]
        lib.kai_fused_prepare_view.restype = ctypes.c_int
        lib.kai_fused_prepare_swiglu.argtypes = [ptr, ctypes.c_int, i64, i64, i64, ptr, ctypes.c_uint64]
        lib.kai_fused_prepare_swiglu.restype = ctypes.c_int
        lib.kai_fused_norm_prepare.argtypes = [ptr, ctypes.c_int, i64, ptr, ctypes.c_int, i64, ptr, ptr,
                                               ctypes.c_float, ptr, i64, i64, ptr, ctypes.c_uint64]
        lib.kai_fused_norm_prepare.restype = ctypes.c_int
    return lib


_LIBRARY_ENVS = (
    "VLLM_GGUF_NATIVE_BODY_LIBRARY",   # SM86 profiles
    "VLLM_GGUF_NATIVE_SM120_LIBRARY",  # Blackwell profiles
    "VLLM_GGUF_NATIVE_HEAD_LIBRARY",
)


def library():
    """First configured native GGML bridge that exports the fused entry points."""
    for env in _LIBRARY_ENVS:
        path = os.getenv(env)
        if path and os.path.isfile(path):
            lib = _lib(path)
            if lib is not None:
                return lib
    return None


def _check(lib, status):
    if status:
        raise RuntimeError(lib.current_direct_error().decode())


# Activation producers: a norm stashes the activation of its output for its registered consumer
# projections (first weight pointer -> producer id), so they skip their own preparation.
CONSUMER_OF = {}
STASH = {}


def ext_available():
    lib = library()
    return lib is not None and hasattr(lib, "kai_fused_norm_prepare") and mode() == 1


def _key(x):
    # Inference tensors have no version counter; the producer clears its entry whenever it does not stash.
    return (x.data_ptr(), x.shape[0], x.shape[1], x.stride(0), x.dtype, x.device)


def _mmvq_into(lib, activation, weights, types, rows_out, m, k, out_f32, device):
    total = sum(rows_out)
    out = torch.empty((m, total), dtype=torch.float32 if out_f32 else torch.bfloat16, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream
    offset = 0
    element = out.element_size()
    for weight, qtype, n in zip(weights, types, rows_out):
        _check(lib, lib.kai_fused_mmvq(weight.data_ptr(), qtype, n, k, m, activation.data_ptr(), 0,
                                       out.data_ptr() + offset * element, 0 if out_f32 else 1, total, stream))
        offset += n
    return out


def _new_activation(lib, m, k, device):
    return torch.empty(lib.kai_fused_activation_bytes(m, k, 0), dtype=torch.uint8, device=device)


def norm_prepare(producer, x, residual, weight32, eps):
    """Fused Gemma norm for <= 8 rows: (y, F32 residual_out or None); y's activation is stashed."""
    if not ext_available() or x.dim() != 2 or not 0 < x.shape[0] <= MAX_ROWS or not x.is_cuda:
        return None
    if x.dtype not in (torch.bfloat16, torch.float32) or x.stride(-1) != 1:
        return None
    if residual is not None and (residual.dtype not in (torch.bfloat16, torch.float32)
                                 or residual.shape != x.shape or residual.stride(-1) != 1):
        return None
    lib = library()
    m, k = x.shape
    if k % 32:
        return None
    y = torch.empty((m, k), dtype=torch.bfloat16, device=x.device)
    rout = None if residual is None else torch.empty((m, k), dtype=torch.float32, device=x.device)
    activation = _new_activation(lib, m, k, x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    _check(lib, lib.kai_fused_norm_prepare(
        x.data_ptr(), _DTYPE_CODES[x.dtype], x.stride(0),
        None if residual is None else residual.data_ptr(), 0 if residual is None else _DTYPE_CODES[residual.dtype],
        0 if residual is None else residual.stride(0), None if rout is None else rout.data_ptr(),
        weight32.data_ptr(), float(eps), y.data_ptr(), m, k, activation.data_ptr(), stream))
    STASH[producer] = (_key(y), activation)
    return y, rout


@torch.library.custom_op("kaillm::fused_mmvq_view", mutates_args=())
def fused_mmvq_view(x: torch.Tensor, weights: list[torch.Tensor], types: list[int], rows_out: list[int],
                    a: int, b: int, s: int, out_f32: bool) -> torch.Tensor:
    lib = library()
    m, k = x.shape
    activation = _new_activation(lib, m, k, x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    _check(lib, lib.kai_fused_prepare_view(x.data_ptr(), _DTYPE_CODES[x.dtype], m, k, x.stride(0),
                                           activation.data_ptr(), a, b, s, stream))
    return _mmvq_into(lib, activation, weights, types, rows_out, m, k, out_f32, x.device)


@fused_mmvq_view.register_fake
def _fake_view(x, weights, types, rows_out, a, b, s, out_f32):
    return x.new_empty((x.shape[0], sum(rows_out)), dtype=torch.float32 if out_f32 else torch.bfloat16)


@torch.library.custom_op("kaillm::fused_mmvq_swiglu", mutates_args=())
def fused_mmvq_swiglu(gate_up: torch.Tensor, weights: list[torch.Tensor], types: list[int], rows_out: list[int],
                      out_f32: bool) -> torch.Tensor:
    lib = library()
    m, k = gate_up.shape[0], gate_up.shape[1] // 2
    activation = _new_activation(lib, m, k, gate_up.device)
    stream = torch.cuda.current_stream(gate_up.device).cuda_stream
    _check(lib, lib.kai_fused_prepare_swiglu(gate_up.data_ptr(), _DTYPE_CODES[gate_up.dtype], m, k,
                                             gate_up.stride(0), activation.data_ptr(), stream))
    return _mmvq_into(lib, activation, weights, types, rows_out, m, k, out_f32, gate_up.device)


@fused_mmvq_swiglu.register_fake
def _fake_swiglu(gate_up, weights, types, rows_out, out_f32):
    return gate_up.new_empty((gate_up.shape[0], sum(rows_out)), dtype=torch.float32 if out_f32 else torch.bfloat16)


def _run(x, weights, types, rows_out, two_pass, out_f32):
    lib = library()
    if two_pass == 0 and CONSUMER_OF:
        producer = CONSUMER_OF.get(weights[0].data_ptr())
        entry = STASH.get(producer) if producer is not None else None
        if entry is not None and entry[0] == _key(x):
            return _mmvq_into(lib, entry[1], weights, types, rows_out, x.shape[0], x.shape[1], out_f32, x.device)
    k = x.shape[1]
    m = x.shape[0]
    total = sum(rows_out)
    out = torch.empty((m, total), dtype=torch.float32 if out_f32 else torch.bfloat16, device=x.device)
    activation = torch.empty(lib.kai_fused_activation_bytes(m, k, two_pass), dtype=torch.uint8,
                             device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    _check(lib, lib.kai_fused_prepare(x.data_ptr(), _DTYPE_CODES[x.dtype], m, k, x.stride(0),
                                      activation.data_ptr(), two_pass, stream))
    offset = 0
    element = out.element_size()
    for weight, qtype, n in zip(weights, types, rows_out):
        _check(lib, lib.kai_fused_mmvq(weight.data_ptr(), qtype, n, k, m, activation.data_ptr(), two_pass,
                                       out.data_ptr() + offset * element, 0 if out_f32 else 1, total,
                                       stream))
        offset += n
    return out


_MMQ_GROUPS = {}


def _mmq_run(x, weights, types, rows_out, out_f32):
    lib = library()
    m, k = x.shape
    key = (tuple(w.data_ptr() for w in weights), tuple(types), tuple(rows_out))
    group = _MMQ_GROUPS.get(key)
    if group is None:
        count = len(weights)
        group = ((ctypes.c_void_p * count)(*[w.data_ptr() for w in weights]),
                 (ctypes.c_int * count)(*types), (ctypes.c_int64 * count)(*rows_out))
        _MMQ_GROUPS[key] = group
    ptrs, ctypes_types, ctypes_rows = group
    device = x.device.index
    size = lib.kai_fused_mmq_workspace(device, ctypes_types, ctypes_rows, len(weights), k, m, int(out_f32))
    if not size:
        raise RuntimeError(lib.current_direct_error().decode())
    total = sum(rows_out)
    out = torch.empty((m, total), dtype=torch.float32 if out_f32 else torch.bfloat16, device=x.device)
    scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    # out_f32: the MMQ tile is the output itself (stride 0 = direct); BF16: merged tile + cast.
    _check(lib, lib.kai_fused_mmq_run(ptrs, ctypes_types, ctypes_rows, len(weights), x.data_ptr(),
                                      _DTYPE_CODES[x.dtype], x.stride(0), k, m, out.data_ptr(),
                                      0 if out_f32 else 1, 0 if out_f32 else total, scratch.data_ptr(), size,
                                      device, stream))
    return out


@torch.library.custom_op("kaillm::fused_mmq", mutates_args=())
def fused_mmq(x: torch.Tensor, weights: list[torch.Tensor], types: list[int], rows_out: list[int],
              out_f32: bool) -> torch.Tensor:
    return _mmq_run(x, weights, types, rows_out, out_f32)


@fused_mmq.register_fake
def _fake_mmq(x, weights, types, rows_out, out_f32):
    return x.new_empty((x.shape[0], sum(rows_out)), dtype=torch.float32 if out_f32 else torch.bfloat16)


@torch.library.custom_op("kaillm::fused_mmvq", mutates_args=())
def fused_mmvq(x: torch.Tensor, weights: list[torch.Tensor], types: list[int], rows_out: list[int],
               two_pass: int, out_f32: bool) -> torch.Tensor:
    return _run(x, weights, types, rows_out, two_pass, out_f32)


@fused_mmvq.register_fake
def _fake(x, weights, types, rows_out, two_pass, out_f32):
    return x.new_empty((x.shape[0], sum(rows_out)), dtype=torch.float32 if out_f32 else torch.bfloat16)


class FusedProjection:
    """A merged projection (one or more original GGUF matrices sharing an input)."""

    def __init__(self, parts):
        self.weights = [p.weight for p in parts]
        self.types = [int(p.weight_type) for p in parts]
        self.rows_out = [int(p.shape[0]) for p in parts]
        self.k = int(parts[0].shape[1])
        self.dtype = parts[0].dtype
        self.total = sum(self.rows_out)
        lib = library()
        self._supported = tuple(
            rows for rows in range(1, MAX_ROWS + 1)
            if all(lib.kai_fused_supported(t, n, self.k, rows) for t, n in zip(self.types, self.rows_out))
        )
        self._mmq = (
            mmq_mode()
            and hasattr(lib, "kai_fused_mmq_run")
            and all(t in MMQ_TYPES for t in self.types)
            and self.total % 4 == 0
            and self.k % 4 == 0
        )
        if self._mmq:
            device = parts[0].weight.device.index
            count = len(parts)
            types = (ctypes.c_int * count)(*self.types)
            rows = (ctypes.c_int64 * count)(*self.rows_out)
            self._mmq = all(
                lib.kai_fused_mmq_workspace(device, types, rows, count, self.k, m, 0) > 0 for m in (16, 512)
            )

    def supports(self, x):
        rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
        if not (x.dtype in _DTYPE_CODES and x.is_cuda):
            return False
        return rows in self._supported or (self._mmq and rows > MAX_ROWS)

    def _finish(self, out, lead, output_dtype):
        if output_dtype not in (torch.float32, torch.bfloat16):
            out = out.to(output_dtype)
        return out.reshape(*lead, self.total)

    def call_view(self, x, view, output_dtype=None):
        output_dtype = self.dtype if output_dtype is None else output_dtype
        x2 = x.reshape(-1, self.k).contiguous()
        out = fused_mmvq_view(x2, self.weights, self.types, self.rows_out, int(view[0]), int(view[1]),
                              int(view[2]), output_dtype == torch.float32)
        return self._finish(out, x.shape[:-1], output_dtype)

    def call_swiglu(self, gate_up, output_dtype=None):
        output_dtype = self.dtype if output_dtype is None else output_dtype
        g2 = gate_up.reshape(-1, 2 * self.k).contiguous()
        out = fused_mmvq_swiglu(g2, self.weights, self.types, self.rows_out, output_dtype == torch.float32)
        return self._finish(out, gate_up.shape[:-1], output_dtype)

    def __call__(self, x, output_dtype=None):
        output_dtype = self.dtype if output_dtype is None else output_dtype
        x2 = x.reshape(-1, self.k).contiguous()
        if x2.shape[0] > MAX_ROWS:
            out = fused_mmq(x2, self.weights, self.types, self.rows_out, output_dtype == torch.float32)
        else:
            out = fused_mmvq(x2, self.weights, self.types, self.rows_out, mode() - 1,
                             output_dtype == torch.float32)
        if output_dtype not in (torch.float32, torch.bfloat16):
            out = out.to(output_dtype)
        return out.reshape(*x.shape[:-1], self.total)


def prepare(parts):
    """FusedProjection for compact GGUF parts, or None when any part is ineligible."""
    if mode() == 0 or not parts or library() is None:
        return None
    k = parts[0].shape[1]
    for part in parts:
        if (
            not part.is_quantized
            or int(part.weight_type) not in TYPES
            or part.shape[1] != k
            or not part.weight.is_cuda
            or part.weight.dtype != torch.uint8
            or not part.weight.is_contiguous()
            or part.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return None
    fused = FusedProjection(parts)
    return fused if (fused._supported or fused._mmq) else None
