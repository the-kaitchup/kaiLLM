# SPDX-License-Identifier: Apache-2.0
"""ITQ (integer trellis) packed linear: yaqa.manifest codec "itq", version 1.

Decode (1..16 rows): rotate + int8-quantize the input once, integer-dp4a trellis GEMV, then the
output rotation, signs, scales and bias in one epilogue.  Prefill (> 16 rows): rotate the input to
BF16, decode the rotated weight transiently to BF16 and use cuBLAS; no persistent dense cache.
"""

from __future__ import annotations

import ctypes
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from ..native.cuda_architecture import bind_library_architecture, device_architecture
from .yaqa_cuda import build_cuda_library

SOURCE = Path(__file__).parents[1] / "native/kernels/itq_kernels.cu"
BUILD_CONTRACT = "sm86-itq-v1-dp4a-gemv-prep-post-decode-vocab-bps12-r6-addnorm-v2b8"
_LIBRARIES: dict = {}
MAX_DECODE_ROWS = 16          # dp4a decode kernels (and the vocabulary head's classes)
MAX_MMA_ROWS = 64             # tensor-core decode GEMVs (sm_80+, ITQ_MMA != 0): MTP verification and larger batches


@lru_cache(maxsize=8)
def max_gemv_rows(device_index: int) -> int:
    major = torch.cuda.get_device_capability(device_index)[0]
    return MAX_MMA_ROWS if major >= 8 and os.getenv("ITQ_MMA", "1") != "0" else MAX_DECODE_ROWS


@lru_cache(maxsize=1)
def _small_mma() -> bool:
    """3-4 rows run the 8-row tensor-core GEMV (mirrors use_small_mma() in itq_kernels.cu: same env, same device)."""
    return (os.getenv("ITQ_SMALL_MMA", "1") != "0" and os.getenv("ITQ_MMA", "1") != "0"
            and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)


def pad_rows(rows: int) -> int:
    """Activation rows of the decode buffers (the kernels' NB): 1, 2, 4 (8 with the small-row tensor-core GEMV), 8,
    16, 32, 48 or 64.  Must match pad_rows() in itq_kernels.cu."""
    if rows <= 2:
        return 1 if rows <= 1 else 2
    if rows <= 4:
        return 8 if _small_mma() else 4
    for nb in (8, 16, 32, 48):
        if rows <= nb:
            return nb
    return 64
# Prefill (> 16 rows): "int8" = per-token int8 activations after the input rotation, the weight decoded to
# its raw int8 table values (bounded per-matrix workspace), int8 tensor-core GEMM, scales applied after;
# "bf16" = rotated BF16 activations x weight decoded to BF16, cuBLAS.
# "auto" (default): int8 on sm_8x, bf16 elsewhere.  GeForce Ampere runs BF16 tensor-core GEMMs with FP32 accumulation at
# half rate but int8 at full rate: RTX 3060, Qwen3.8 27B 7.0 GB, 8000-token prompt in 256-token chunks, time to first
# token 28.8 -> 18.9 s (EAR-equivalent path, validated on the RTX PRO 6000, where int8 was 8 % slower than bf16).
PREFILL_MODE = os.getenv("ITQ_PREFILL", "auto")
if PREFILL_MODE not in ("int8", "bf16", "auto"):
    raise ValueError("ITQ_PREFILL must be int8, bf16 or auto")
_PREFILL_BY_DEVICE: dict = {}


def prefill_mode(device) -> str:
    if PREFILL_MODE != "auto":
        return PREFILL_MODE
    key = str(device)
    if key not in _PREFILL_BY_DEVICE:
        _PREFILL_BY_DEVICE[key] = "int8" if torch.cuda.get_device_capability(device)[0] < 9 else "bf16"
    return _PREFILL_BY_DEVICE[key]


def block_size(kind: str, width: int) -> int:
    if kind == "fwht":
        return width
    if kind.startswith("block"):
        size = int(kind[5:])
        if width % size or size & (size - 1):
            raise ValueError(f"ITQ rotation {kind} does not divide width {width}")
        return size
    raise ValueError(f"Unsupported ITQ rotation {kind!r}")


@lru_cache(maxsize=4)
def _build(architecture=None):
    return build_cuda_library(SOURCE, BUILD_CONTRACT, "itq_kernels", architecture=architecture)


def library(device):
    architecture = device_architecture(torch.cuda.get_device_capability(device))
    if architecture is None:
        raise RuntimeError("ITQ kernels need a supported CUDA architecture")
    identity = _build(architecture)
    key = json.dumps(identity, sort_keys=True)
    bind_library_architecture(key, architecture)
    if key not in _LIBRARIES:
        lib = ctypes.CDLL(identity["library"])
        p, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        lib.itq_gemv.argtypes = [p, p, p, p, p, i, i, i, i, i, i, f, i, i, p]
        lib.itq_prep.argtypes = [p, i, p, p, p, p, p, i, i, i, i, i, p, i, p]
        lib.itq_post.argtypes = [p, p, p, p, p, i, i, i, i, i, p]
        lib.itq_decode.argtypes = [p, p, p, i, i, i, i, f, p]
        lib.itq_decode_i8.argtypes = [p, p, p, i, i, i, p]
        lib.itq_decode_i8.restype = ctypes.c_int
        lib.itq_quant_rows.argtypes = [p, p, p, i, i, p]
        lib.itq_quant_rows.restype = ctypes.c_int
        lib.itq_post_i32.argtypes = [p, p, p, p, p, p, i, i, i, i, i, p]
        lib.itq_post_i32.restype = ctypes.c_int
        lib.itq_gemv_group.argtypes = [i, p, p, p, p, p, p, p, p, i, i, i, i, i, p]
        lib.itq_post_group.argtypes = [i, p, p, p, p, p, p, p, i, i, p]
        lib.itq_swiglu_prep.argtypes = [p, p, i, p, p, p, p, p, p, p, p, i, i, i, p, i, p]
        lib.itq_add_norm_prep.argtypes = [p, i, p, p, p, i, i, p, p, f, p, i, i, i, p, p, p, p, p, i, p]
        lib.itq_gated_norm_prep.argtypes = [p, i, p, i, p, f, i, p, p, p, p, i, i, i, p, i, p]
        lib.itq_gemv_warps.argtypes = [i, i, i]
        lib.itq_gemv_warps.restype = ctypes.c_int
        for fn in (lib.itq_gemv, lib.itq_prep, lib.itq_post, lib.itq_decode, lib.itq_gemv_group,
                   lib.itq_post_group, lib.itq_swiglu_prep, lib.itq_add_norm_prep, lib.itq_gated_norm_prep):
            fn.restype = ctypes.c_int
        _LIBRARIES[key] = lib
    return _LIBRARIES[key]


def _check(code: int, what: str):
    if code:
        raise RuntimeError(f"ITQ {what} launch failed with CUDA error {code}")


def _ptr(t):
    return None if t is None else t.data_ptr()


def launch_params(rows: int, bps: int, n: int, major: int, group: bool = False):
    """(tile pairs in flight per half-warp, blocks per SM) of a decode GEMV, from cold-L2 benchmarks on Qwen3.8 27B
    shapes (bench_itq_cold.py).  RTX 3060 (sm_86): one row -> 1 pair, 12 blocks/SM (gate+up 2 bits 323 -> 240 us);
    more rows -> 4 pairs for long rows (n >= 8192, down_proj) else 2, 12 blocks/SM.  RTX PRO 6000 (sm_120): one row ->
    4 pairs at <= 8 bits per step else 2; more rows -> 4 pairs for long rows else 2; 4 blocks/SM.
    bps: the kernel format code (PackedITQLinear.bps); a V2 format (32 + b) is tuned as the V4 BPS of its rate (2 b)."""
    if bps >= 32:
        bps = 2 * (bps - 32)
    long_rows = n >= 8192
    if rows > 32:      # 33-64 rows: 1 pair, so the activations of the round are hoisted (PF x CH <= 8); RTX 3060 cold
        return 1, (12 if major == 8 else 8)    # L2, 48 rows: down 2 bits 525 -> 311 us, gate+up 813 -> 557
    if major == 8:
        return (1 if rows == 1 else 4 if long_rows else 2), 12
    if group and rows <= 2 and bps >= 8 and pad_rows(rows) < 8:   # 8-warp grouped dp4a GEMV (itq_kernels.cu
        return 4, 4                                                 # group_warps): 4 pairs, 4 blocks/SM
    if rows == 1:
        return (4 if bps <= 8 else 2), 4
    if rows > 16:      # tensor-core GEMV, pair-major activations: 2 pairs, 8 blocks/SM (32 rows: 429 -> 400 us over
        return 2, 8    # 6 shapes)
    if pad_rows(rows) >= 8:   # 3-16 rows on the hoisted tensor-core GEMV: 2 pairs, 8 blocks/SM (cold sweep 13:00 HST:
        return 2, 8           # gate+up 2.5 bits 16 rows 67.0 -> 63.5 us, 3 bits 69.8 -> 66.1)
    return (4 if long_rows else 2), 4


def gemv_warps(lib, rows: int, bps: int, group: bool) -> int:
    """Warps per block (= work items per block) of the decode GEMV the library runs for this call."""
    cache = lib.__dict__.setdefault("_itq_warps", {})
    key = (rows, bps, bool(group))
    if key not in cache:
        cache[key] = int(lib.itq_gemv_warps(rows, bps, 1 if group else 0))
    return cache[key]


def grid_blocks(lib, sms: int, rows: int, bps: int, n: int, major: int, items: int, group: bool = False) -> int:
    """Grid of a decode GEMV: launch_params' blocks per SM, at most one block per gemv_warps() work items."""
    w = gemv_warps(lib, rows, bps, group)
    return max(1, min(sms * launch_params(rows, bps, n, major, group)[1], (items + w - 1) // w))


def gemv_blocks(p, rows: int) -> int:
    """Grid of a decode GEMV for this call (blocks per SM depend on the row count, see launch_params)."""
    m, n = p.shape
    return grid_blocks(p._lib, p._sms, rows, p.bps, n, p._major, (m // 16) * p.ksplit)


def prefetch_for(rows: int, bps: int, n: int = 0, major: int = 12, group: bool = False) -> int:
    return launch_params(rows, bps, n, major, group)[0]


def ksplit_for(m: int, n: int, sms: int) -> int:
    rows, pairs = m // 16, n // 32
    k = 1
    while rows * k < sms * 8 and 2 * k <= max(1, pairs // 2):
        k *= 2
    return k


class PackedITQLinear(torch.nn.Module):
    def __init__(self, descriptor, components, params_dtype, *, device):
        super().__init__()
        m, n = descriptor["shape"]
        fmt = (descriptor.get("V"), descriptor.get("table_bits"))
        if descriptor.get("version") != 1 or fmt not in ((4, 12), (2, 10)) or descriptor.get("L") != 16 \
                or descriptor.get("hash") != "mul" \
                or descriptor.get("layout") != "tile16-rowmajor-steps-lsb":
            raise ValueError("Unsupported ITQ descriptor")
        if params_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("ITQ needs a floating activation dtype")
        self.shape = (m, n)
        self.V = int(fmt[0])
        step_bits = int(descriptor["bps"])
        # kernel format code (itq_kernels.cu fmt_*): V4 = bits per step (4-12), V2 = 32 + bits per step (35-40;
        # 39 / 40 = 3.5 / 4 bits per weight)
        self.bps = step_bits if self.V == 4 else 32 + step_bits
        self.table_bits = int(fmt[1])
        self.block_in = block_size(descriptor["rotation_in"], n)
        self.block_out = block_size(descriptor["rotation_out"], m)
        stream = components["stream"].view(torch.uint8).reshape(-1)
        if stream.numel() != m * n * step_bits // self.V // 8:
            raise ValueError("ITQ stream size disagrees with shape and rate")
        self.register_buffer("words", stream.view(torch.int32).to(device), persistent=False)
        table = components["table"].to(torch.int32) & 0xFF
        if self.V == 4:      # four int8 per 32-bit entry (one dp4a operand)
            table = table[:, 0] | (table[:, 1] << 8) | (table[:, 2] << 16) | (table[:, 3] << 24)
        else:                # V2: the int8 pair in the low 16 bits
            table = table[:, 0] | (table[:, 1] << 8)
        self.register_buffer("table", table.to(torch.int32).contiguous().to(device), persistent=False)
        self.register_buffer("su", components["su"].to(torch.int8).to(device), persistent=False)
        self.register_buffer("sv", components["sv"].to(torch.int8).to(device), persistent=False)
        self.wscale = float(components["wscale"].reshape(-1)[0])
        for name in ("post_input_scale", "post_output_scale", "bias"):
            value = components.get(name)
            self.register_buffer(name, None if value is None else value.float().to(device), persistent=False)
        self.out_dtype = params_dtype
        self._lib = None
        self._sms = None

    def _setup(self, x):
        if self._lib is None:
            self._lib = library(x.device)
            self._sms = torch.cuda.get_device_properties(x.device).multi_processor_count
            self._major = torch.cuda.get_device_capability(x.device)[0]
            m, n = self.shape
            self.ksplit = ksplit_for(m, n, self._sms)
            self.bpsm = launch_params(1, self.bps, n, self._major)[1]

    def forward(self, x):
        return forward_group([self], x)

    def shares_input_with(self, other) -> bool:
        def same(a, b):
            return (a is None and b is None) or (a is not None and b is not None and a.shape == b.shape
                                                 and bool(torch.equal(a, b)))
        return (self.shape[1] == other.shape[1] and self.block_in == other.block_in and same(self.su, other.su)
                and same(self.post_input_scale, other.post_input_scale))


# Decode residual-stream fusion.  A producer (down / o_proj / out_proj flagged _itq_raw_out) returns its raw F32 GEMV
# output and registers it in RAW; the next bound Gemma norm (kai_gemma_norm) turns it into (y, residual) with
# add_norm (one itq_add_norm_prep launch: the producer's post, the residual add, the norm and the next projection
# group's prep) and stashes that group's int8 activations in ACT, which forward_group / forward_mlp use instead of
# their own prep.  Both registries hold at most one entry per producer / consumer.
ADD_NORM_MAX_H = 5120
RAW = {}     # data_ptr of a raw output -> producer PackedITQLinear
ACT = {}     # id(first part of the consumer group) -> (data_ptr of the norm output, rows, xq, xs, ybuf)


def _raw_output(part, y, rows, m):
    out = y[: rows * m].view(rows, m)
    previous = part.__dict__.get("_raw_ptr")
    if previous is not None:
        RAW.pop(previous, None)
    RAW[out.data_ptr()] = part
    part.__dict__["_raw_ptr"] = out.data_ptr()
    return out


def _stashed(first, x2, rows):
    entry = ACT.pop(id(first), None) if ACT else None
    if entry is not None and entry[0] == x2.data_ptr() and entry[1] == rows:
        return entry[2], entry[3], entry[4]
    return None


def raw_producer(x):
    """The producer of x when x is a registered raw decode output (F32 [rows, H]), else None."""
    if x.dtype != torch.float32 or x.dim() != 2 or not x.is_contiguous() or not RAW:
        return None
    producer = RAW.get(x.data_ptr())
    if producer is None or producer.shape[0] != x.shape[1]:
        return None
    return producer


def add_norm(x, residual, weight32, eps, consumer=None):
    """(y, residual_out) of a Gemma norm applied to producer(x) + residual, x being a raw output (raw_producer).
    consumer: the next projection layer (all-ITQ, shared input) whose activations are stashed in ACT when its input
    block equals the producer's output block."""
    producer = RAW.pop(x.data_ptr())
    producer.__dict__["_raw_ptr"] = None
    rows, H = x.shape
    dev = x.device
    producer._setup(x)
    lib = producer._lib
    stream = torch.cuda.current_stream(dev).cuda_stream
    nb = pad_rows(rows)
    res = residual.reshape(rows, H)
    if not res.is_contiguous():
        res = res.contiguous()
    y = torch.empty((rows, H), dtype=torch.bfloat16, device=dev)
    rout = torch.empty((rows, H), dtype=torch.float32, device=dev)
    parts = list(consumer.yaqa_parts) if consumer is not None else None
    if parts is not None and (parts[0].shape[1] != H or parts[0].block_in != producer.block_out):
        parts = None
    xq = xs = ybuf = zero = None
    zero_count = 0
    if parts is not None:
        for q in parts:
            q._setup(x)
        total = sum(q.shape[0] for q in parts)
        xq = torch.empty((nb, H // 4), dtype=torch.int32, device=dev)
        xs = torch.empty((nb, H // 32), dtype=torch.float32, device=dev)
        ybuf = torch.empty(nb * total, dtype=torch.float32, device=dev)
        if any(q.ksplit > 1 for q in parts):
            zero, zero_count = ybuf.data_ptr(), nb * total
    first = parts[0] if parts is not None else None
    _check(lib.itq_add_norm_prep(x.data_ptr(), H, producer.sv.data_ptr(), _ptr(producer.post_output_scale),
                                 res.data_ptr(), 1 if res.dtype == torch.bfloat16 else 0, res.stride(0),
                                 rout.data_ptr(), weight32.data_ptr(), float(eps), y.data_ptr(), rows, H,
                                 producer.block_out, _ptr(first.post_input_scale) if first is not None else None,
                                 first.su.data_ptr() if first is not None else None, _ptr(xq), _ptr(xs), zero,
                                 zero_count, stream), "add_norm_prep")
    if first is not None:
        ACT[id(first)] = (y.data_ptr(), rows, xq, xs, ybuf)
    return y, rout


def gated_norm_prep(consumer, x, z, weight32, eps):
    """Decode: RMSNormGated(x, z) (128-wide heads, silu gate, norm before gate, F32 weight) for a single-part ITQ
    consumer (the GDN out_proj), fused with the consumer's int8 prep (one itq_gated_norm_prep launch).  The prepared
    activations are stashed in ACT under the returned tensor (x as [rows, n]); calling the consumer on it runs only
    the GEMV and its epilogue.  The norm output itself is never written."""
    part = consumer.yaqa_parts[0]
    rows = x.shape[0]
    n = part.shape[1]
    x2 = x.reshape(rows, n)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    z2 = z.reshape(rows, n)
    part._setup(x2)
    dev = x2.device
    nb = pad_rows(rows)
    xq = torch.empty((nb, n // 4), dtype=torch.int32, device=dev)
    xs = torch.empty((nb, n // 32), dtype=torch.float32, device=dev)
    ybuf = torch.empty(nb * part.shape[0], dtype=torch.float32, device=dev)
    zero, zero_count = (ybuf.data_ptr(), nb * part.shape[0]) if part.ksplit > 1 else (None, 0)
    _check(part._lib.itq_gated_norm_prep(x2.data_ptr(), x2.stride(0), z2.data_ptr(), z2.stride(0), weight32.data_ptr(),
                                         float(eps), 128, _ptr(part.post_input_scale), part.su.data_ptr(), xq.data_ptr(),
                                         xs.data_ptr(), rows, n, part.block_in, zero, zero_count,
                                         torch.cuda.current_stream(dev).cuda_stream), "gated_norm_prep")
    ACT[id(part)] = (x2.data_ptr(), rows, xq, xs, ybuf)
    return x2


def gated_norm_supported(consumer, norm) -> bool:
    """Load-time check for gated_norm_prep: an all-ITQ single-part consumer whose input is [heads x 128] and a
    vLLM RMSNormGated with the Qwen3.5 / 3.8 GDN settings."""
    parts = getattr(consumer, "yaqa_parts", None)
    if parts is None or len(parts) != 1 or not isinstance(parts[0], PackedITQLinear):
        return False
    part = parts[0]
    weight = getattr(norm, "weight", None)
    return (getattr(consumer, "_itq_shared_input", None) is not None and part.bias is None
            and part.block_in in (256, 512, 1024, 2048) and part.shape[1] % part.block_in == 0
            and weight is not None and weight.dtype == torch.float32 and weight.numel() == 128
            and getattr(norm, "activation", None) in ("silu", "swish") and getattr(norm, "norm_before_gate", None) is True
            and getattr(norm, "group_size", None) in (None, 128))


def mlp_parts(gate_up_parts, down_parts):
    """((gate, up), down) when a decode MLP can run fused (forward_mlp), else None.  The caller checks that gate
    and up share their input (the load-time _itq_shared_input flag: no host-synchronising comparison here)."""
    parts = list(gate_up_parts)
    downs = list(down_parts)
    if len(parts) != 2 or len(downs) != 1 or not all(isinstance(p, PackedITQLinear) for p in parts + downs):
        return None
    gate, up = parts
    down = downs[0]
    m = gate.shape[0]
    if (up.shape != gate.shape or down.shape[1] != m or gate.bias is not None
            or up.bias is not None or down.bias is not None):
        return None
    if not (gate.block_out == up.block_out == down.block_in and gate.block_out in (256, 512, 1024, 2048)):
        return None
    return (gate, up), down


def forward_mlp(gate_up, down, x, raw=False):
    """Decode MLP down(silu(gate(x)) * up(x)) for <= MAX_DECODE_ROWS rows in five launches: prep, gate+up GEMV,
    itq_swiglu_prep (post of gate / up + vLLM's SiLU-and-mul + down's prep), down GEMV, down post.  Bit-identical
    to forward_group(gate_up) -> act_and_mul -> forward_group(down), with three launches fewer per layer."""
    gate, up = gate_up
    n, m, out_m = gate.shape[1], gate.shape[0], down.shape[0]
    shape = x.shape
    x2 = x.reshape(-1, n)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows = x2.shape[0]
    for p in (gate, up, down):
        p._setup(x2)
    lib = gate._lib
    stream = torch.cuda.current_stream(x2.device).cuda_stream
    bf16 = 1 if x2.dtype == torch.bfloat16 else 0
    if x2.dtype not in (torch.bfloat16, torch.float32):
        x2 = x2.float()
        bf16 = 0
    nb = pad_rows(rows)
    dev = x2.device
    split = gate.ksplit > 1 or up.ksplit > 1
    stashed = _stashed(gate, x2, rows)
    if stashed is not None:         # prepared (and split-K output zeroed) by the norm's itq_add_norm_prep
        xq, xs, ybuf = stashed
    else:
        xq = torch.empty((nb, n // 4), dtype=torch.int32, device=dev)
        xs = torch.empty((nb, n // 32), dtype=torch.float32, device=dev)
        ybuf = torch.empty(2 * nb * m, dtype=torch.float32, device=dev)
        _check(lib.itq_prep(x2.data_ptr(), bf16, _ptr(gate.post_input_scale), gate.su.data_ptr(), xq.data_ptr(),
                            xs.data_ptr(), None, rows, n, n, gate.block_in, 1, ybuf.data_ptr() if split else None,
                            2 * nb * m if split else 0, stream), "prep")
    yg, yu = ybuf.data_ptr(), ybuf.data_ptr() + 4 * nb * m
    if gate.bps == up.bps:
        vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        items = (m // 16) * (gate.ksplit + up.ksplit)
        blocks = grid_blocks(gate._lib, gate._sms, rows, gate.bps, n, gate._major, items, group=True)
        _check(lib.itq_gemv_group(2, (vp * 2)(gate.words.data_ptr(), up.words.data_ptr()), (vp * 2)(yg, yu),
                                  (ci * 2)(m, m), (ci * 2)(gate.ksplit, up.ksplit), (cf * 2)(gate.wscale, up.wscale),
                                  gate.table.data_ptr(), xq.data_ptr(), xs.data_ptr(), n, gate.bps, rows, blocks,
                                  prefetch_for(rows, gate.bps, n, gate._major, group=True), stream), "gemv_group")
    else:
        for p, y in ((gate, yg), (up, yu)):
            _check(lib.itq_gemv(p.words.data_ptr(), p.table.data_ptr(), xq.data_ptr(), xs.data_ptr(), y, m, n, p.bps,
                                p.table_bits, rows, p.ksplit, p.wscale, gemv_blocks(p, rows),
                                prefetch_for(rows, p.bps, n, p._major) + (1000 if split else 0), stream), "gemv")
    xq_d = torch.empty((nb, m // 4), dtype=torch.int32, device=dev)
    xs_d = torch.empty((nb, m // 32), dtype=torch.float32, device=dev)
    y_d = torch.empty(nb * out_m, dtype=torch.float32, device=dev)
    split_d = down.ksplit > 1
    _check(lib.itq_swiglu_prep(yg, yu, m, gate.sv.data_ptr(), up.sv.data_ptr(), _ptr(gate.post_output_scale),
                               _ptr(up.post_output_scale), _ptr(down.post_input_scale), down.su.data_ptr(),
                               xq_d.data_ptr(), xs_d.data_ptr(), rows, m, gate.block_out,
                               y_d.data_ptr() if split_d else None, nb * out_m if split_d else 0, stream), "swiglu_prep")
    _check(lib.itq_gemv(down.words.data_ptr(), down.table.data_ptr(), xq_d.data_ptr(), xs_d.data_ptr(), y_d.data_ptr(),
                        out_m, m, down.bps, down.table_bits, rows, down.ksplit, down.wscale, gemv_blocks(down, rows),
                        prefetch_for(rows, down.bps, m, down._major) + (1000 if split_d else 0), stream), "gemv")
    if raw and out_m <= ADD_NORM_MAX_H and len(shape) == 2:
        return _raw_output(down, y_d, rows, out_m)      # the next norm applies down's post (add_norm)
    out = torch.empty((rows, out_m), dtype=torch.bfloat16, device=dev)
    _check(lib.itq_post(y_d.data_ptr(), down.sv.data_ptr(), _ptr(down.post_output_scale), None, out.data_ptr(), rows,
                        out_m, out_m, out_m, down.block_out, stream), "post")
    out = out.reshape(*shape[:-1], out_m)
    return out if down.out_dtype == torch.bfloat16 else out.to(down.out_dtype)


def forward_group(parts, x, raw=False):
    """One input rotation/quantization shared by all parts, then GEMV + epilogue per part written
    straight into its column slice of the concatenated output (no concat kernel)."""
    first = parts[0]
    n = first.shape[1]
    total = sum(p.shape[0] for p in parts)
    shape = x.shape
    x2 = x.reshape(-1, n)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows = x2.shape[0]
    for p in parts:
        p._setup(x2)
    lib = first._lib
    stream = torch.cuda.current_stream(x2.device).cuda_stream
    bf16 = 1 if x2.dtype == torch.bfloat16 else 0
    if x2.dtype not in (torch.bfloat16, torch.float32):
        x2 = x2.float()
        bf16 = 0
    out = torch.empty((rows, total), dtype=torch.bfloat16, device=x2.device)
    if rows == 0:
        return out.reshape(*shape[:-1], total).to(first.out_dtype)
    if rows <= max_gemv_rows(x2.device.index if x2.device.index is not None else torch.cuda.current_device()):
        nb = pad_rows(rows)
        # Padded rows (rows..nb-1) stay uninitialized: GEMV rows are independent and post writes
        # only the first `rows`, so no fill kernels are needed.
        split = any(p.ksplit > 1 for p in parts)
        stashed = _stashed(first, x2, rows)
        if stashed is not None:     # prepared (and split-K output zeroed) by the norm's itq_add_norm_prep
            xq, xs, ybuf = stashed
        else:
            xq = torch.empty((nb, n // 4), dtype=torch.int32, device=x2.device)
            xs = torch.empty((nb, n // 32), dtype=torch.float32, device=x2.device)
            ybuf = torch.empty(nb * total, dtype=torch.float32, device=x2.device)
            _check(lib.itq_prep(x2.data_ptr(), bf16, _ptr(first.post_input_scale), first.su.data_ptr(), xq.data_ptr(),
                                xs.data_ptr(), None, rows, n, n, first.block_in, 1,
                                ybuf.data_ptr() if split else None, nb * total if split else 0, stream), "prep")
        if len(parts) > 1 and len({p.bps for p in parts}) == 1 and len(parts) <= 8:
            k = len(parts)
            vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
            ys, offset = [], 0
            for p in parts:
                ys.append(ybuf.data_ptr() + 4 * offset)
                offset += nb * p.shape[0]
            items = sum((p.shape[0] // 16) * p.ksplit for p in parts)
            blocks = grid_blocks(first._lib, first._sms, rows, first.bps, n, first._major, items, group=True)
            _check(lib.itq_gemv_group(k, (vp * k)(*[p.words.data_ptr() for p in parts]), (vp * k)(*ys),
                                      (ci * k)(*[p.shape[0] for p in parts]), (ci * k)(*[p.ksplit for p in parts]),
                                      (cf * k)(*[p.wscale for p in parts]), first.table.data_ptr(), xq.data_ptr(),
                                      xs.data_ptr(), n, first.bps, rows, blocks,
                                      prefetch_for(rows, first.bps, n, first._major, group=True), stream), "gemv_group")
            _check(lib.itq_post_group(k, (vp * k)(*ys), (vp * k)(*[p.sv.data_ptr() for p in parts]),
                                      (vp * k)(*[_ptr(p.post_output_scale) for p in parts]),
                                      (vp * k)(*[_ptr(p.bias) for p in parts]), (ci * k)(*[p.shape[0] for p in parts]),
                                      (ci * k)(*[p.block_out for p in parts]), out.data_ptr(), total, rows, stream),
                   "post_group")
            out = out.reshape(*shape[:-1], total)
            return out if first.out_dtype == torch.bfloat16 else out.to(first.out_dtype)
        offset, col = 0, 0
        for p in parts:
            m = p.shape[0]
            y = ybuf[offset:offset + nb * m]
            _check(lib.itq_gemv(p.words.data_ptr(), p.table.data_ptr(), xq.data_ptr(), xs.data_ptr(), y.data_ptr(),
                                m, n, p.bps, p.table_bits, rows, p.ksplit, p.wscale, gemv_blocks(p, rows),
                                prefetch_for(rows, p.bps, n, p._major) + (1000 if split else 0), stream), "gemv")
            if raw and len(parts) == 1 and p.bias is None and m <= ADD_NORM_MAX_H and len(shape) == 2:
                return _raw_output(p, ybuf, rows, m)    # the next norm applies the post (add_norm)
            _check(lib.itq_post(y.data_ptr(), p.sv.data_ptr(), _ptr(p.post_output_scale), _ptr(p.bias),
                                out[:, col:].data_ptr(), rows, m, total, m, p.block_out, stream), "post")
            offset += nb * m
            col += m
    else:
        xr = torch.empty((rows, n), dtype=torch.bfloat16, device=x2.device)
        _check(lib.itq_prep(x2.data_ptr(), bf16, _ptr(first.post_input_scale), first.su.data_ptr(), None, None,
                            xr.data_ptr(), rows, n, n, first.block_in, 0, None, 0, stream), "prep")
        col = 0
        if prefill_mode(x2.device) == "int8":
            # per-token int8 activations (the rotation spreads each token's energy over the row), one fused kernel
            # (bit-identical to torch's abs / amax / div / round / clamp chain); the int32 GEMM output goes straight
            # to the post kernel with the per-row scale (= torch's y.float().mul_(scale * wscale), no F32 copy)
            x8 = torch.empty((rows, n), dtype=torch.int8, device=x2.device)
            scale = torch.empty((rows, 1), dtype=torch.float32, device=x2.device)
            _check(lib.itq_quant_rows(xr.data_ptr(), x8.data_ptr(), scale.data_ptr(), rows, n, stream), "quant_rows")
            for p in parts:
                m = p.shape[0]
                w8 = torch.empty((m, n), dtype=torch.int8, device=x2.device)
                _check(lib.itq_decode_i8(p.words.data_ptr(), p.table.data_ptr(), w8.data_ptr(), m, n, p.bps,
                                         stream), "decode_i8")
                y = torch._int_mm(x8, w8.t())
                del w8
                rs = scale * p.wscale
                _check(lib.itq_post_i32(y.data_ptr(), rs.data_ptr(), p.sv.data_ptr(), _ptr(p.post_output_scale),
                                        _ptr(p.bias), out[:, col:].data_ptr(), rows, m, total, m, p.block_out, stream),
                       "post_i32")
                col += m
            out = out.reshape(*shape[:-1], total)
            return out if first.out_dtype == torch.bfloat16 else out.to(first.out_dtype)
        for p in parts:
            m = p.shape[0]
            w = torch.empty((m, n), dtype=torch.bfloat16, device=x2.device)
            _check(lib.itq_decode(p.words.data_ptr(), p.table.data_ptr(), w.data_ptr(), m, n, p.bps, p.table_bits,
                                  p.wscale, stream), "decode")
            y = torch.matmul(xr, w.t()).float()
            _check(lib.itq_post(y.data_ptr(), p.sv.data_ptr(), _ptr(p.post_output_scale), _ptr(p.bias),
                                out[:, col:].data_ptr(), rows, m, total, m, p.block_out, stream), "post")
            col += m
    out = out.reshape(*shape[:-1], total)
    return out if first.out_dtype == torch.bfloat16 else out.to(first.out_dtype)
