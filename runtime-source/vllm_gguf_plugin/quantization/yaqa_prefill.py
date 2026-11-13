# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded, temporary latent F16 weights for large QTIP prefill."""

import hashlib
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from ..native.cuda_architecture import resolve_architecture

MAX_WEIGHT_BYTES = 32 * 1024**2
_MEASURED_TILES = {
    ((7680, 2048), 2): 64,
    ((2048, 7680), 2): 64,
    ((7680, 2048), 3): 64,
    ((2048, 7680), 3): 64,
    ((7680, 2048), 4): 64,
    ((2048, 7680), 4): 64,
    ((6144, 2048), 3): 128,
}


# Token-count-exact extensions; original measured entries retain their policy.
_EXTRA_MEASURED_TILES = {
    ((512, 2048), 2, 2048): 32,
    ((512, 2048), 3, 2048): 32,
    ((512, 2048), 4, 2048): 32,
    ((2048, 2048), 2, 512): 32,
    ((2048, 2048), 2, 2048): 32,
    ((2048, 2048), 3, 512): 32,
    ((2048, 2048), 3, 2048): 32,
    ((2048, 2048), 4, 512): 32,
    ((2048, 2048), 4, 2048): 32,
    ((6144, 2048), 2, 2048): 32,
    ((6144, 2048), 4, 2048): 32,
}
_EXTRA_MEASURED_PROOF_SHA256 = (
    "5843a6fd40152a78baf531231590b3d2ee629494d383376913cbfaa22361ac8c"
)


def prefill_mode():
    mode = os.getenv("VLLM_GGUF_YAQA_PREFILL", "packed")
    if mode not in ("packed", "transient", "cuda-transient"):
        raise ValueError(
            "VLLM_GGUF_YAQA_PREFILL must be packed, transient, or cuda-transient"
        )
    return mode


def transient_prefill_runtime_marker():
    marker = {
        "mode": prefill_mode(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_prepare_source_sha256": hashlib.sha256(
            Path(__file__).with_name("yaqa_kernels.py").read_bytes()
        ).hexdigest(),
        "max_weight_bytes": MAX_WEIGHT_BYTES,
        "minimum_batch": 512,
        "additional_policy": "exact-token512-or2048/whole-speedup-at-least1.05",
        "additional_policy_proof_sha256": _EXTRA_MEASURED_PROOF_SHA256,
        "additional_tiles": [
            {"shape": list(shape), "bits": bits, "tokens": tokens, "tile": tile}
            for (shape, bits, tokens), tile in sorted(_EXTRA_MEASURED_TILES.items())
        ],
    }
    if marker["mode"] == "cuda-transient":
        from .yaqa_latent_cuda import runtime_identity

        marker["cuda_decoder"] = runtime_identity()
    return marker


def _cublas_tile(layer):
    """cuBLAS mode: any eligible matrix within the workspace bound takes the transient path; the
    tile then only sizes the FP16-overflow repair kernel."""
    m, n = layer.shape
    if m < 32 or m % 16 or n % 16 or m * n * 2 > MAX_WEIGHT_BYTES:
        return None
    return 64 if m >= 64 else 32


def _prefill_tile_sm86(layer, batch):
    if (
        batch < 512
        or getattr(layer, "prefill_backend", "packed")
        not in ("transient", "cuda-transient")
        or not getattr(layer, "transient_prefill_eligible", False)
    ):
        return None
    key = (tuple(layer.shape), int(layer.bits))
    measured = _MEASURED_TILES.get(key)
    if measured is not None:
        return measured
    m, n = layer.shape
    if m < 128 or m % 16 or n % 16 or m * n * 2 > MAX_WEIGHT_BYTES:
        return _cublas_tile(layer) if _PREFILL_GEMM == "cublas" else None
    tile = _EXTRA_MEASURED_TILES.get((key[0], key[1], batch))
    if tile is None and _PREFILL_GEMM == "cublas":
        tile = _cublas_tile(layer)
    return tile

def _prefill_tile_sm120(layer, batch):
    if (
        batch < 512
        or getattr(layer, "prefill_backend", "packed")
        not in ("transient", "cuda-transient")
        or not getattr(layer, "transient_prefill_eligible", False)
    ):
        return None
    tile = _MEASURED_TILES.get((tuple(layer.shape), int(layer.bits)))
    if tile is None and _PREFILL_GEMM == "cublas":
        tile = _cublas_tile(layer)
    return tile

def _prefill_tile_for_arch(layer, batch, capability):
    if tuple(capability) == (8, 6):
        return _prefill_tile_sm86(layer, batch)
    return _prefill_tile_sm120(layer, batch)


def prefill_tile(layer, batch):
    device = layer.trellis.device
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else resolve_architecture()
    return _prefill_tile_for_arch(layer, batch, capability)


@triton.jit
def decode_latent_half_pairs(
    W,
    LutPairs,
    DensePairs,
    M: tl.constexpr,
    N: tl.constexpr,
    BITS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    clamped = tl.minimum(index, M * N // 2 - 1)
    row, col = clamped // (N // 2), (clamped % (N // 2)) * 2
    tile = (row // 16) * (N // 16) + col // 16
    p = (row % 16) * 16 + col % 16
    if PERMUTE:
        p = (p // 16 % 8) * 32 + (p // 2 % 4) * 8 + (p // 8 % 2) * 4 + (p // 128) * 2
    start = (p // 2) * (2 * BITS)
    word, shift = start // 16, start % 16
    low = tl.load(W + tile * (16 * BITS) + word).to(tl.uint16).to(tl.uint32)
    high = (
        tl.load(W + tile * (16 * BITS) + (word + 1) % (16 * BITS))
        .to(tl.uint16)
        .to(tl.uint32)
    )
    state = (((low << 16) | high) >> (16 - shift)) & 65535
    hashed = state * (state + 1)
    value = tl.load(LutPairs + ((hashed >> 6) & 511)).to(tl.uint32)
    value ^= ((hashed >> 15) & 1) << 15
    tl.store(DensePairs + index, value, index < M * N // 2)


@triton.jit(
    do_not_specialize=["BATCH"],
    do_not_specialize_on_alignment=["BATCH"],
)
def dense_prepared_gemm(
    X,
    HalfX,
    SafeRows,
    Dense,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REPAIR_ONLY: tl.constexpr = False,
):
    # REPAIR_ONLY: Y already holds the FP16 x FP16 -> FP32 product of every row (cuBLAS); only tiles
    # with a row outside the FP16 range are recomputed (and stored) by the fallback below.
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    safe_rows = tl.load(SafeRows + batch, batch < BATCH, 1)
    all_safe = tl.min(safe_rows, 0) != 0
    acc = tl.zeros((BM, BN), tl.float32)
    if all_safe:
        if not REPAIR_ONLY:
            for begin in range(tl.cdiv(N, BK)):
                c = begin * BK + columns
                weight = tl.load(
                    Dense + c[:, None] + rows[None, :] * N,
                    (c[:, None] < N) & (rows[None, :] < M),
                    0,
                )
                value = tl.load(
                    HalfX + batch[:, None] * N + c[None, :],
                    (batch[:, None] < BATCH) & (c[None, :] < N),
                    0,
                )
                acc = tl.dot(value, weight, acc)
            tl.store(
                Y + batch[:, None] * M + rows[None, :],
                acc,
                (batch[:, None] < BATCH) & (rows[None, :] < M),
            )
    else:
        fallback_columns = tl.arange(0, 32)
        for begin in range(tl.cdiv(N, 32)):
            c = begin * 32 + fallback_columns
            weight = tl.load(
                Dense + c[:, None] + rows[None, :] * N,
                (c[:, None] < N) & (rows[None, :] < M),
                0,
            )
            value = tl.load(
                X + batch[:, None] * N + c[None, :],
                (batch[:, None] < BATCH) & (c[None, :] < N),
                0,
            ).to(tl.float32)
            safe = tl.max(tl.max(tl.abs(value), 0), 0) <= 65504.0
            if safe:
                acc = tl.dot(value.to(tl.float16), weight, acc)
            else:
                acc = tl.dot(
                    value, weight.to(tl.float32), acc, input_precision="tf32x3"
                )
        tl.store(
            Y + batch[:, None] * M + rows[None, :],
            acc,
            (batch[:, None] < BATCH) & (rows[None, :] < M),
        )


# cuBLAS for the all-FP16-safe product (the default); "triton" keeps the original single-kernel GEMM.
_PREFILL_GEMM = os.getenv("VLLM_GGUF_YAQA_PREFILL_GEMM", "cublas")
if _PREFILL_GEMM not in ("cublas", "triton"):
    raise ValueError("VLLM_GGUF_YAQA_PREFILL_GEMM must be cublas or triton")


def transient_prefill_linear(source, layer, block_n, output=None):
    """Decode one bounded matrix for this call; never retain it on the layer."""
    from .yaqa_kernels import _prepare_half_input

    m, n = layer.shape
    batch = source.shape[0]
    if m * n * 2 > MAX_WEIGHT_BYTES:
        raise ValueError("Transient latent matrix exceeds its workspace bound")
    decoder_key = getattr(layer, "cuda_latent_key", None)
    if decoder_key is not None:
        from .yaqa_latent_cuda import decode_half, decoder_warps

        warps = decoder_warps(layer)
        if warps is None:
            raise ValueError("Loaded CUDA latent decoder has no eligible policy")
        dense = decode_half(
            layer.trellis, layer.mma_lut, m, n, layer.bits, warps, decoder_key
        )
    else:
        dense = torch.empty((m, n), dtype=torch.float16, device=source.device)
        half_lut = layer.tlut.half().view(torch.int32)
        decode_latent_half_pairs[(triton.cdiv(m * n // 2, 1024),)](
            layer.trellis,
            half_lut,
            dense.view(torch.int32),
            M=m,
            N=n,
            BITS=layer.bits,
            PERMUTE=layer.permutation,
            BLOCK=1024,
            num_warps=4,
        )
    half_source = torch.empty_like(source, dtype=torch.float16)
    safe_rows = torch.empty((batch,), dtype=torch.int32, device=source.device)
    _prepare_half_input[(batch,)](
        source,
        half_source,
        safe_rows,
        N=n,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
    )
    result = (
        torch.empty((batch, m), dtype=torch.float32, device=source.device)
        if output is None
        else output
    )
    repair_only = _PREFILL_GEMM == "cublas"
    if repair_only:
        # FP16 operands, FP32 accumulation and output: the math of the kernel's all-safe branch.
        # Rows outside the FP16 range are recomputed by the kernel below (no host synchronization),
        # in 16-row tiles so a single out-of-range row (e.g. a first-token spike) costs little.
        torch.mm(half_source, dense.t(), out_dtype=torch.float32, out=result)
    bm = 16 if repair_only else 64
    dense_prepared_gemm[(triton.cdiv(batch, bm), triton.cdiv(m, block_n))](
        source,
        half_source,
        safe_rows,
        dense,
        result,
        M=m,
        N=n,
        BATCH=batch,
        BM=bm,
        BN=block_n,
        BK=64,
        REPAIR_ONLY=repair_only,
        num_warps=4,
        num_stages=3,
    )
    return result
