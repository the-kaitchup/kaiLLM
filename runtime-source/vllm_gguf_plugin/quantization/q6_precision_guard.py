# SPDX-License-Identifier: Apache-2.0
"""Default-off Q6 precision operators; original whole-M unsafe repair."""

import torch
import triton
import triton.language as tl

from . import canonical_q6_tile as tile
from . import native_bf16_q8q6_power2 as guard
from .compact_decode import load_gguf


@triton.jit
def _finish(
    X,
    WPtrs,
    Y,
    Out,
    Unsafe,
    Inverse,
    M: tl.constexpr,
    K: tl.constexpr,
    TYPES: tl.constexpr,
    NS: tl.constexpr,
    TOTAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    batches = tl.arange(0, BM)
    unsafe = tl.load(Unsafe + batches, batches < M, other=0)
    inverse = tl.load(Inverse + batches, batches < M, other=1.0)
    tile_start = 0
    output_start = 0
    for part in tl.static_range(len(NS)):
        n = NS[part]
        tiles = tl.cdiv(n, BN)
        if (tile >= tile_start) & (tile < tile_start + tiles):
            cols = (tile - tile_start) * BN + tl.arange(0, BN)
            mask = (batches[:, None] < M) & (cols[None, :] < n)
            result = tl.load(
                Y + batches[:, None] * TOTAL + output_start + cols[None, :],
                mask,
                other=0,
            )
            # Preserve native F32 store -> F32 unscale -> final BF16 RN order.
            result = tl.where(
                inverse[:, None] != 1.0, result * inverse[:, None], result
            )
            if tl.sum(unsafe, 0) != 0:
                w = tl.load(WPtrs + part).to(tl.pointer_type(tl.uint8))
                columns = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + columns
                    weight = load_gguf(
                        w,
                        cols[:, None] * K + k[None, :],
                        (cols[:, None] < n) & (k[None, :] < K),
                        w,
                        w,
                        w,
                        TYPES[part],
                    )
                    x = tl.load(
                        X + batches[:, None] * K + k[None, :],
                        (batches[:, None] < M) & (k[None, :] < K),
                        other=0,
                    )
                    acc += tl.sum(x[:, None, :] * weight[None, :, :], 2)
                result = tl.where(unsafe[:, None] != 0, acc, result)
            tl.store(
                Out + batches[:, None] * TOTAL + output_start + cols[None, :],
                result,
                mask,
            )
        tile_start += tiles
        output_start += n


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _finish_gemm(
    x_ptr,
    w_ptr,
    fast_ptr,
    y_ptr,
    unsafe_ptr,
    inverse_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The unsafe branch is the original whole-M F32 compact GEMM tree.
    batches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    unsafe = tl.load(unsafe_ptr + batches, batches < M, other=0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (batches[:, None] < M) & (rows[None, :] < N)
    result = tl.load(fast_ptr + batches[:, None] * N + rows[None, :], mask, other=0)
    inverse = tl.load(inverse_ptr + batches, batches < M, other=1.0)
    result = tl.where(inverse[:, None] != 1.0, result * inverse[:, None], result)
    if tl.sum(unsafe, axis=0) != 0:
        columns = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            k = start * BLOCK_K + columns
            w = load_gguf(
                w_ptr,
                rows[:, None] * K + k[None, :],
                (rows[:, None] < N) & (k[None, :] < K),
                w_ptr,
                w_ptr,
                w_ptr,
                QTYPE,
            )
            x = tl.load(
                x_ptr + batches[:, None] * K + k[None, :],
                (batches[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.float32)
            acc = tl.dot(x, tl.trans(w), acc, input_precision="tf32x3")
        result = tl.where(unsafe[:, None] != 0, acc, result)
    tl.store(y_ptr + batches[:, None] * N + rows[None, :], result, mask)


def sanitize(x):
    raw = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    clean = torch.empty_like(x)
    unsafe = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    inverse = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    guard._sanitize_d4[(x.shape[0],)](
        x,
        raw,
        clean,
        unsafe,
        inverse,
        K=x.shape[1],
        BLOCKS=triton.next_power_of_2(triton.cdiv(x.shape[1], 32)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return raw, clean, unsafe, inverse


def finish(raw, packed, pointers, fast, unsafe, inverse, fp32_output):
    m, n = fast.shape
    out = torch.empty(
        (m, n),
        dtype=torch.float32 if fp32_output else torch.bfloat16,
        device=raw.device,
    )
    if m <= 8:
        _finish[(triton.cdiv(n, 4),)](
            raw,
            pointers,
            fast,
            out,
            unsafe,
            inverse,
            M=m,
            K=raw.shape[1],
            TYPES=(14,),
            NS=(n,),
            TOTAL=n,
            BM=triton.next_power_of_2(m),
            BN=4,
            BK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _finish_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            raw,
            packed,
            fast,
            out,
            unsafe,
            inverse,
            M=m,
            N=n,
            K=raw.shape[1],
            QTYPE=14,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return out


def validate(x, packed, pointers, n, k):
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or tuple(torch.cuda.get_device_capability(x.device)) not in ((8, 6), (12, 0))
        or not 1 <= x.shape[0] <= 32
        or x.shape[1] != k
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.device != x.device
        or packed.shape != (n, k // 256 * 210)
        or pointers.dtype != torch.int64
        or pointers.shape != (1,)
        or pointers.device != x.device
    ):
        raise ValueError("SM86/SM120 BF16 Q6 precision operator metadata differs")
    tile.validate_layout(x.shape[0], n, k, 14, packed.numel())


@torch.library.custom_op("gguf_yaqa::q6_humming_guarded", mutates_args=("locks",))
def body(
    x: torch.Tensor,
    packed: torch.Tensor,
    pointers: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    locks: torch.Tensor,
    n: int,
    k: int,
    layer_config: str,
    compute_config: str,
    tuning_config: str,
    fp32_output: bool,
) -> torch.Tensor:
    validate(x, packed, pointers, n, k)
    if (
        any(t.device != x.device for t in (weight, scale, locks))
        or scale.dtype != torch.float16
    ):
        raise ValueError("Original prepared Humming device/scale differs")
    from humming.ops import humming_gemm

    raw, clean, unsafe, inverse = sanitize(x)
    fast = humming_gemm(
        layer_config=layer_config,
        compute_config=compute_config,
        tuning_config=tuning_config,
        inputs=clean.half(),
        weight=weight,
        weight_scale=scale,
        locks=locks,
    )
    if fast.dtype != torch.float16 or fast.shape[0] != x.shape[0] or fast.shape[1] < n:
        raise ValueError("Original FP16 Humming output contract differs")
    fast = fast[:, :n].contiguous()
    return finish(raw, packed, pointers, fast, unsafe, inverse, fp32_output)


@body.register_fake
def _fake_body(
    x,
    packed,
    pointers,
    weight,
    scale,
    locks,
    n,
    k,
    layer_config,
    compute_config,
    tuning_config,
    fp32_output,
):
    return x.new_empty(
        (x.shape[0], n), dtype=torch.float32 if fp32_output else torch.bfloat16
    )


@torch.library.custom_op("gguf_yaqa::q6_canonical_head_guarded", mutates_args=())
def head(
    x: torch.Tensor,
    packed: torch.Tensor,
    pointers: torch.Tensor,
    codes: torch.Tensor,
    n: int,
    k: int,
    library: str,
    sms: int,
    fp32_output: bool,
) -> torch.Tensor:
    validate(x, packed, pointers, n, k)
    from .q6_precision import head_library_identity

    if tuple(head_library_identity(library)["target"]) != tuple(
        torch.cuda.get_device_capability(x.device)
    ):
        raise ValueError("Actual Q6 head tensor target and library differ")
    if (
        codes.dtype != torch.uint8
        or codes.ndim != 1
        or codes.device != x.device
        or not codes.is_contiguous()
        or codes.numel() != triton.cdiv(n, 16) * (k // 16) * 256
    ):
        raise ValueError("Canonical Q6 tile-code metadata differs")
    raw, clean, unsafe, inverse = sanitize(x)
    fast = tile.registered_op()(clean, codes, n, k, False, library, sms, 0)
    return finish(raw, packed, pointers, fast, unsafe, inverse, fp32_output)


@head.register_fake
def _fake_head(x, packed, pointers, codes, n, k, library, sms, fp32_output):
    return x.new_empty(
        (x.shape[0], n), dtype=torch.float32 if fp32_output else torch.bfloat16
    )
