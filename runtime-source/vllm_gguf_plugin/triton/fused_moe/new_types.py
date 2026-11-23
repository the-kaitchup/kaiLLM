# SPDX-License-Identifier: Apache-2.0
"""Triton fused-MoE GEMMs for expert formats outside the GGML MMQ / MMVQ families (Inkling 2 GGUFs, ITQ prefill):

    MXFP4 (39), NVFP4 (40), Q1_0 (41), Q2_0 (42)    llama.cpp master block types, dequantized per block in registers
    float (BF16 / F16 / F32 experts)                 W [E, N, K] in its own dtype
    int8 + per-expert scale                          ITQ prefill: W_rot [E, N, K] int8 table values (itq_decode_i8),
                                                     out = scale[e] * (x . W)

Same contract as the other kernels of this package: tokens sorted by expert with vLLM's moe_align_block_size, grid
sized from tensor shapes (no host synchronisation, CUDA-graph safe), BF16 / F16 activations, float32 accumulation.
out[t * top_k + j] = x[t] . W[expert(t, j)]^T, or with top_k = 1 one activation row per routed slot.
Dequantization matches vllm_gguf_plugin/inkling/ggml_types.py (bit-exact with ggml), rounded to the activation dtype.
Decode batches (a few routed rows): moe_new_type_gemv (MXFP4 / NVFP4 / Q2_0), one program row per routed row that
reads its expert id directly (no sorting, no padded M tiles), products in float32 (as tensor cores on BF16 inputs).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..gemm.utils import load_f16_from_u8
from .utils import load_moe_token_info, load_moe_x_tile

MXFP4, NVFP4, Q1_0, Q2_0 = 39, 40, 41, 42
BLOCK = {MXFP4: (32, 17), NVFP4: (64, 36), Q1_0: (128, 18), Q2_0: (64, 18)}
NEW_TYPES = frozenset(BLOCK)


@triton.jit
def _fp4(n):
    """kvalues_fp4[n] (doubled E2M1): 0 1 2 3 4 6 8 12 and their negatives."""
    m = n & 7
    mag = tl.where(m < 4, m, tl.where(m < 6, (m - 4) * 2 + 4, (m - 6) * 4 + 8))
    return tl.where(n >= 8, -mag, mag).to(tl.float32)


@triton.jit
def _ue4m3_half(x):
    """ggml_ue4m3_to_fp32: unsigned E4M3 (bias 7, sign ignored) x 0.5, 0 for 0 and 0x7F."""
    x = x.to(tl.int32)
    e = (x >> 3) & 15
    m = (x & 7).to(tl.float32)
    raw = tl.where(e == 0, m * 0.001953125, (1.0 + m * 0.125) * tl.exp2((e - 7).to(tl.float32)))
    return tl.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)


@triton.jit
def _moe_prologue(sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_output, offs_token, token_mask = load_moe_token_info(sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens,
                                                              BLOCK_M=BLOCK_M)
    expert = tl.load(expert_ids_ptr + pid_m)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    return offs_output, offs_token, token_mask, expert, offs_n, offs_n < n


@triton.jit
def mxfp4_moe_kernel(x_ptr, w_ptr, y_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                     num_valid_tokens, top_k, n, num_k_blocks, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                     stride_ym, stride_yn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_BLOCKS: tl.constexpr):
    offs_output, offs_token, token_mask, expert, offs_n, n_mask = _moe_prologue(
        sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M, BLOCK_N)
    if expert < 0:
        return
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_b = tl.arange(0, 16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_rows = w_ptr + expert.to(tl.int64) * stride_we + offs_n[:, None] * stride_wn
    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        # MXFP4 block: e (E8M0) then 16 bytes; low nibbles = values 0-15, high = 16-31 (the Q4_0 interleave)
        x_tile, cur_kb, kb_mask = load_moe_x_tile(x_ptr, num_k_blocks, stride_xm, stride_xk, offs_token, token_mask,
                                                  kb_start, offs_kb, offs_b, BLOCK_M=BLOCK_M,
                                                  BLOCK_K_BLOCKS=BLOCK_K_BLOCKS)
        dt = x_tile.dtype
        blk = w_rows + cur_kb[None, :] * 17
        m2 = n_mask[:, None] & kb_mask[None, :]
        e = tl.load(blk, mask=m2, other=128).to(tl.float32)
        d = tl.exp2(e - 128.0)
        q = tl.load(blk[:, :, None] + 1 + offs_b[None, None, :], mask=m2[:, :, None], other=0).to(tl.int32)
        lo = (_fp4(q & 15) * d[:, :, None]).to(dt)
        hi = (_fp4(q >> 4) * d[:, :, None]).to(dt)
        w_tile = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K_BLOCKS * 32))
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)
    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


@triton.jit
def nvfp4_moe_kernel(x_ptr, w_ptr, y_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                     num_valid_tokens, top_k, n, num_k_blocks, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                     stride_ym, stride_yn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_BLOCKS: tl.constexpr):
    offs_output, offs_token, token_mask, expert, offs_n, n_mask = _moe_prologue(
        sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M, BLOCK_N)
    if expert < 0:
        return
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_s = tl.arange(0, 4)
    offs_j = tl.arange(0, 8)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_rows = w_ptr + expert.to(tl.int64) * stride_we + offs_n[:, None] * stride_wn
    x_rows = x_ptr + offs_token[:, None, None, None] * stride_xm
    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        # NVFP4 block (64 values): 4 UE4M3 scales, then per 16-value sub-block s: 8 bytes, low nibbles = values 0-7,
        # high = 8-15 of the sub-block; activations loaded in the same (j, j + 8) interleave
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks
        k_lo = cur_kb[None, :, None, None] * 64 + offs_s[None, None, :, None] * 16 + offs_j[None, None, None, :]
        xm = token_mask[:, None, None, None] & kb_mask[None, :, None, None]
        x_lo = tl.load(x_rows + k_lo * stride_xk, mask=xm, other=0.0)
        x_hi = tl.load(x_rows + (k_lo + 8) * stride_xk, mask=xm, other=0.0)
        x_tile = tl.reshape(tl.join(x_lo, x_hi), (BLOCK_M, BLOCK_K_BLOCKS * 64))
        dt = x_tile.dtype
        blk = w_rows[:, :, None] + cur_kb[None, :, None] * 36                              # [N, KB, 1]
        m3 = n_mask[:, None, None] & kb_mask[None, :, None]
        d = _ue4m3_half(tl.load(blk + offs_s[None, None, :], mask=m3, other=0))            # [N, KB, 4]
        q = tl.load(blk[:, :, :, None] + 4 + offs_s[None, None, :, None] * 8 + offs_j[None, None, None, :],
                    mask=m3[:, :, :, None], other=0).to(tl.int32)                           # [N, KB, 4, 8]
        lo = (_fp4(q & 15) * d[:, :, :, None]).to(dt)
        hi = (_fp4(q >> 4) * d[:, :, :, None]).to(dt)
        w_tile = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K_BLOCKS * 64))
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)
    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


@triton.jit
def q2_0_moe_kernel(x_ptr, w_ptr, y_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                    num_valid_tokens, top_k, n, num_k_blocks, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                    stride_ym, stride_yn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_BLOCKS: tl.constexpr):
    offs_output, offs_token, token_mask, expert, offs_n, n_mask = _moe_prologue(
        sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M, BLOCK_N)
    if expert < 0:
        return
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_b = tl.arange(0, 16)
    shifts = tl.arange(0, 4) * 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_rows = w_ptr + expert.to(tl.int64) * stride_we + offs_n[:, None] * stride_wn
    k_all = tl.arange(0, BLOCK_K_BLOCKS * 64)
    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        # Q2_0 block (64 values): f16 d, 16 bytes of 2-bit codes, value 4 i + s = bits 2 s of byte i, (q - 1) d
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks
        km = (kb_start * 64 + k_all) < num_k_blocks * 64
        x_tile = tl.load(x_ptr + offs_token[:, None] * stride_xm + (kb_start * 64 + k_all)[None, :] * stride_xk,
                         mask=token_mask[:, None] & km[None, :], other=0.0)
        dt = x_tile.dtype
        blk = w_rows + cur_kb[None, :] * 18
        m2 = n_mask[:, None] & kb_mask[None, :]
        d = load_f16_from_u8(blk, m2).to(tl.float32)
        q = tl.load(blk[:, :, None] + 2 + offs_b[None, None, :], mask=m2[:, :, None], other=0).to(tl.int32)
        v = ((q[:, :, :, None] >> shifts[None, None, None, :]) & 3).to(tl.float32) - 1.0     # [N, KB, 16, 4]
        w_tile = tl.reshape((v * d[:, :, None, None]).to(dt), (BLOCK_N, BLOCK_K_BLOCKS * 64))
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)
    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


@triton.jit
def q1_0_moe_kernel(x_ptr, w_ptr, y_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                    num_valid_tokens, top_k, n, num_k_blocks, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                    stride_ym, stride_yn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_BLOCKS: tl.constexpr):
    offs_output, offs_token, token_mask, expert, offs_n, n_mask = _moe_prologue(
        sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M, BLOCK_N)
    if expert < 0:
        return
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_b = tl.arange(0, 16)
    bits = tl.arange(0, 8)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_rows = w_ptr + expert.to(tl.int64) * stride_we + offs_n[:, None] * stride_wn
    k_all = tl.arange(0, BLOCK_K_BLOCKS * 128)
    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        # Q1_0 block (128 values): f16 d, 16 bytes of signs (LSB first): +d / -d
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks
        km = (kb_start * 128 + k_all) < num_k_blocks * 128
        x_tile = tl.load(x_ptr + offs_token[:, None] * stride_xm + (kb_start * 128 + k_all)[None, :] * stride_xk,
                         mask=token_mask[:, None] & km[None, :], other=0.0)
        dt = x_tile.dtype
        blk = w_rows + cur_kb[None, :] * 18
        m2 = n_mask[:, None] & kb_mask[None, :]
        d = load_f16_from_u8(blk, m2).to(tl.float32)
        q = tl.load(blk[:, :, None] + 2 + offs_b[None, None, :], mask=m2[:, :, None], other=0).to(tl.int32)
        sgn = ((q[:, :, :, None] >> bits[None, None, None, :]) & 1).to(tl.float32) * 2.0 - 1.0    # [N, KB, 16, 8]
        w_tile = tl.reshape((sgn * d[:, :, None, None]).to(dt), (BLOCK_N, BLOCK_K_BLOCKS * 128))
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)
    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


@triton.jit
def dense_moe_kernel(x_ptr, w_ptr, y_ptr, scale_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_valid_tokens, top_k,
                     n, k, stride_xm, stride_xk, stride_we, stride_wn, stride_wk, stride_ym, stride_yn,
                     HAS_SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Float or int8 experts W [E, N, K] (converted to the activation dtype in registers); HAS_SCALE: out * scale[e]."""
    offs_output, offs_token, token_mask, expert, offs_n, n_mask = _moe_prologue(
        sorted_token_ids_ptr, expert_ids_ptr, top_k, num_valid_tokens, n, BLOCK_M, BLOCK_N)
    if expert < 0:
        return
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_rows = w_ptr + expert.to(tl.int64) * stride_we + offs_n[:, None] * stride_wn
    offs_k = tl.arange(0, BLOCK_K)
    for k0 in range(0, k, BLOCK_K):
        kk = k0 + offs_k
        km = kk < k
        x_tile = tl.load(x_ptr + offs_token[:, None] * stride_xm + kk[None, :] * stride_xk,
                         mask=token_mask[:, None] & km[None, :], other=0.0)
        w_tile = tl.load(w_rows + kk[None, :] * stride_wk, mask=n_mask[:, None] & km[None, :], other=0)
        acc = tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)), acc=acc)
    if HAS_SCALE:
        acc = acc * tl.load(scale_ptr + expert)
    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


# ---- decode: one routed row per program row-block, no sorting (GEMV; the weights rounded to the activation dtype
# as in the grouped kernels, float32 products and accumulation) --------------------------------------------------

@triton.jit
def _gemv_prologue(ids_ptr, x_ptr, stride_xm, act_div, n, BLOCK_N: tl.constexpr):
    r = tl.program_id(0)
    expert = tl.load(ids_ptr + r).to(tl.int64)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x_row = x_ptr + (r // act_div).to(tl.int64) * stride_xm
    return r, expert, offs_n, offs_n < n, x_row


@triton.jit
def _fp4_f16x(nib):
    """Doubled E2M1 (kvalues_fp4: 0 1 2 3 4 6 8 12, signed) x 2^-15, exactly: the nibble's sign / exponent /
    mantissa bits placed into an F16 (E2M1 subnormal -> F16 subnormal) and widened."""
    bits = ((nib & 8) << 12) | ((nib & 7) << 9)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def nvfp4_moe_gemv(x_ptr, w_ptr, y_ptr, ids_ptr, act_div, n, num_k_blocks, stride_xm, stride_xk, stride_we,
                   stride_wn, stride_ym, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    """w_ptr: the expert bytes viewed as int32 words (strides in words); an NVFP4 block = 9 words: the 4 UE4M3
    scales, then 8 words of nibbles (sub-block s = words 2s, 2s + 1; byte b of word w: low nibble = value
    (w % 2) * 4 + b of the sub-block, high nibble = that + 8).  fp4 x scale is exact in BF16 (as the grouped
    kernel's weights): each word's dot is taken on the raw fp4 values and scaled by its sub-block scale x 2^15."""
    r, expert, offs_n, n_mask, x_row = _gemv_prologue(ids_ptr, x_ptr, stride_xm, act_div, n, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_w = tl.arange(0, 8)
    offs_q = tl.arange(0, 8)
    shifts = offs_q * 4
    # activation index of nibble q of word w inside a 64-value block
    xi = (offs_w[:, None] // 2) * 16 + (offs_w[:, None] % 2) * 4 + offs_q[None, :] // 2 + (offs_q[None, :] % 2) * 8
    acc = tl.zeros((BLOCK_N, BLOCK_KB, 8), dtype=tl.float32)
    w_rows = w_ptr + expert * stride_we + offs_n * stride_wn
    for kb0 in range(0, num_k_blocks, BLOCK_KB):
        cur_kb = kb0 + offs_kb
        kb_mask = cur_kb < num_k_blocks
        x = tl.load(x_row + (cur_kb[:, None, None] * 64 + xi[None, :, :]) * stride_xk,
                    mask=kb_mask[:, None, None], other=0.0).to(tl.float32)                       # [KB, 8, 8]
        blk = w_rows[:, None] + cur_kb[None, :] * 9                                             # [N, KB] (words)
        m2 = n_mask[:, None] & kb_mask[None, :]
        sw = tl.load(blk, mask=m2, other=0)                                                    # 4 UE4M3 scales
        d = _ue4m3_half((sw[:, :, None] >> (offs_w // 2 * 8)[None, None, :]) & 255) * 32768.0  # [N, KB, 8 words]
        q = tl.load(blk[:, :, None] + 1 + offs_w[None, None, :], mask=m2[:, :, None], other=0)   # [N, KB, 8]
        v = _fp4_f16x((q[:, :, :, None] >> shifts[None, None, None, :]) & 15)                 # [N, KB, 8, 8]
        acc += tl.sum(v * x[None], axis=3) * d
    out = tl.sum(tl.sum(acc, axis=2), axis=1)
    tl.store(y_ptr + r.to(tl.int64) * stride_ym + offs_n, out, mask=n_mask)


@triton.jit
def q2_0_moe_gemv(x_ptr, w_ptr, y_ptr, ids_ptr, act_div, n, num_k_blocks, stride_xm, stride_xk, stride_we,
                  stride_wn, stride_ym, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    """w_ptr: the expert bytes viewed as int16 halfwords (strides in halfwords); a Q2_0 block = 9 halfwords: f16 d,
    then 8 halfwords of 2-bit codes (code c of halfword h = value 8 h + c, (q - 1) d).  bf16((q - 1) d) =
    (q - 1) bf16(d) for q - 1 in {-1, 0, 1, 2}: d is rounded to BF16 once per block, so every term (q - 1) (bf16(d)
    x) is the grouped kernel's product, exact in float32; accumulated elementwise and reduced once after the loop
    (measured faster than per-halfword reductions for this format, the reverse of NVFP4)."""
    r, expert, offs_n, n_mask, x_row = _gemv_prologue(ids_ptr, x_ptr, stride_xm, act_div, n, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_h = tl.arange(0, 8)
    offs_c = tl.arange(0, 8)
    acc = tl.zeros((BLOCK_N, BLOCK_KB, 8, 8), dtype=tl.float32)      # elementwise, one reduction after the loop
    w_rows = w_ptr + expert * stride_we + offs_n * stride_wn
    for kb0 in range(0, num_k_blocks, BLOCK_KB):
        cur_kb = kb0 + offs_kb
        kb_mask = cur_kb < num_k_blocks
        x = tl.load(x_row + (cur_kb[:, None, None] * 64 + offs_h[None, :, None] * 8 + offs_c[None, None, :])
                    * stride_xk, mask=kb_mask[:, None, None], other=0.0).to(tl.float32)        # [KB, 8, 8]
        blk = w_rows[:, None] + cur_kb[None, :] * 9                                             # [N, KB] (halfwords)
        m2 = n_mask[:, None] & kb_mask[None, :]
        d = tl.load(blk, mask=m2, other=0).to(tl.float16, bitcast=True).to(tl.float32)
        d = d.to(x_ptr.dtype.element_ty).to(tl.float32)                                         # BF16 weights
        q = tl.load(blk[:, :, None] + 1 + offs_h[None, None, :], mask=m2[:, :, None], other=0).to(tl.int32)
        v = ((q[:, :, :, None] >> (offs_c * 2)[None, None, None, :]) & 3).to(tl.float32) - 1.0  # [N, KB, 8, 8]
        acc += v * (x[None] * d[:, :, None, None])
    out = tl.sum(tl.reshape(acc, (BLOCK_N, BLOCK_KB * 64)), axis=1)
    tl.store(y_ptr + r.to(tl.int64) * stride_ym + offs_n, out, mask=n_mask)


@triton.jit
def mxfp4_moe_gemv(x_ptr, w_ptr, y_ptr, ids_ptr, act_div, n, num_k_blocks, stride_xm, stride_xk, stride_we,
                   stride_wn, stride_ym, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    r, expert, offs_n, n_mask, x_row = _gemv_prologue(ids_ptr, x_ptr, stride_xm, act_div, n, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_b = tl.arange(0, 16)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_rows = w_ptr + expert * stride_we + offs_n * stride_wn
    for kb0 in range(0, num_k_blocks, BLOCK_KB):
        cur_kb = kb0 + offs_kb
        kb_mask = cur_kb < num_k_blocks
        k_lo = cur_kb[:, None] * 32 + offs_b[None, :]                                                   # [KB, 16]
        x_lo = tl.load(x_row + k_lo * stride_xk, mask=kb_mask[:, None], other=0.0)
        x_hi = tl.load(x_row + (k_lo + 16) * stride_xk, mask=kb_mask[:, None], other=0.0)
        dt = x_lo.dtype
        blk = w_rows[:, None] + cur_kb[None, :] * 17
        m2 = n_mask[:, None] & kb_mask[None, :]
        e = tl.load(blk, mask=m2, other=128).to(tl.float32)
        dsc = tl.exp2(e - 128.0)
        q = tl.load(blk[:, :, None] + 1 + offs_b[None, None, :], mask=m2[:, :, None], other=0).to(tl.int32)
        lo = (_fp4(q & 15) * dsc[:, :, None]).to(dt).to(tl.float32)
        hi = (_fp4(q >> 4) * dsc[:, :, None]).to(dt).to(tl.float32)
        prod = lo * x_lo.to(tl.float32)[None] + hi * x_hi.to(tl.float32)[None]
        acc += tl.sum(tl.sum(prod, axis=2), axis=1)
    tl.store(y_ptr + r.to(tl.int64) * stride_ym + offs_n, acc, mask=n_mask)


# type: (kernel, word dtype the expert bytes are viewed as, block size in those words)
_GEMV = {NVFP4: (nvfp4_moe_gemv, torch.int32, 9), Q2_0: (q2_0_moe_gemv, torch.int16, 9),
         MXFP4: (mxfp4_moe_gemv, torch.uint8, 17)}
GEMV_TYPES = frozenset(_GEMV)
# type: (block_n, block_kb, warps, the largest routed-row count served by the GEMV; above it the grouped kernel)
GEMV_TILES = {NVFP4: (32, 2, 4, 64), Q2_0: (32, 1, 1, 128), MXFP4: (16, 4, 1, 64)}
for _item in filter(None, os.environ.get("KAILLM_INKLING_GEMV_TILES", "").split(";")):   # "40=16,2,4,64;..."
    _t, _v = _item.split("=")
    GEMV_TILES[int(_t)] = tuple(int(v) for v in _v.split(","))


def gemv_rows(quant_type: int) -> int:
    """Routed rows up to which moe_new_type_gemv serves this type (0: never)."""
    return GEMV_TILES[quant_type][3] if quant_type in GEMV_TILES else 0


def moe_new_type_gemv(X, W, ids, quant_type: int, row: int, act_div: int, out_dtype=torch.float32,
                      block_n: int | None = None, block_kb: int | None = None, num_warps: int | None = None):
    """Decode path: X [rows / act_div, K] (bf16 / f16), ids int32 [rows] expert of each routed row (row r reads
    activation row r // act_div), W uint8 [E, row, K / qk * block bytes] -> [rows, row] float32; no sorting.
    block_n / block_kb / num_warps: tile overrides (benchmarks)."""
    kernel, word, words = _GEMV[quant_type]
    bn, bkb, nw = GEMV_TILES[quant_type][:3]
    bn, bkb, nw = block_n or bn, block_kb or bkb, num_warps or nw
    qk, size = BLOCK[quant_type]
    W = W.reshape(W.shape[0], row, -1)
    num_k_blocks = W.shape[2] // size
    if num_k_blocks * qk != X.shape[1]:
        raise ValueError(f"expert width {num_k_blocks * qk} != activation width {X.shape[1]}")
    if word != torch.uint8:
        W = W.view(word)                    # row bytes are a multiple of the word size (36 / 18-byte blocks)
    rows = ids.numel()
    out = torch.empty((rows, row), device=X.device, dtype=torch.float32)
    if rows == 0:
        return out if out_dtype == torch.float32 else out.to(out_dtype)
    kernel[(rows, triton.cdiv(row, bn))](
        X, W, out, ids, act_div, row, num_k_blocks, X.stride(0), X.stride(1), W.stride(0), W.stride(1),
        out.stride(0), BLOCK_N=bn, BLOCK_KB=bkb, num_warps=nw, num_stages=2)
    return out if out_dtype == torch.float32 else out.to(out_dtype)


_KERNELS = {MXFP4: (mxfp4_moe_kernel, 4), NVFP4: (nvfp4_moe_kernel, 2), Q2_0: (q2_0_moe_kernel, 2),
            Q1_0: (q1_0_moe_kernel, 1)}
BLOCK_M = 16
BLOCK_N = 64


def moe_block_m() -> int:
    return BLOCK_M


def _grid(expert_ids, row, block_n):
    return (expert_ids.shape[0], triton.cdiv(row, block_n))


def _pad_sorted(sorted_token_ids, expert_ids, block_m):
    need = expert_ids.numel() * block_m
    if sorted_token_ids.numel() < need:
        pad = torch.full((need - sorted_token_ids.numel(),), -1, dtype=sorted_token_ids.dtype,
                         device=sorted_token_ids.device)
        sorted_token_ids = torch.cat([sorted_token_ids, pad])
    return sorted_token_ids.reshape(-1).contiguous()


def moe_new_type(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type: int, row: int, top_k: int,
                 tokens: int, out_dtype=None):
    """X [tokens, K] (bf16 / f16), W uint8 [E, row, K / qk * block bytes] -> [tokens * top_k, row]."""
    kernel, bkb = _KERNELS[quant_type]
    qk, size = BLOCK[quant_type]
    W = W.reshape(W.shape[0], row, -1)
    num_k_blocks = W.shape[2] // size
    if num_k_blocks * qk != X.shape[1]:
        raise ValueError(f"expert width {num_k_blocks * qk} != activation width {X.shape[1]}")
    out = torch.zeros((tokens * top_k, row), device=X.device, dtype=out_dtype or X.dtype)
    sorted_token_ids = _pad_sorted(sorted_token_ids, expert_ids, BLOCK_M)
    kernel[_grid(expert_ids, row, BLOCK_N)](
        X, W, out, sorted_token_ids, expert_ids.reshape(-1), num_tokens_post_padded, tokens * top_k, top_k, row,
        num_k_blocks, X.stride(0), X.stride(1), W.stride(0), W.stride(1), W.stride(2), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K_BLOCKS=bkb, num_warps=4, num_stages=2)
    return out


def moe_dense(X, W, sorted_token_ids, expert_ids, row: int, top_k: int, tokens: int, scale=None, out_dtype=None):
    """W [E, row, K] float or int8; scale [E] (per-expert multiplier, int8 ITQ prefill) -> [tokens * top_k, row]."""
    out = torch.zeros((tokens * top_k, row), device=X.device, dtype=out_dtype or X.dtype)
    sorted_token_ids = _pad_sorted(sorted_token_ids, expert_ids, BLOCK_M)
    dense_moe_kernel[_grid(expert_ids, row, BLOCK_N)](
        X, W, out, scale if scale is not None else out, sorted_token_ids, expert_ids.reshape(-1), tokens * top_k,
        top_k, row, X.shape[1], X.stride(0), X.stride(1), W.stride(0), W.stride(1), W.stride(2), out.stride(0),
        out.stride(1), HAS_SCALE=scale is not None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=64, num_warps=4,
        num_stages=2)
    return out
