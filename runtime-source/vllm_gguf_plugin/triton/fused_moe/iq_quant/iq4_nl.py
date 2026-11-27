# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gemm.utils import (
    GGML_TYPE_IQ4_NL,
    load_f16_from_u8,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_tile,
    run_triton_fused_moe_kernel,
)


@triton.jit
def iq4nl_register_value(code):
    # prmt indexes one of eight bytes. The two registers in each pair
    # concatenate to spans 0..7 and 8..15 of GGML's signed IQ4_NL codebook.
    return tl.inline_asm_elementwise(
        asm="""{
            .reg .b32 ix, lo, hi, value;
            .reg .pred low_bank;
            and.b32 ix, $1, 7;
            prmt.b32 lo, 0xbfad9881, 0xf6eaddcf, ix;
            prmt.b32 hi, 0x26190d01, 0x71594535, ix;
            setp.lt.u32 low_bank, $1, 8;
            selp.b32 value, lo, hi, low_bank;
            cvt.s32.s8 $0, value;
        }""",
        constraints="=r,r", args=[code.to(tl.uint32)], dtype=tl.int32,
        is_pure=True, pack=1,
    )


def _iq4_kernel_name(specialization):
    # The physical CUPTI trace can identify tiled prefill without host callbacks.
    tile = specialization.constants["BLOCK_M"]
    if tile in (16, 32):
        if specialization.constants["BLOCK_K_BLOCKS"] == 2 and specialization.constants["BLOCK_N"] == 64:
            return f"iq4_nl_moe_kernel_m{tile}_n64_k64"
        return f"iq4_nl_moe_kernel_m{tile}"
    if tile == 4 and specialization.constants["BLOCK_N"] in (16, 32):
        return f"iq4_nl_moe_kernel_m4_n{specialization.constants['BLOCK_N']}"
    return "iq4_nl_moe_kernel"


@triton.jit(repr=_iq4_kernel_name)
def iq4_nl_moe_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    top_k,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    values_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
    ORIGINAL_TABLE: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_output, offs_token, token_mask = load_moe_token_info(
        sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens, BLOCK_M=BLOCK_M
    )
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_nibble = tl.arange(0, 16)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        x_tile, cur_kb, kb_mask = load_moe_x_tile(
            x_ptr,
            num_k_blocks,
            stride_xm,
            stride_xk,
            offs_token,
            token_mask,
            kb_start,
            offs_kb,
            offs_nibble,
            BLOCK_M=BLOCK_M,
            BLOCK_K_BLOCKS=BLOCK_K_BLOCKS,
        )
        x_dtype = x_tile.dtype
        block_ptrs = w_row_ptrs + cur_kb[None, :] * 18
        scale_mask = n_mask[:, None] & kb_mask[None, :]
        d = load_f16_from_u8(block_ptrs + 0, scale_mask).to(x_dtype)
        packed = tl.load(
            block_ptrs[:, :, None] + 2 + offs_nibble[None, None, :],
            mask=n_mask[:, None, None] & kb_mask[None, :, None],
            other=0,
        )
        if not ORIGINAL_TABLE and ((BLOCK_N == 64 and BLOCK_K_BLOCKS == 2) or (BLOCK_M == 4 and BLOCK_N == 16 and BLOCK_K_BLOCKS == 4)):
            if values_ptr is None:
                lo_code = iq4nl_register_value(packed & 0x0F)
                hi_code = iq4nl_register_value(packed >> 4)
            else:
                lo_code = tl.load(values_ptr + (packed & 0x0F).to(tl.int32))
                hi_code = tl.load(values_ptr + (packed >> 4).to(tl.int32))
            low = lo_code.to(x_dtype) * d[:, :, None]
            high = hi_code.to(x_dtype) * d[:, :, None]
        else:
            low = (
                tl.load(values_ptr + (packed & 0x0F).to(tl.int32)).to(x_dtype)
                * d[:, :, None]
            )
            high = (
                tl.load(values_ptr + ((packed >> 4) & 0x0F).to(tl.int32)).to(x_dtype)
                * d[:, :, None]
            )
        q_tile = tl.reshape(tl.join(low, high), (BLOCK_N, BLOCK_K_BLOCKS * 32))
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_iq4_nl_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_fused_moe_kernel(
        iq4_nl_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ4_NL,
        extra_args=(tables["kvalues_iq4nl"],),
    )
