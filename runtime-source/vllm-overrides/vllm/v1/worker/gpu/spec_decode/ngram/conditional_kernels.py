# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _invalid_drafts(
    out_ptr, count_ptr, local_pos_ptr, N: tl.constexpr, B: tl.constexpr
):
    offsets = tl.arange(0, B)
    count = tl.load(count_ptr)
    position = tl.load(local_pos_ptr + offsets, offsets < N, other=0)
    tl.store(out_ptr + offsets, position > count, offsets < N)


def invalid_drafts(counts, local_positions):
    size = local_positions.numel()
    mask = torch.empty_like(local_positions, dtype=torch.bool)
    _invalid_drafts[(1,)](
        mask,
        counts,
        local_positions,
        N=size,
        B=triton.next_power_of_2(size),
        num_warps=1,
    )
    return mask


@triton.jit
def _copy_hidden(
    source_ptr,
    output_ptr,
    source_stride0,
    source_stride1,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = offsets // WIDTH, offsets % WIDTH
    values = tl.load(
        source_ptr + row * source_stride0 + column * source_stride1,
        mask=(row < ROWS) & (offsets < 4 * WIDTH),
        other=0,
    )
    tl.store(output_ptr + offsets, values, mask=offsets < 4 * WIDTH)


def copy_hidden(source, output):
    assert source.dtype == output.dtype and source.dtype in (torch.float32, torch.bfloat16)
    assert source.ndim == output.ndim == 2
    assert source.shape[0] in (1, 4) and output.shape[0] == 4
    assert source.shape[1] == output.shape[1] and output.is_contiguous()
    width = source.shape[1]
    _copy_hidden[(triton.cdiv(4 * width, 256),)](
        source,
        output,
        source.stride(0),
        source.stride(1),
        ROWS=source.shape[0],
        WIDTH=width,
        BLOCK=256,
        num_warps=4,
    )


@triton.jit
def _copy_projection(
    source_ptr,
    output_ptr,
    source_stride0,
    source_stride1,
    output_stride0,
    output_stride1,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = offsets // WIDTH, offsets % WIDTH
    values = tl.load(
        source_ptr + row * source_stride0 + column * source_stride1,
        mask=offsets < ROWS * WIDTH,
        other=0,
    )
    tl.store(
        output_ptr + row * output_stride0 + column * output_stride1,
        values,
        mask=offsets < ROWS * WIDTH,
    )


def copy_projection(source, output):
    assert source.dtype == output.dtype and source.dtype in (torch.float32, torch.bfloat16)
    assert source.ndim == output.ndim == 2 and source.shape == output.shape
    rows, width = source.shape
    _copy_projection[(triton.cdiv(rows * width, 256),)](
        source,
        output,
        source.stride(0),
        source.stride(1),
        output.stride(0),
        output.stride(1),
        ROWS=rows,
        WIDTH=width,
        BLOCK=256,
        num_warps=4,
    )
