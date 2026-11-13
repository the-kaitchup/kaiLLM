# SPDX-License-Identifier: Apache-2.0
"""GPU embedding lookup from packed integers without dense dequantization.

This module is imported only for the additional GGUF formats lacking a native
embedding kernel. Loading repacks on CPU once; inference reads and decodes only
the requested vocabulary rows on GPU. Metadata remains float32 for GGUF parity.
"""

import torch
import triton
import triton.language as tl

from .extra_repack import EXTRA_FORMATS
from .gguf_repack import Q8_1, repack_gguf

SUPPORTED_PACKED_EMBEDDING_TYPES = frozenset(EXTRA_FORMATS) | {Q8_1}


@triton.jit
def _packed_embedding_gather(
    ids_ptr,
    weight_ptr,
    scale_ptr,
    zero_ptr,
    output_ptr,
    total_elements,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total_elements
    row = tl.load(ids_ptr + offsets // HIDDEN, mask=active, other=0)
    column = offsets % HIDDEN
    valid = active & (row >= 0) & (row < ROWS)
    bit_position = column * BITS
    word_position = row * (HIDDEN * BITS // 32) + bit_position // 32
    shift = bit_position % 32
    low = tl.load(weight_ptr + word_position, mask=valid, other=0).to(tl.uint32)
    crosses = shift + BITS > 32
    high = tl.load(weight_ptr + word_position + 1, mask=valid & crosses, other=0).to(
        tl.uint32
    )
    # Avoid an undefined 32-bit shift for non-crossing words.
    high_shift = tl.where(crosses, 32 - shift, 0)
    code = ((low >> shift) | tl.where(crosses, high << high_shift, 0)) & (
        (1 << BITS) - 1
    )
    group_position = row * (HIDDEN // GROUP_SIZE) + column // GROUP_SIZE
    scale = tl.load(scale_ptr + group_position, mask=valid, other=0)
    if HAS_ZERO:
        zero = tl.load(zero_ptr + group_position, mask=valid, other=0)
    else:
        zero = 1 << (BITS - 1)
    value = (code.to(tl.float32) - zero) * scale
    tl.store(output_ptr + offsets, tl.where(valid, value, 0), mask=active)


class PackedGGUFEmbedding(torch.nn.Module):
    def __init__(self, packed, device, dtype):
        super().__init__()
        self.shape = packed.shape
        self.bits = packed.bits
        self.group_size = packed.group_size
        self.dtype = dtype
        self.register_buffer("weight", torch.from_numpy(packed.weight).to(device))
        self.register_buffer("scales", torch.from_numpy(packed.scales).to(device))
        self.register_buffer(
            "zero_points",
            torch.from_numpy(packed.zero_points).to(device)
            if packed.zero_points is not None
            else None,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Packed embedding indices must be int32 or int64")
        if indices.device != self.weight.device:
            raise ValueError("Packed embedding indices and weights must share a device")
        indices = indices.contiguous()
        output = torch.empty(
            (*indices.shape, self.shape[1]), device=indices.device, dtype=self.dtype
        )
        total = output.numel()
        if total == 0:
            return output
        _packed_embedding_gather[(triton.cdiv(total, 256),)](
            indices,
            self.weight,
            self.scales,
            self.zero_points,
            output,
            total,
            ROWS=self.shape[0],
            HIDDEN=self.shape[1],
            BITS=self.bits,
            GROUP_SIZE=self.group_size,
            HAS_ZERO=self.zero_points is not None,
            BLOCK=256,
        )
        return output


def prepare_packed_embedding(
    data: torch.Tensor, quant_type: int, dtype: torch.dtype
) -> PackedGGUFEmbedding:
    """Prepare a local vocabulary partition for GPU-only gathered decoding."""
    if int(quant_type) not in SUPPORTED_PACKED_EMBEDDING_TYPES:
        raise ValueError(f"Generic packed embedding does not handle type {quant_type}")
    if data.dtype != torch.uint8 or data.ndim != 2 or not data.is_cuda:
        raise ValueError("Packed embedding needs a two-dimensional CUDA uint8 tensor")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Packed embedding output must be FP16, BF16, or FP32")
    packed = repack_gguf(data.detach().cpu().numpy(), int(quant_type))
    return PackedGGUFEmbedding(packed, data.device, dtype)
