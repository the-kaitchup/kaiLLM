# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF wire decoders shared by compact GEMV/GEMM and embedding lookup.

Only requested elements are reconstructed, in FP32, inside the consuming kernel.
The standard and IQ formulas match the standalone Triton dequantization kernels.
"""

import triton
import triton.language as tl

from ..triton.dequantize.utils import (
    load_q3_k_scale_vector,
    load_scale_min_k4_vector,
)
from ..triton.gemm.utils import (
    load_f16_from_u8,
    load_u16_from_u8,
    load_u32_from_u8,
)


@triton.jit
def _load_q4_0(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 32) * 18
    pos = offs % 32
    packed = tl.load(block_ptrs + 2 + (pos % 16), mask=mask, other=0)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    out = (q.to(tl.float32) - 8.0) * d
    return out


@triton.jit
def _load_q4_1(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 32) * 20
    pos = offs % 32
    packed = tl.load(block_ptrs + 4 + (pos % 16), mask=mask, other=0)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    m0 = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q.to(tl.float32) * d + m0
    return out


@triton.jit
def _load_q5_0(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 32) * 22
    pos = offs % 32
    packed = tl.load(block_ptrs + 6 + (pos % 16), mask=mask, other=0)
    qh = load_u32_from_u8(block_ptrs + 2, mask)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    q = q | (((qh >> pos) & 0x01) << 4)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    out = (q.to(tl.float32) - 16.0) * d
    return out


@triton.jit
def _load_q5_1(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 32) * 24
    pos = offs % 32
    packed = tl.load(block_ptrs + 8 + (pos % 16), mask=mask, other=0)
    qh = load_u32_from_u8(block_ptrs + 4, mask)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    q = q | (((qh >> pos) & 0x01) << 4)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    m0 = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q.to(tl.float32) * d + m0
    return out


@triton.jit
def _load_q8_0(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 32) * 34
    pos = offs % 32
    q = tl.load(block_ptrs + 2 + pos, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    out = tl.cast(q, tl.int8, bitcast=True).to(tl.float32) * d
    return out


@triton.jit
def _load_q2_k(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 256) * 84
    pos = offs % 256
    group = pos // 128
    rem = pos % 128
    part = rem // 32
    q_idx = rem % 32

    scale_idx = 8 * group + 2 * part + tl.where(q_idx >= 16, 1, 0)
    q = tl.load(block_ptrs + 16 + 32 * group + q_idx, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + scale_idx, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 80, mask).to(tl.float32)
    dmin = load_f16_from_u8(block_ptrs + 82, mask).to(tl.float32)
    scale = (scale_byte & 0x0F).to(tl.float32) * d
    minv = (scale_byte >> 4).to(tl.float32) * dmin
    out = (((q >> (2 * part)) & 0x03).to(tl.float32) * scale) - minv
    return out


@triton.jit
def _load_q3_k(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 256) * 110
    pos = offs % 256
    group = pos // 128
    rem = pos % 128
    j = rem // 32
    half = (rem % 32) // 16
    sub_idx = rem % 16

    is_idx = 8 * group + 2 * j + half
    hm_mask = 1 << (4 * group + j)
    scale = load_q3_k_scale_vector(block_ptrs + 96, is_idx, mask)
    ql = tl.load(block_ptrs + 32 + 32 * group + 16 * half + sub_idx, mask=mask, other=0)
    hm = tl.load(block_ptrs + 16 * half + sub_idx, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 108, mask).to(tl.float32)
    q = ((ql >> (2 * j)) & 0x03).to(tl.int16)
    q = (q - tl.where((hm & hm_mask) != 0, 0, 4).to(tl.int16)).to(tl.float32)
    out = q * ((scale.to(tl.int16) - 32).to(tl.float32) * d)
    return out


@triton.jit
def _load_q4_k(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 256) * 144
    pos = offs % 256
    group = pos // 64
    rem = pos % 64
    part = rem // 32
    q_idx = rem % 32
    j = 2 * group + part

    scale_q, min_q = load_scale_min_k4_vector(block_ptrs + 4, j, mask)
    q = tl.load(block_ptrs + 16 + 32 * group + q_idx, mask=mask, other=0)
    q = tl.where(part == 0, q & 0x0F, q >> 4)
    dall = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dmin = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q.to(tl.float32) * (scale_q.to(tl.float32) * dall)
    out = out - min_q.to(tl.float32) * dmin
    return out


@triton.jit
def _load_q5_k(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 256) * 176
    pos = offs % 256
    group = pos // 64
    rem = pos % 64
    part = rem // 32
    q_idx = rem % 32
    j = 2 * group + part

    scale_q, min_q = load_scale_min_k4_vector(block_ptrs + 4, j, mask)
    qh = tl.load(block_ptrs + 16 + q_idx, mask=mask, other=0)
    q = tl.load(block_ptrs + 48 + 32 * group + q_idx, mask=mask, other=0)
    q = tl.where(part == 0, q & 0x0F, q >> 4)
    q = q.to(tl.float32) + 16.0 * ((qh & (1 << (2 * group + part))) != 0).to(tl.float32)
    dall = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dmin = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q * (scale_q.to(tl.float32) * dall)
    out = out - min_q.to(tl.float32) * dmin
    return out


@triton.jit
def _load_q6_k(w_ptr, offs, mask):
    block_ptrs = w_ptr + (offs // 256) * 210
    pos = offs % 256
    half_block = pos // 128
    rem = pos % 128
    group = rem // 32
    half = (rem % 32) // 16
    sub_idx = rem % 16
    il = 16 * half + sub_idx

    scale = tl.load(
        block_ptrs + 192 + 8 * half_block + 2 * group + half,
        mask=mask,
        other=0,
    )
    d = load_f16_from_u8(block_ptrs + 208, mask).to(tl.float32)
    scale = tl.cast(scale, tl.int8, bitcast=True).to(tl.float32) * d
    qh = tl.load(block_ptrs + 128 + 32 * half_block + il, mask=mask, other=0)
    ql_idx = 64 * half_block + tl.where((group % 2) == 1, 32, 0) + il
    ql = tl.load(block_ptrs + ql_idx, mask=mask, other=0)
    q = tl.where(group < 2, ql & 0x0F, ql >> 4)
    q = q.to(tl.float32) + 16.0 * (((qh >> (2 * group)) & 0x03).to(tl.float32))
    out = (q - 32.0) * scale
    return out


@triton.jit
def _load_iq1_m(w_ptr, offs, mask, grid_ptr):
    block_ptrs = w_ptr + (offs // 256) * 56
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    sc0 = load_u16_from_u8(block_ptrs + 48 + 0, mask)
    sc1 = load_u16_from_u8(block_ptrs + 48 + 2, mask)
    sc2 = load_u16_from_u8(block_ptrs + 48 + 4, mask)
    sc3 = load_u16_from_u8(block_ptrs + 48 + 6, mask)
    base_bits = (
        (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
    )
    qh = tl.load(block_ptrs + 32 + 2 * ib + (il // 2), mask=mask, other=0).to(tl.int32)
    idx = tl.load(block_ptrs + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    idx = idx | (((qh >> (4 * (il % 2))) & 0x07) << 8)
    delta_num = tl.where((qh & (0x08 << (4 * (il % 2)))) != 0, -9.0, -7.0)
    ib16 = 2 * ib + (il // 2)
    sc_sel = tl.where(
        ib16 // 4 == 0,
        sc0,
        tl.where(ib16 // 4 == 1, sc1, tl.where(ib16 // 4 == 2, sc2, sc3)),
    )
    base = tl.cast(base_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
    scale = (
        base
        * (2.0 * (((sc_sel >> (3 * (ib16 % 4))) & 0x07).to(tl.float32)) + 1.0)
        * 0.125
    )
    grid = tl.load(grid_ptr + idx * 8 + k, mask=mask, other=0).to(tl.float32)
    out = ((grid * 8.0) + delta_num) * scale
    return out


@triton.jit
def _load_iq1_s(w_ptr, offs, mask, grid_ptr):
    block_ptrs = w_ptr + (offs // 256) * 50
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    qh = load_u16_from_u8(block_ptrs + 34 + 2 * ib, mask)
    idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    idx = idx | ((((qh >> (3 * il)) & 0x7).to(tl.int32)) << 8)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = d * (2.0 * ((qh >> 12) & 0x7).to(tl.float32) + 1.0) * 0.125
    delta_num = tl.where((qh & 0x8000) != 0, -9.0, -7.0)
    grid = tl.load(grid_ptr + idx * 8 + k, mask=mask, other=0).to(tl.float32)
    out = ((grid * 8.0) + delta_num) * scale
    return out


@triton.jit
def _load_iq2_s(w_ptr, offs, mask, grid_ptr):
    block_ptrs = w_ptr + (offs // 256) * 82
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    qh = tl.load(block_ptrs + 66 + ib, mask=mask, other=0).to(tl.int32)
    grid_idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    grid_idx = grid_idx | ((qh << (8 - 2 * il)) & 0x300)
    signs = tl.load(block_ptrs + 34 + 4 * ib + il, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + 74 + ib, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = (((scale_byte >> (4 * (il // 2))) & 0x0F).to(tl.float32) + 0.5) * 0.25
    grid = tl.load(grid_ptr + grid_idx * 8 + k, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * d * scale
    return out


@triton.jit
def _load_iq2_xs(w_ptr, offs, mask, grid_ptr, sign_ptr):
    block_ptrs = w_ptr + (offs // 256) * 74
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    q2 = load_u16_from_u8(block_ptrs + 2 + 2 * (4 * ib + il), mask).to(tl.int32)
    grid_idx = q2 & 0x1FF
    signs = tl.load(sign_ptr + (q2 >> 9), mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + 66 + ib, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = (((scale_byte >> (4 * (il // 2))) & 0x0F).to(tl.float32) + 0.5) * 0.25
    grid = tl.load(grid_ptr + grid_idx * 8 + k, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * d * scale
    return out


@triton.jit
def _load_iq2_xxs(w_ptr, offs, mask, grid_ptr, sign_ptr):
    block_ptrs = w_ptr + (offs // 256) * 66
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    q2_base = block_ptrs + 2 + 8 * ib
    aux32 = load_u32_from_u8(q2_base + 4, mask)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dscale = d * ((aux32 >> 28).to(tl.float32) + 0.5) * 0.25
    grid_idx = tl.load(q2_base + il, mask=mask, other=0).to(tl.int32)
    signs = tl.load(sign_ptr + ((aux32 >> (7 * il)) & 127), mask=mask, other=0)
    grid = tl.load(grid_ptr + grid_idx * 8 + k, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * dscale
    return out


@triton.jit
def _load_iq3_s(w_ptr, offs, mask, grid_ptr):
    block_ptrs = w_ptr + (offs // 256) * 110
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8
    half = k // 4
    k4 = k % 4

    qh = tl.load(block_ptrs + 66 + ib, mask=mask, other=0).to(tl.int32)
    idx = tl.load(block_ptrs + 2 + 8 * ib + 2 * il + half, mask=mask, other=0).to(
        tl.int32
    )
    idx = idx | tl.where(
        half == 0,
        (qh << (8 - 2 * il)) & 0x100,
        (qh << (7 - 2 * il)) & 0x100,
    )
    signs = tl.load(block_ptrs + 74 + 4 * ib + il, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + 106 + (ib // 2), mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = (scale_byte >> (4 * (ib % 2))) & 0x0F
    dscale = d * (1.0 + 2.0 * scale.to(tl.float32))
    grid = tl.load(grid_ptr + idx * 4 + k4, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * dscale
    return out


@triton.jit
def _load_iq3_xxs(w_ptr, offs, mask, grid_ptr, sign_ptr):
    block_ptrs = w_ptr + (offs // 256) * 98
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8
    half = k // 4
    k4 = k % 4

    aux32 = load_u32_from_u8(block_ptrs + 66 + 4 * ib, mask)
    idx = tl.load(block_ptrs + 2 + 8 * ib + 2 * il + half, mask=mask, other=0).to(
        tl.int32
    )
    signs = tl.load(sign_ptr + ((aux32 >> (7 * il)) & 127), mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dscale = d * ((aux32 >> 28).to(tl.float32) + 0.5) * 0.5
    grid = tl.load(grid_ptr + idx * 4 + k4, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * dscale
    return out


@triton.jit
def _load_iq4_nl(w_ptr, offs, mask, values_ptr):
    block_ptrs = w_ptr + (offs // 32) * 18
    pos = offs % 32
    packed = tl.load(block_ptrs + 2 + (pos % 16), mask=mask, other=0)
    nibble = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    value = tl.load(values_ptr + nibble.to(tl.int32), mask=mask, other=0)
    out = value.to(tl.float32) * d
    return out


@triton.jit
def _load_iq4_xs(w_ptr, offs, mask, values_ptr):
    block_ptrs = w_ptr + (offs // 256) * 136
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    packed = tl.load(block_ptrs + 8 + 16 * ib + (rem % 16), mask=mask, other=0)
    nibble = tl.where(rem < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    scales_h = load_u16_from_u8(block_ptrs + 2, mask)
    scales_l = tl.load(block_ptrs + 4 + (ib // 2), mask=mask, other=0).to(tl.int32)
    scale = (
        ((scales_l >> (4 * (ib % 2))) & 0x0F)
        | (((scales_h.to(tl.int32) >> (2 * ib)) & 0x03) << 4)
    ).to(tl.int16)
    scale = (scale - 32).to(tl.float32)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    value = tl.load(values_ptr + nibble.to(tl.int32), mask=mask, other=0)
    out = value.to(tl.float32) * d * scale
    return out


@triton.jit
def _load_extra(w_ptr, offs, mask, values_ptr, QTYPE: tl.constexpr):
    if QTYPE == 9:
        block = w_ptr + (offs // 32) * 36
        code = tl.load(block + 4 + offs % 32, mask=mask, other=0)
        scale = load_f16_from_u8(block, mask).to(tl.float32)
        out = code.to(tl.int8, bitcast=True).to(tl.float32) * scale
    elif QTYPE == 15:
        block = w_ptr + (offs // 256) * 292
        code = tl.load(block + 4 + offs % 256, mask=mask, other=0)
        scale = load_u32_from_u8(block, mask).to(tl.float32, bitcast=True)
        out = code.to(tl.int8, bitcast=True).to(tl.float32) * scale
    elif QTYPE == 34:
        block = w_ptr + (offs // 256) * 54
        pos = offs % 256
        index = tl.where(pos < 160, pos % 32, 32 + (pos - 160) % 16)
        index = tl.where(pos >= 240, 48 + (pos - 240) % 4, index)
        power = tl.where(pos < 160, pos // 32, (pos - 160) // 16)
        power = tl.where(pos >= 240, (pos - 240) // 4, power)
        factor = tl.where(
            power == 0,
            1,
            tl.where(
                power == 1, 3, tl.where(power == 2, 9, tl.where(power == 3, 27, 81))
            ),
        )
        raw = tl.load(block + index, mask=mask, other=0).to(tl.uint32)
        trit = (((raw * factor) & 255) * 3) >> 8
        out = (trit.to(tl.float32) - 1) * load_f16_from_u8(block + 52, mask)
    elif QTYPE == 35:
        block = w_ptr + (offs // 256) * 66
        pos = offs % 256
        raw = tl.load(block + (pos // 128) * 32 + pos % 32, mask=mask, other=0)
        trit = (raw >> (2 * ((pos % 128) // 32))) & 3
        out = (trit.to(tl.float32) - 1) * load_f16_from_u8(block + 64, mask)
    elif QTYPE == 39:
        block = w_ptr + (offs // 32) * 17
        pos = offs % 32
        raw = tl.load(block + 1 + pos % 16, mask=mask, other=0)
        code = tl.where(pos < 16, raw & 15, raw >> 4)
        exponent = tl.load(block, mask=mask, other=0).to(tl.uint32)
        scale_bits = tl.where(
            exponent < 2, 0x00200000 << exponent, (exponent - 1) << 23
        ).to(tl.uint32)
        scale = scale_bits.to(tl.float32, bitcast=True)
        value = tl.load(values_ptr + code, mask=mask, other=0).to(tl.float32)
        out = value * scale
    elif QTYPE == 40:
        block = w_ptr + (offs // 64) * 36
        pos = offs % 64
        raw = tl.load(block + 4 + (pos // 16) * 8 + pos % 8, mask=mask, other=0)
        code = tl.where(pos % 16 < 8, raw & 15, raw >> 4)
        encoded_scale = tl.load(block + pos // 16, mask=mask, other=0).to(tl.int32)
        exponent = (encoded_scale >> 3) & 15
        mantissa = (encoded_scale & 7).to(tl.float32)
        scale = tl.where(
            exponent == 0,
            mantissa * (2.0**-9),
            (1 + mantissa / 8) * tl.exp2(exponent.to(tl.float32) - 7),
        )
        scale = tl.where((encoded_scale == 0) | (encoded_scale == 127), 0, scale)
        value = tl.load(values_ptr + code, mask=mask, other=0).to(tl.float32)
        out = value * scale * 0.5
    elif QTYPE == 41:  # Q1_0
        block = w_ptr + (offs // 128) * 18
        pos = offs % 128
        raw = tl.load(block + 2 + pos // 8, mask=mask, other=0)
        sign = 2 * ((raw >> (pos % 8)) & 1).to(tl.float32) - 1
        out = sign * load_f16_from_u8(block, mask)
    elif QTYPE == 42:  # Q2_0
        block = w_ptr + (offs // 64) * 18
        pos = offs % 64
        raw = tl.load(block + 2 + pos // 4, mask=mask, other=0)
        code = (raw >> (2 * (pos % 4))) & 3
        out = (code.to(tl.float32) - 1) * load_f16_from_u8(block, mask)
    else:
        tl.static_assert(False, "Unsupported compact GGUF wire type")
    return out.to(tl.float32)


@triton.jit
def load_gguf(w_ptr, offs, mask, grid_ptr, sign_ptr, values_ptr, QTYPE: tl.constexpr):
    if QTYPE == 2:
        out = _load_q4_0(w_ptr, offs, mask)
    elif QTYPE == 3:
        out = _load_q4_1(w_ptr, offs, mask)
    elif QTYPE == 6:
        out = _load_q5_0(w_ptr, offs, mask)
    elif QTYPE == 7:
        out = _load_q5_1(w_ptr, offs, mask)
    elif QTYPE == 8:
        out = _load_q8_0(w_ptr, offs, mask)
    elif QTYPE == 10:
        out = _load_q2_k(w_ptr, offs, mask)
    elif QTYPE == 11:
        out = _load_q3_k(w_ptr, offs, mask)
    elif QTYPE == 12:
        out = _load_q4_k(w_ptr, offs, mask)
    elif QTYPE == 13:
        out = _load_q5_k(w_ptr, offs, mask)
    elif QTYPE == 14:
        out = _load_q6_k(w_ptr, offs, mask)
    elif QTYPE == 16:
        out = _load_iq2_xxs(w_ptr, offs, mask, grid_ptr, sign_ptr)
    elif QTYPE == 17:
        out = _load_iq2_xs(w_ptr, offs, mask, grid_ptr, sign_ptr)
    elif QTYPE == 18:
        out = _load_iq3_xxs(w_ptr, offs, mask, grid_ptr, sign_ptr)
    elif QTYPE == 19:
        out = _load_iq1_s(w_ptr, offs, mask, grid_ptr)
    elif QTYPE == 20:
        out = _load_iq4_nl(w_ptr, offs, mask, values_ptr)
    elif QTYPE == 21:
        out = _load_iq3_s(w_ptr, offs, mask, grid_ptr)
    elif QTYPE == 22:
        out = _load_iq2_s(w_ptr, offs, mask, grid_ptr)
    elif QTYPE == 23:
        out = _load_iq4_xs(w_ptr, offs, mask, values_ptr)
    elif QTYPE == 29:
        out = _load_iq1_m(w_ptr, offs, mask, grid_ptr)
    else:
        out = _load_extra(w_ptr, offs, mask, values_ptr, QTYPE)
    return tl.where(mask, out, 0.0)
