# SPDX-License-Identifier: Apache-2.0
"""Expand GGML importance codebooks into exact Humming integer weights.

Only integer codewords are expanded; floating point values are computed for
group scales, never for the dense weight matrix. This preserves every GGUF
codebook value, but increases runtime storage: IQ1 and IQ3_S become INT5, IQ2
becomes INT7, IQ3_XXS becomes INT6, and IQ4 becomes INT8. Native compressed IQ
storage would require dedicated codebook-aware kernels.

The block layouts and codebook constants follow ``gguf.quants`` (MIT license).
Codebooks are read from that dependency rather than duplicated here.
"""

from functools import cache, lru_cache

import numpy as np

IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S = 16, 17, 18, 19
IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M = 20, 21, 22, 23, 29

# Humming bits, group size, GGML block elements, GGML block bytes.
IQ_FORMATS = {
    IQ2_XXS: (7, 32, 256, 66),
    IQ2_XS: (7, 16, 256, 74),
    IQ3_XXS: (6, 32, 256, 98),
    IQ1_S: (5, 32, 256, 50),
    IQ4_NL: (8, 32, 32, 18),
    IQ3_S: (5, 32, 256, 110),
    IQ2_S: (7, 16, 256, 82),
    IQ4_XS: (8, 32, 256, 136),
    IQ1_M: (5, 16, 256, 56),
}

_GRID_NAMES = {
    IQ2_XXS: "IQ2_XXS",
    IQ2_XS: "IQ2_XS",
    IQ3_XXS: "IQ3_XXS",
    IQ1_S: "IQ1_S",
    IQ3_S: "IQ3_S",
    IQ2_S: "IQ2_S",
    IQ1_M: "IQ1_M",
}


@cache
def _grid(quant_type: int) -> np.ndarray:
    from gguf import quants

    quant = getattr(quants, _GRID_NAMES[quant_type])
    table = np.array(quant.grid_map, dtype=np.int16)
    bits = (len(table) - 1).bit_length()
    per_byte = 8 // bits
    packed = np.frombuffer(bytes.fromhex(quant.grid_hex.decode()), dtype=np.uint8)
    indices = (packed[:, None] >> np.arange(per_byte) * (8 // per_byte)) & (
        (1 << bits) - 1
    )
    return table[indices].reshape(quant.grid_shape)


@lru_cache(maxsize=1)
def _sign_table() -> np.ndarray:
    from gguf.quants import IQ2_XXS as GGMLIQ2XXS

    return np.frombuffer(GGMLIQ2XXS.ksigns, dtype=np.uint8)


@lru_cache(maxsize=1)
def _iq4_values() -> np.ndarray:
    from gguf.quants import IQ4_NL as GGMLIQ4NL

    return np.array(GGMLIQ4NL.kvalues, dtype=np.int16)


def _u16(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<u2")


def _u32(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<u4")


def _half(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32)


def _nibbles(raw: np.ndarray) -> np.ndarray:
    """Unpack low/high nibbles in byte order (GGML scale metadata)."""
    return np.stack((raw & 15, raw >> 4), axis=-1).reshape(raw.shape[0], -1)


def _signs(raw: np.ndarray) -> np.ndarray:
    return 1 - 2 * ((raw[..., None] >> np.arange(8)) & 1).astype(np.int16)


def _compressed_signs(raw: np.ndarray) -> np.ndarray:
    indices = (raw[..., None] >> np.array([0, 7, 14, 21])) & 127
    return _signs(_sign_table()[indices])


def _midpoint_codes(values: np.ndarray, bits: int) -> np.ndarray:
    return (values + (1 << (bits - 1))).astype(np.uint8).reshape(values.shape[0], -1)


def decode_iq_blocks(
    blocks: np.ndarray, quant_type: int
) -> tuple[np.ndarray, np.ndarray, None]:
    """Return midpoint-biased codes and float32 group scales, with no offsets.

    The result obeys ``weight = (code - 2**(bits-1)) * scale``. IQ1's
    fractional +/-1/8 codebook offset is folded into exact integer values,
    so IQ1_M's per-eight-element offsets need no sub-16 group sizes.
    """
    quant_type = int(quant_type)
    if quant_type not in IQ_FORMATS:
        raise ValueError(f"Unsupported importance quantization type: {quant_type}")
    bits, _, _, block_bytes = IQ_FORMATS[quant_type]
    if blocks.dtype != np.uint8 or blocks.ndim != 2:
        raise ValueError("IQ blocks must be a two-dimensional uint8 array")
    if blocks.shape[1] != block_bytes:
        raise ValueError(f"Expected {block_bytes} bytes per IQ block")
    count = blocks.shape[0]

    if quant_type == IQ2_XXS:
        d = _half(blocks[:, :2])
        packed = _u32(blocks[:, 2:]).reshape(count, 8, 2)
        sub_scales = (packed[:, :, 1] >> 28).astype(np.float32)
        scales = d * (np.float32(0.5) + sub_scales) * np.float32(0.25)
        indices = np.ascontiguousarray(packed[:, :, 0]).view(np.uint8)
        values = _grid(quant_type)[indices].reshape(count, 8, 4, 8)
        values *= _compressed_signs(packed[:, :, 1])

    elif quant_type == IQ2_XS:
        d = _half(blocks[:, :2])
        packed = _u16(blocks[:, 2:66])
        scales = d * (np.float32(0.5) + _nibbles(blocks[:, 66:74])) * np.float32(0.25)
        values = _grid(quant_type)[packed & 511]
        values *= _signs(_sign_table()[packed >> 9])

    elif quant_type == IQ2_S:
        d = _half(blocks[:, :2])
        high = ((blocks[:, 66:74, None] >> np.array([0, 2, 4, 6])) & 3).reshape(
            count, 32
        )
        indices = blocks[:, 2:34].astype(np.uint16) | (high.astype(np.uint16) << 8)
        scales = d * (np.float32(0.5) + _nibbles(blocks[:, 74:82])) * np.float32(0.25)
        values = _grid(quant_type)[indices]
        values *= _signs(blocks[:, 34:66])

    elif quant_type == IQ3_XXS:
        d = _half(blocks[:, :2])
        metadata = _u32(blocks[:, 66:98])
        # Every codebook entry is even. Factoring out two saves one bit.
        scales = d * (np.float32(0.5) + (metadata >> 28).astype(np.float32))
        values = (_grid(quant_type)[blocks[:, 2:66]] // 2).reshape(count, 8, 4, 8)
        values *= _compressed_signs(metadata)

    elif quant_type == IQ3_S:
        d = _half(blocks[:, :2])
        high = ((blocks[:, 66:74, None] >> np.arange(8)) & 1).reshape(count, 64)
        indices = blocks[:, 2:66].astype(np.uint16) | (high.astype(np.uint16) << 8)
        scales = d * (1 + 2 * _nibbles(blocks[:, 106:110]))
        values = _grid(quant_type)[indices].reshape(count, 32, 8)
        values *= _signs(blocks[:, 74:106])

    elif quant_type == IQ1_S:
        d = _half(blocks[:, :2])
        metadata = _u16(blocks[:, 34:50])
        high = ((metadata[..., None] >> np.array([0, 3, 6, 9])) & 7).reshape(count, 32)
        indices = blocks[:, 2:34].astype(np.uint16) | (high.astype(np.uint16) << 8)
        scales = d * (2 * ((metadata >> 12) & 7) + 1) * np.float32(0.125)
        delta = 1 - 2 * (metadata >> 15).astype(np.int16)
        values = _grid(quant_type)[indices].reshape(count, 8, 4, 8) * 8
        values += delta[:, :, None, None]

    elif quant_type == IQ1_M:
        metadata = _u16(blocks[:, 48:56])
        fragments = (metadata & 0xF000) >> np.array([12, 8, 4, 0], dtype=np.uint16)
        d_bits = np.bitwise_or.reduce(fragments, axis=1).astype("<u2")
        d = d_bits.view("<f2").astype(np.float32)[:, None]
        sub_scales = ((metadata[..., None] >> np.array([0, 3, 6, 9])) & 7).reshape(
            count, 16
        )
        scales = d * (2 * sub_scales + 1).astype(np.float32) * np.float32(0.125)
        high_delta = _nibbles(blocks[:, 32:48])
        indices = blocks[:, :32].astype(np.uint16) | (
            (high_delta.astype(np.uint16) & 7) << 8
        )
        delta = 1 - 2 * (high_delta >> 3).astype(np.int16)
        values = _grid(quant_type)[indices] * 8 + delta[:, :, None]

    elif quant_type == IQ4_NL:
        scales = _half(blocks[:, :2])
        low = blocks[:, 2:18]
        indices = np.concatenate((low & 15, low >> 4), axis=1)
        values = _iq4_values()[indices]

    else:  # IQ4_XS
        d = _half(blocks[:, :2])
        high = (_u16(blocks[:, 2:4]) >> (2 * np.arange(8))) & 3
        sub_scales = (_nibbles(blocks[:, 4:8]) | (high << 4)).astype(np.int16) - 32
        scales = d * sub_scales
        low = blocks[:, 8:136].reshape(count, 8, 16)
        indices = np.concatenate((low & 15, low >> 4), axis=2)
        values = _iq4_values()[indices]

    return _midpoint_codes(values, bits), scales.astype(np.float32), None
