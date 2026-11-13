# SPDX-License-Identifier: Apache-2.0
"""Exact integer representations for ternary and microscale GGUF formats."""

import numpy as np

Q8_K, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 = 15, 34, 35, 39, 40, 41, 42

# Integer bits, scale group, original block elements, original block bytes.
EXTRA_FORMATS = {
    Q8_K: (8, 256, 256, 292),
    TQ1_0: (2, 256, 256, 54),
    TQ2_0: (2, 256, 256, 66),
    MXFP4: (5, 32, 32, 17),
    NVFP4: (5, 16, 64, 36),
    Q1_0: (2, 128, 128, 18),
    Q2_0: (2, 64, 64, 18),
}

_FP4_DOUBLED = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int16,
)


def _half(raw):
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32)


def decode_extra_blocks(blocks: np.ndarray, quant_type: int):
    """Return midpoint-biased integers, scales, and an optional affine offset.

    FP4's half-integer codebook is doubled exactly and its scale halved, fitting
    signed INT5 without quantization. Ternary and binary weights use signed INT2.
    """
    quant_type = int(quant_type)
    if quant_type not in EXTRA_FORMATS:
        raise ValueError(f"Unsupported extended GGUF type: {quant_type}")
    bits, _, block_size, block_bytes = EXTRA_FORMATS[quant_type]
    if blocks.dtype != np.uint8 or blocks.ndim != 2 or blocks.shape[1] != block_bytes:
        raise ValueError(f"Expected uint8 blocks with {block_bytes} bytes")
    count = len(blocks)
    if quant_type == Q2_0:
        scales = _half(blocks[:, :2])
        positions = np.arange(64)
        codes = (blocks[:, 2 + positions // 4] >> (2 * (positions % 4))) & 3
        # Q2_0 has the asymmetric values [-1, 0, 1, 2] * scale.
        return codes.astype(np.uint8), scales, scales.copy()
    if quant_type == Q8_K:
        scales = np.ascontiguousarray(blocks[:, :4]).view("<f4")
        values = blocks[:, 4:260].view(np.int8).astype(np.int16)
        # The remaining int16 sums are dot-product helpers, not weight values.
    elif quant_type == TQ1_0:
        low = blocks[:, :48]
        powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
        first = low[:, :32, None].transpose(0, 2, 1) * powers[None, :, None]
        second = low[:, 32:48, None].transpose(0, 2, 1) * powers[None, :, None]
        last = blocks[:, 48:52, None].transpose(0, 2, 1) * powers[None, :4, None]
        trits = np.concatenate(
            (
                first.reshape(count, 160),
                second.reshape(count, 80),
                last.reshape(count, 16),
            ),
            axis=1,
        )
        # uint8 multiplication above deliberately wraps, matching GGML's trits.
        values = ((trits.astype(np.uint16) * 3) >> 8).astype(np.int16) - 1
        scales = _half(blocks[:, 52:54])
    elif quant_type == TQ2_0:
        packed = blocks[:, :64].reshape(count, 2, 1, 32)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]
        trits = ((packed >> shifts) & 3).reshape(count, 256)
        if np.any(trits == 3):
            raise ValueError("TQ2_0 contains reserved ternary code 3")
        values = trits.astype(np.int16) - 1
        scales = _half(blocks[:, 64:66])
    elif quant_type == MXFP4:
        exponent = blocks[:, :1].astype(np.uint32)
        scale_bits = np.where(
            exponent < 2,
            np.uint32(0x00200000) << exponent,
            (exponent - 1) << np.uint32(23),
        ).astype(np.uint32)
        scales = scale_bits.view(np.float32)
        packed = blocks[:, 1:]
        indices = np.concatenate((packed & 15, packed >> 4), axis=1)
        values = _FP4_DOUBLED[indices]
    elif quant_type == NVFP4:
        raw = blocks[:, :4]
        exponent = ((raw >> 3) & 15).astype(np.int32)
        mantissa = (raw & 7).astype(np.float32)
        scales = np.where(
            exponent == 0,
            mantissa * np.float32(2**-9),
            (1 + mantissa / 8) * np.exp2(exponent.astype(np.float32) - 7),
        ) * np.float32(0.5)
        scales = np.where((raw == 0) | (raw == 127), np.float32(0), scales)
        packed = blocks[:, 4:].reshape(count, 4, 8)
        indices = np.concatenate((packed & 15, packed >> 4), axis=2)
        values = _FP4_DOUBLED[indices].reshape(count, block_size)
    else:  # Q1_0
        scales = _half(blocks[:, :2])
        binary = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little")
        values = 2 * binary.astype(np.int16) - 1
    codes = (values + (1 << (bits - 1))).astype(np.uint8)
    return codes, scales.astype(np.float32), None
