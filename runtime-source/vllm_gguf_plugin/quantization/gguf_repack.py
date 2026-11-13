# SPDX-License-Identifier: Apache-2.0
"""Decode GGML blocks into Humming's packed integers without requantization.

The conversion only unpacks integer codes and scale metadata. It never creates
an N-by-K floating point weight matrix. Temporary integer storage is bounded by
``chunk_rows``. Standard/K formats retain their bit width; IQ codebooks expand
to wider exact integers as documented in ``iq_repack``.
"""

from dataclasses import dataclass

import numpy as np

from .extra_repack import EXTRA_FORMATS, decode_extra_blocks
from .iq_repack import IQ_FORMATS, decode_iq_blocks

# GGMLQuantizationType values, kept here so CPU conversion needs only NumPy.
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 = 2, 3, 6, 7, 8, 9
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 10, 11, 12, 13, 14
FORMATS = {
    Q4_0: (4, 32, 32, 18),
    Q4_1: (4, 32, 32, 20),
    Q5_0: (5, 32, 32, 22),
    Q5_1: (5, 32, 32, 24),
    Q8_1: (8, 32, 32, 36),
    Q2_K: (2, 16, 256, 84),
    Q3_K: (3, 16, 256, 110),
    Q4_K: (4, 32, 256, 144),
    Q5_K: (5, 32, 256, 176),
    Q6_K: (6, 16, 256, 210),
    Q8_0: (8, 32, 32, 34),
    **IQ_FORMATS,
    **EXTRA_FORMATS,
}


@dataclass(frozen=True)
class PackedGGUF:
    """Standard (pre-transform) Humming tensors and their logical shape.

    Affine formats use ``(code - zero_point) * scale``. Symmetric formats
    use ``(code - 2**(bits-1)) * scale``, matching Humming's uintN convention.
    ``scales`` and ``zero_points`` stay float32 until GPU preparation.
    """

    weight: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray | None
    bits: int
    group_size: int
    shape: tuple[int, int]


def pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack successive unsigned codes into little-endian uint32 bitstreams.

    Values crossing a word boundary are split, which matters for INT5/INT6.
    This matches Humming's ``common_pack_weight`` standard storage format.
    """
    if bits not in range(1, 9):
        raise ValueError(f"Unsupported bit width: {bits}")
    if codes.ndim != 2 or codes.shape[1] % 32:
        raise ValueError("Integer codes must have shape [N,K] with K divisible by 32")
    if np.any(codes < 0) or np.any(codes >= 1 << bits):
        raise ValueError(f"Integer codes do not fit unsigned INT{bits}")
    groups = codes.reshape(codes.shape[0], -1, 32)
    words = np.zeros((*groups.shape[:-1], bits), dtype=np.uint32)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        value = groups[..., i].astype(np.uint32)
        words[..., word] |= value << np.uint32(shift)
        if shift + bits > 32:
            words[..., word + 1] |= value >> np.uint32(32 - shift)
    return words.reshape(codes.shape[0], -1).view(np.int32)


def unpack_bits(weight: np.ndarray, bits: int) -> np.ndarray:
    """Reference inverse of :func:`pack_bits`, useful for CPU validation."""
    if bits not in range(1, 9) or weight.ndim != 2:
        raise ValueError("Expected a packed INT1 through INT8 matrix")
    if weight.dtype != np.int32 or weight.shape[1] % bits:
        raise ValueError("Packed weights must be int32 with whole 32-code groups")
    words = weight.view(np.uint32).reshape(weight.shape[0], -1, bits)
    codes = np.empty((*words.shape[:-1], 32), dtype=np.uint8)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        value = words[..., word] >> np.uint32(shift)
        if shift + bits > 32:
            value |= words[..., word + 1] << np.uint32(32 - shift)
        codes[..., i] = value & ((1 << bits) - 1)
    return codes.reshape(weight.shape[0], -1)


def _half(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32)


def _k_scales(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Eight 6-bit scale/min pairs encoded in twelve bytes by GGML.
    scales = np.concatenate(
        (raw[:, :4] & 63, (raw[:, 8:12] & 15) | ((raw[:, :4] >> 6) << 4)),
        axis=1,
    )
    mins = np.concatenate(
        (raw[:, 4:8] & 63, (raw[:, 8:12] >> 4) | ((raw[:, 4:8] >> 6) << 4)),
        axis=1,
    )
    return scales, mins


def _decode_blocks(
    blocks: np.ndarray, quant_type: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return uint codes, float32 scales, and positive subtractive offsets."""
    count = blocks.shape[0]
    if quant_type in IQ_FORMATS:
        return decode_iq_blocks(blocks, quant_type)
    if quant_type in EXTRA_FORMATS:
        return decode_extra_blocks(blocks, quant_type)
    if quant_type in (Q4_0, Q4_1, Q5_0, Q5_1):
        affine = quant_type in (Q4_1, Q5_1)
        start = 4 if affine else 2
        d = _half(blocks[:, :2])
        offset = -_half(blocks[:, 2:4]) if affine else None
        high = None
        if quant_type in (Q5_0, Q5_1):
            high = np.unpackbits(
                blocks[:, start : start + 4], axis=1, bitorder="little"
            )
            start += 4
        low = blocks[:, start:]
        codes = np.concatenate((low & 15, low >> 4), axis=1)
        if high is not None:
            codes |= high << 4
        return codes, d, offset
    if quant_type == Q2_K:
        scales = blocks[:, :16]
        qs = blocks[:, 16:80].reshape(count, 2, 1, 32)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]
        codes = ((qs >> shifts) & 3).reshape(count, 256)
        return (
            codes,
            _half(blocks[:, 80:82]) * (scales & 15),
            _half(blocks[:, 82:84]) * (scales >> 4),
        )
    if quant_type == Q3_K:
        packed_scales = blocks[:, 96:108]
        low_scale = packed_scales[:, :8, None].transpose(0, 2, 1)
        low_scale = (low_scale >> np.array([0, 4], dtype=np.uint8)[None, :, None]) & 15
        high_scale = packed_scales[:, 8:12, None].transpose(0, 2, 1)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, :, None]
        high_scale = (high_scale >> shifts) & 3
        scales = (
            low_scale.reshape(count, 16) | (high_scale.reshape(count, 16) << 4)
        ).astype(np.int16) - 32
        low = blocks[:, 32:96].reshape(count, 2, 1, 32)
        low = ((low >> shifts[:, None]) & 3).reshape(count, 8, 32)
        high = blocks[:, :32, None].transpose(0, 2, 1)
        high = (high >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
        codes = (low | (high << 2)).reshape(count, 256)
        return codes, _half(blocks[:, 108:110]) * scales, None
    if quant_type in (Q4_K, Q5_K):
        d, dmin = _half(blocks[:, :2]), _half(blocks[:, 2:4])
        sub_scales, sub_mins = _k_scales(blocks[:, 4:16])
        start = 16 if quant_type == Q4_K else 48
        low = blocks[:, start:].reshape(count, 4, 32)
        codes = np.stack((low & 15, low >> 4), axis=2).reshape(count, 8, 32)
        if quant_type == Q5_K:
            high = blocks[:, 16:48, None].transpose(0, 2, 1)
            shifts = np.arange(8, dtype=np.uint8)[None, :, None]
            codes |= ((high >> shifts) & 1) << 4
        return codes.reshape(count, 256), d * sub_scales, dmin * sub_mins
    if quant_type == Q6_K:
        low = blocks[:, :128].reshape(count, 2, 64)
        high = blocks[:, 128:192].reshape(count, 2, 32)
        codes = np.stack(
            (
                (low[..., :32] & 15) | ((high & 3) << 4),
                (low[..., 32:] & 15) | (((high >> 2) & 3) << 4),
                (low[..., :32] >> 4) | (((high >> 4) & 3) << 4),
                (low[..., 32:] >> 4) | (((high >> 6) & 3) << 4),
            ),
            axis=2,
        ).reshape(count, 256)
        scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
        return codes, scales * _half(blocks[:, 208:210]), None
    if quant_type in (Q8_0, Q8_1):
        # GGML stores signed bytes; Humming's uint8 storage is midpoint biased.
        start = 2 if quant_type == Q8_0 else 4
        codes = (blocks[:, start:].view(np.int8).astype(np.int16) + 128).astype(
            np.uint8
        )
        # Both wire formats store a half delta; Q8_1's second half is
        # an auxiliary dot-product sum, not part of the reconstructed weight.
        scales = _half(blocks[:, :2])
        return codes, scales, None
    raise ValueError(f"Unsupported GGML quantization type: {quant_type}")


def repack_gguf(
    data: np.ndarray, quant_type: int, *, chunk_rows: int = 128
) -> PackedGGUF:
    """Repack a two-dimensional GGUF tensor into Humming standard layout.

    The GGUF tensor must already be sliced/permuted for this TP partition.
    ``data`` is [output rows, encoded bytes per row], as returned by GGUFReader.
    This supports the standard and K integer formats, including mixed files.

    Q4_K/Q5_K require float zero points; their ratio is formed in float32.
    Humming later rounds scale/zero-point metadata to the activation dtype.
    No original integer code is quantized again. Constant groups whose scale
    is zero are explicitly encoded without division by zero.
    """
    quant_type = int(quant_type)
    if quant_type not in FORMATS:
        raise ValueError(f"Unsupported GGML quantization type: {quant_type}")
    if data.dtype != np.uint8 or data.ndim != 2:
        raise ValueError("GGUF weight data must be a two-dimensional uint8 array")
    bits, group_size, block_size, block_bytes = FORMATS[quant_type]
    if data.shape[1] % block_bytes:
        raise ValueError(f"Encoded rows must contain whole {block_bytes}-byte blocks")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    shape_n = data.shape[0]
    shape_k = data.shape[1] // block_bytes * block_size
    if shape_n == 0 or shape_k == 0:
        raise ValueError("GGUF weights must have nonzero dimensions")
    weight = np.empty((shape_n, shape_k * bits // 32), dtype=np.int32)
    scales = np.empty((shape_n, shape_k // group_size), dtype=np.float32)
    zero_points = None
    for row in range(0, shape_n, chunk_rows):
        end = min(shape_n, row + chunk_rows)
        blocks = np.ascontiguousarray(data[row:end]).reshape(-1, block_bytes)
        codes, sub_scales, offsets = _decode_blocks(blocks, quant_type)
        if not np.all(np.isfinite(sub_scales)):
            raise ValueError("GGUF contains non-finite weight scales")
        if offsets is not None:
            if zero_points is None:
                zero_points = np.empty_like(scales)
            if not np.all(np.isfinite(offsets)):
                raise ValueError("GGUF contains non-finite weight offsets")
            constant = sub_scales == 0
            if np.any(constant):
                codes = codes.reshape(*sub_scales.shape, group_size)
                codes[constant] = 0
                codes = codes.reshape(-1, block_size)
                sub_scales = np.where(constant, np.float32(1), sub_scales)
            assert zero_points is not None
            zero_points[row:end] = (offsets / sub_scales).reshape(end - row, -1)
        scales[row:end] = sub_scales.reshape(end - row, -1)
        weight[row:end] = pack_bits(codes.reshape(end - row, shape_k), bits)
    return PackedGGUF(weight, scales, zero_points, bits, group_size, (shape_n, shape_k))
