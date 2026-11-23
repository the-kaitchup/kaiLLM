# SPDX-License-Identifier: Apache-2.0
"""Dequantization of the GGML block types new in llama.cpp master that Inkling 2 GGUFs use (MXFP4 39, NVFP4 40,
Q1_0 41, Q2_0 42, F8_E4M3 43) in torch (CPU or GPU), bit-exact with ggml-quants.c (MIT):

    Q1_0   128 values: f16 d, 16 bytes of bits (LSB first): +d / -d
    Q2_0    64 values: f16 d, 16 bytes of 2-bit codes (LSB first): (q - 1) d, q in 0..3
    MXFP4   32 values: E8M0 e, 16 bytes; low nibbles -> values 0..15, high -> 16..31; kvalues_fp4[n] * 2^(e - 128)
    NVFP4   64 values: 4 UE4M3 scales (one per 16; x 0.5 like ggml_ue4m3_to_fp32), 32 bytes; per 16-block: low
            nibbles -> 0..7, high -> 8..15; kvalues_fp4[n] * scale
    F8_E4M3  1 byte per value (ggml_f8_e4m3_to_fp32)
"""

from __future__ import annotations

import torch

MXFP4, NVFP4, Q1_0, Q2_0, F8_E4M3 = 39, 40, 41, 42, 43
BLOCK = {MXFP4: (32, 17), NVFP4: (64, 36), Q1_0: (128, 18), Q2_0: (64, 18), F8_E4M3: (1, 1)}
NEW_TYPES = frozenset(BLOCK)
KVALUES_FP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _f16(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (lo.to(torch.int32) | (hi.to(torch.int32) << 8)).to(torch.int16).view(torch.float16).float()


def _e8m0_half(e: torch.Tensor) -> torch.Tensor:
    """ggml_e8m0_to_fp32_half: 2^(e - 128) (denormal patterns for e < 2)."""
    e = e.to(torch.int32)
    bits = torch.where(e < 2, torch.tensor(0x00200000, dtype=torch.int32, device=e.device) << e, (e - 1) << 23)
    return bits.view(torch.float32)


def _ue4m3_half(x: torch.Tensor) -> torch.Tensor:
    """ggml_ue4m3_to_fp32: unsigned E4M3 (bias 7, sign bit ignored), x 0.5; 0 and 0x7F -> 0."""
    x = x.to(torch.int32)
    exp = (x >> 3) & 0xF
    man = (x & 0x7).float()
    raw = torch.where(exp == 0, man * 2.0 ** -9, (1.0 + man / 8.0) * torch.exp2((exp - 7).float()))
    return torch.where((x == 0) | (x == 0x7F), torch.zeros_like(raw), raw * 0.5)


def f8_e4m3_to_f32(x: torch.Tensor) -> torch.Tensor:
    """ggml_f8_e4m3_to_fp32 (OCP E4M3FN: bias 7, NaN at 0x7F / 0xFF): via torch's float8_e4m3fn."""
    return x.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).float()


def dequantize(raw: torch.Tensor, ggml_type: int, n_values: int | None = None) -> torch.Tensor:
    """raw: uint8 bytes (any shape whose last axis is a whole number of blocks per row) -> float32 values,
    shape [..., values per row]."""
    if ggml_type not in BLOCK:
        raise ValueError(f"not a new GGML type: {ggml_type}")
    qk, size = BLOCK[ggml_type]
    raw = raw.contiguous().view(torch.uint8)
    lead = raw.shape[:-1]
    b = raw.reshape(-1, size)
    if ggml_type == F8_E4M3:
        out = f8_e4m3_to_f32(b.reshape(-1))
    elif ggml_type == Q1_0:
        d = _f16(b[:, 0], b[:, 1])
        bits = (b[:, 2:, None] >> torch.arange(8, device=b.device, dtype=torch.uint8)) & 1     # [nb, 16, 8]
        out = torch.where(bits.reshape(-1, qk).bool(), d[:, None], -d[:, None])
    elif ggml_type == Q2_0:
        d = _f16(b[:, 0], b[:, 1])
        q = (b[:, 2:, None] >> torch.tensor([0, 2, 4, 6], device=b.device, dtype=torch.uint8)) & 3   # [nb, 16, 4]
        out = (q.reshape(-1, qk).float() - 1.0) * d[:, None]
    else:
        kv = torch.tensor(KVALUES_FP4, dtype=torch.float32, device=b.device)
        if ggml_type == MXFP4:
            d = _e8m0_half(b[:, 0])
            qs = b[:, 1:]                                                    # [nb, 16]
            vals = torch.cat([kv[(qs & 0xF).long()], kv[(qs >> 4).long()]], dim=1)
            out = vals * d[:, None]
        else:                                                                # NVFP4
            d = _ue4m3_half(b[:, :4])                                        # [nb, 4]
            qs = b[:, 4:].reshape(-1, 4, 8)                                  # [nb, sub, 8]
            vals = torch.cat([kv[(qs & 0xF).long()], kv[(qs >> 4).long()]], dim=2)   # [nb, sub, 16]
            out = (vals * d[:, :, None]).reshape(-1, qk)
    out = out.reshape(*lead, -1)
    return out if n_values is None else out[..., :n_values]
