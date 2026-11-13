# SPDX-License-Identifier: Apache-2.0
"""Optional lossless QTIP index repack; the GGUF checkpoint is unchanged.

Three V2 lookup indices and signs occupy each 32-bit word. This raises resident
QTIP codes to approximately 5.33 bits/weight while removing trellis decoding
from inference. It does not allocate a dense weight cache or round activations.
This module is experimental and is not selected by the model loader.
"""

import torch
import triton
import triton.language as tl

from .yaqa import hadamard_reference


def _geometry(trellis, shape, bits):
    m, n = map(int, shape)
    if m < 16 or n < 16 or m % 16 or n % 16 or bits not in range(1, 9):
        raise ValueError("QTIP index repack requires L16/V2 matrices and K1 through K8")
    if trellis.dtype != torch.int16 or tuple(trellis.shape) != (
        m * n // 256,
        16 * bits,
    ):
        raise ValueError("QTIP trellis storage disagrees with its matrix geometry")
    return m, n, (n // 2 + 2) // 3


@triton.jit
def _repack_indices(
    W,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    BITS: tl.constexpr,
    WORDS: tl.constexpr,
    PERMUTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, word = offset // WORDS, offset % WORDS
    packed = tl.zeros((BLOCK,), tl.uint32)
    for part in tl.static_range(3):
        pair = word * 3 + part
        col = pair * 2
        valid = (row < M) & (col < N)
        tile = row // 16 * (N // 16) + col // 16
        p = (row % 16) * 16 + col % 16
        if PERMUTE:
            p = (
                (p // 16 % 8) * 32
                + (p // 2 % 4) * 8
                + (p // 8 % 2) * 4
                + (p // 128) * 2
                + p % 2
            )
        start = p // 2 * (2 * BITS)
        position, shift = start // 16, start % 16
        lo = (
            tl.load(W + tile * (16 * BITS) + position, valid, 0)
            .to(tl.uint16)
            .to(tl.uint32)
        )
        hi = (
            tl.load(W + tile * (16 * BITS) + (position + 1) % (16 * BITS), valid, 0)
            .to(tl.uint16)
            .to(tl.uint32)
        )
        state = (((lo << 16) | hi) >> (16 - shift)) & 65535
        hashed = state * (state + 1)
        code = ((hashed >> 6) & 511) | (((hashed >> 15) & 1) << 9)
        packed |= tl.where(valid, code, 0) << (10 * part)
    tl.store(Out + offset, packed, row < M)


def repack_qtip_indices(trellis, shape, bits, permutation=True):
    """Return row-major int32 words, each containing three exact 10-bit codes."""
    m, n, words_per_row = _geometry(trellis, shape, bits)
    if trellis.device.type == "cuda":
        output = torch.empty(
            (m, words_per_row), dtype=torch.int32, device=trellis.device
        )
        _repack_indices[(triton.cdiv(output.numel(), 256),)](
            trellis,
            output,
            M=m,
            N=n,
            BITS=bits,
            WORDS=words_per_row,
            PERMUTE=permutation,
            BLOCK=256,
        )
        return output
    words = (trellis.to(torch.int64) & 65535).flatten()
    cols = torch.arange(n // 2, device=trellis.device)[None, :] * 2
    output = torch.empty((m, words_per_row), dtype=torch.int32, device=trellis.device)
    for begin in range(0, m, 128):
        rows = torch.arange(begin, min(begin + 128, m), device=trellis.device)[:, None]
        tile = rows // 16 * (n // 16) + cols // 16
        p = (rows % 16) * 16 + cols % 16
        if permutation:
            p = (
                (p // 16 % 8) * 32
                + (p // 2 % 4) * 8
                + (p // 8 % 2) * 4
                + (p // 128) * 2
                + p % 2
            )
        start = p // 2 * (2 * bits)
        position, shift = start // 16, start % 16
        lo = words[tile * (16 * bits) + position]
        hi = words[tile * (16 * bits) + (position + 1) % (16 * bits)]
        state = (((lo << 16) | hi) >> (16 - shift)) & 65535
        hashed = state * (state + 1)
        code = ((hashed >> 6) & 511) | (((hashed >> 15) & 1) << 9)
        code = torch.nn.functional.pad(code, (0, words_per_row * 3 - n // 2))
        code = code.reshape(-1, words_per_row, 3)
        output[begin : begin + len(rows)] = (
            code[..., 0] | (code[..., 1] << 10) | (code[..., 2] << 20)
        ).to(torch.int32)
    return output


def pack_index_lut(lut):
    """Retain BF16/F16 lookup pairs in one word; retain other values in F32."""
    if lut.shape != (512, 2) or not lut.is_floating_point():
        raise ValueError("QTIP requires a floating lookup table of shape [512, 2]")
    lut = lut.detach().contiguous()
    if lut.dtype in (torch.bfloat16, torch.float16):
        values = lut.view(torch.int16).to(torch.int32) & 65535
        packed = values[:, 0] | (values[:, 1] << 16)
        return packed.contiguous(), "bf16" if lut.dtype == torch.bfloat16 else "f16"
    return lut.float().view(torch.int32).contiguous(), "f32"


def decode_qtip_indices(codes, lut, shape):
    """Dense numerical oracle for index validation and CPU execution only."""
    m, n = map(int, shape)
    pairs = torch.arange(n // 2, device=codes.device)
    values = codes.to(torch.int64)[:, pairs // 3]
    values = (values >> ((pairs % 3) * 10)) & 1023
    indices, signs = values & 511, (values >> 9) & 1
    decoded = lut.to(device=codes.device).float()[indices].clone()
    # Negation, including the sign of zero, matches the original QTIP decoder.
    decoded[..., 0] *= 1 - signs * 2
    return decoded.reshape(m, n)


@triton.jit
def _lookup_pair(Lut, code, LUT_KIND: tl.constexpr):
    index, sign = code & 511, code >> 9
    if LUT_KIND < 2:
        word = tl.load(Lut + index).to(tl.uint32) ^ (sign << 15)
        a = (word & 65535).to(tl.uint16)
        b = (word >> 16).to(tl.uint16)
        if LUT_KIND == 0:
            a = a.to(tl.bfloat16, bitcast=True).to(tl.float32)
            b = b.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            a = a.to(tl.float16, bitcast=True).to(tl.float32)
            b = b.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        a = tl.load(Lut + index * 2).to(tl.uint32) ^ (sign << 31)
        b = tl.load(Lut + index * 2 + 1).to(tl.uint32)
        a = a.to(tl.float32, bitcast=True)
        b = b.to(tl.float32, bitcast=True)
    return a, b


@triton.jit
def _decode_index_values(
    Codes,
    Lut,
    Out,
    TOTAL: tl.constexpr,
    PAIRS: tl.constexpr,
    WORDS: tl.constexpr,
    LUT_KIND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, pair = offset // PAIRS, offset % PAIRS
    word = tl.load(Codes + row * WORDS + pair // 3, offset < TOTAL, 0).to(tl.uint32)
    code = (word >> ((pair % 3) * 10)) & 1023
    a, b = _lookup_pair(Lut, code, LUT_KIND)
    tl.store(Out + offset * 2, a, offset < TOTAL)
    tl.store(Out + offset * 2 + 1, b, offset < TOTAL)


def decode_packed_index_values(codes, lut, shape, lut_kind):
    """Materialize indexed values for validation only; inference never calls it."""
    m, n = map(int, shape)
    if codes.device.type != "cuda":
        dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
            lut_kind
        ]
        return decode_qtip_indices(codes, lut.view(dtype).reshape(512, 2), shape)
    result = torch.empty((m, n), dtype=torch.float32, device=codes.device)
    _decode_index_values[(triton.cdiv(m * n // 2, 256),)](
        codes,
        lut,
        result,
        TOTAL=m * n // 2,
        PAIRS=n // 2,
        WORDS=codes.shape[1],
        LUT_KIND={"bf16": 0, "f16": 1, "f32": 2}[lut_kind],
        BLOCK=256,
    )
    return result


@triton.jit
def _index_gemv(
    X,
    Codes,
    Lut,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    WORDS: tl.constexpr,
    LUT_KIND: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    batches = tl.arange(0, BATCH)
    local = tl.arange(0, BK)
    accum = tl.zeros((BATCH, ROWS), tl.float32)
    for begin in range(tl.cdiv(N // 2, BK)):
        pairs = begin * BK + local
        word = tl.load(
            Codes + rows[:, None] * WORDS + pairs[None, :] // 3,
            (rows[:, None] < M) & (pairs[None, :] < N // 2),
            0,
        ).to(tl.uint32)
        code = (word >> ((pairs[None, :] % 3) * 10)) & 1023
        a, b = _lookup_pair(Lut, code, LUT_KIND)
        x0 = tl.load(
            X + batches[:, None] * N + pairs[None, :] * 2, pairs[None, :] < N // 2, 0
        ).to(tl.float32)
        x1 = tl.load(
            X + batches[:, None] * N + pairs[None, :] * 2 + 1,
            pairs[None, :] < N // 2,
            0,
        ).to(tl.float32)
        accum += tl.sum(
            a[None, :, :] * x0[:, None, :] + b[None, :, :] * x1[:, None, :], axis=2
        )
    tl.store(Y + batches[:, None] * M + rows[None, :], accum, rows[None, :] < M)


@triton.jit
def _index_gemv_wide(
    X,
    Codes,
    Lut,
    Y,
    N: tl.constexpr,
    WORDS: tl.constexpr,
    LUT_KIND: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    pairs = tl.arange(0, BK)
    word = tl.load(Codes + row * WORDS + pairs // 3, pairs < N // 2, 0).to(tl.uint32)
    code = (word >> ((pairs % 3) * 10)) & 1023
    a, b = _lookup_pair(Lut, code, LUT_KIND)
    x0 = tl.load(X + pairs * 2, pairs < N // 2, 0).to(tl.float32)
    x1 = tl.load(X + pairs * 2 + 1, pairs < N // 2, 0).to(tl.float32)
    total = tl.sum(a * x0 + b * x1, 0)
    tl.store(Y + row, total)


@triton.jit
def _index_gemm(
    X,
    Codes,
    Lut,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    WORDS: tl.constexpr,
    LUT_KIND: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batches = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    local = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for begin in range(tl.cdiv(N, BK)):
        cols = begin * BK + local
        pairs = cols // 2
        word = tl.load(
            Codes + rows[None, :] * WORDS + pairs[:, None] // 3,
            (rows[None, :] < M) & (cols[:, None] < N),
            0,
        ).to(tl.uint32)
        code = (word >> ((pairs[:, None] % 3) * 10)) & 1023
        a, b = _lookup_pair(Lut, code, LUT_KIND)
        weights = tl.where(cols[:, None] % 2 == 0, a, b)
        x = tl.load(
            X + batches[:, None] * N + cols[None, :],
            (batches[:, None] < BATCH) & (cols[None, :] < N),
            0,
        ).to(tl.float32)
        accum += tl.dot(x, weights, input_precision="tf32x3")
    tl.store(
        Y + batches[:, None] * M + rows[None, :],
        accum,
        (batches[:, None] < BATCH) & (rows[None, :] < M),
    )


def index_qtip_matmul(x, codes, lut, shape, lut_kind, *, route="auto"):
    """Multiply FP32 transformed activations by losslessly indexed QTIP values."""
    m, n = map(int, shape)
    source = x.reshape(-1, n).contiguous()
    batch = len(source)
    result = torch.empty((batch, m), dtype=torch.float32, device=x.device)
    kwargs = dict(
        M=m,
        N=n,
        BATCH=batch,
        WORDS=codes.shape[1],
        LUT_KIND={"bf16": 0, "f16": 1, "f32": 2}[lut_kind],
    )
    if route == "gemv-wide":
        if batch != 1:
            raise ValueError("Whole-K QTIP index GEMV currently requires batch one")
        block = triton.next_power_of_2(n // 2)
        _index_gemv_wide[(m,)](
            source,
            codes,
            lut,
            result,
            N=n,
            WORDS=codes.shape[1],
            LUT_KIND=kwargs["LUT_KIND"],
            BK=block,
            num_warps=4 if block <= 1024 else 8,
            enable_fp_fusion=False,
        )
    elif route == "gemv" or (route == "auto" and batch <= 2):
        padded = triton.next_power_of_2(batch)
        if padded != batch:
            source = torch.nn.functional.pad(source, (0, 0, 0, padded - batch))
            result = torch.empty((padded, m), dtype=torch.float32, device=x.device)
            kwargs["BATCH"] = padded
        _index_gemv[(triton.cdiv(m, 4),)](
            source,
            codes,
            lut,
            result,
            **kwargs,
            ROWS=4,
            BK=128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        result = result[:batch]
    elif route in ("auto", "gemm"):
        bm, bn = (64, 32) if batch >= 64 else (16, 32)
        _index_gemm[(triton.cdiv(batch, bm), triton.cdiv(m, bn))](
            source,
            codes,
            lut,
            result,
            **kwargs,
            BM=bm,
            BN=bn,
            BK=32,
            num_warps=4,
            num_stages=2,
        )
    else:
        raise ValueError(f"Unknown QTIP index multiply route: {route}")
    return result.reshape(*x.shape[:-1], m)


class IndexedYAQALinear(torch.nn.Module):
    """Opt-in QTIP linear with a larger packed resident representation."""

    def __init__(self, descriptor, components, dtype=torch.bfloat16, device=None):
        super().__init__()
        if descriptor.get("codec") != "qtip":
            raise ValueError("Indexed YAQA currently supports QTIP matrices only")
        self.shape = tuple(descriptor["shape"])
        self.codec = "qtip"
        self.bits = int(descriptor["bits"])
        self.dtype = dtype
        target = (
            torch.device(device) if device is not None else components["trellis"].device
        )
        trellis = components["trellis"].to(target)
        codes = repack_qtip_indices(
            trellis,
            self.shape,
            self.bits,
            descriptor.get("tensor_core_permutation", True),
        )
        self.original_trellis_bytes = trellis.numel() * trellis.element_size()
        self.register_buffer("index_codes", codes)
        lut, self.lut_kind = pack_index_lut(components["tlut"].to(target))
        self.register_buffer("index_lut", lut)
        for axis, sign, post in (
            ("input", "SU", "post_input_scale"),
            ("output", "SV", "post_output_scale"),
        ):
            scale = components[sign].to(device=target, dtype=torch.float32)
            if post in components:
                scale = scale * components[post].to(device=target, dtype=torch.float32)
            self.register_buffer(axis + "_scale", scale.contiguous())
            self.register_buffer(
                "hadamard_" + axis,
                components["hadamard_" + axis].to(target).contiguous(),
            )
        self.register_buffer("Wscale", components["Wscale"].to(target))

    def resident_report(self):
        index_bytes = self.index_codes.numel() * self.index_codes.element_size()
        return {
            "original_trellis_bytes": self.original_trellis_bytes,
            "index_bytes": index_bytes,
            "index_bits_per_weight": index_bytes * 8 / (self.shape[0] * self.shape[1]),
            "source_bits_per_weight": self.bits,
            "index_payload_expansion": index_bytes / self.original_trellis_bytes,
            "resident_buffers_bytes": sum(
                v.numel() * v.element_size() for v in self.buffers()
            ),
            "original_trellis_retained": False,
            "dense_weight_cache": False,
        }

    def _cpu_lut(self):
        if self.lut_kind == "f32":
            return self.index_lut.view(torch.float32)
        dtype = torch.bfloat16 if self.lut_kind == "bf16" else torch.float16
        return self.index_lut.view(dtype).reshape(512, 2)

    def forward(self, x, bias=None):
        m, n = self.shape
        if x.shape[-1] != n:
            raise ValueError("YAQA index input width differs from its packed matrix")
        if not x.numel():
            return x.new_empty((*x.shape[:-1], m))
        if x.device.type == "cuda":
            from .yaqa_kernels import _hadamard, _scale_output

            source = _hadamard(
                x.reshape(-1, n).contiguous(),
                self.hadamard_input,
                self.input_scale,
                transpose=True,
            )
            result = index_qtip_matmul(
                source, self.index_codes, self.index_lut, self.shape, self.lut_kind
            )
            result = _hadamard(
                result, self.hadamard_output, self.output_scale, pre=False
            )
            output = torch.empty_like(result, dtype=x.dtype)
            _scale_output[(triton.cdiv(result.numel(), 256),)](
                result,
                self.output_scale,
                self.Wscale,
                output,
                N=m,
                TOTAL=result.numel(),
                BLOCK=256,
            )
            output = output.reshape(*x.shape[:-1], m)
        else:
            source = hadamard_reference(
                x.float() * self.input_scale, self.hadamard_input, transpose=True
            )
            weight = decode_qtip_indices(self.index_codes, self._cpu_lut(), self.shape)
            result = source @ weight.T
            result = hadamard_reference(result, self.hadamard_output)
            output = (result * self.output_scale * self.Wscale.float()).to(x.dtype)
        return output if bias is None else output + bias
