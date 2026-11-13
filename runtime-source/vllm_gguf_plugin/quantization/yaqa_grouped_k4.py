# SPDX-License-Identifier: Apache-2.0
"""Unselected grouped original-Q4 CUDA core and exact merged post transforms."""

import ctypes
import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

MAX_PARTIAL_BYTES = 4 << 20


def descriptor_rows(shapes, splits, rows, warps, pointers):
    if not shapes or len(shapes) != len(splits) or len(shapes) != len(pointers):
        raise ValueError("Nonempty matching group metadata required")
    if warps not in (4, 8) or not 0 < rows <= 65535:
        raise ValueError("Original shader row/warp capability exceeded")
    width = shapes[0][1]
    if not 0 < width <= 2**31 - 1 or width % 16:
        raise ValueError("Positive aligned common input width required")
    ctas = offset = 0
    descriptors = []
    for (n, k), s, (wptr, lptr) in zip(shapes, splits, pointers, strict=True):
        if (
            k != width
            or n <= 0
            or n % 16
            or n > 2**31 - 1
            or n * k // 256 * 32 > 2**31 - 1
            or s not in (1, 2, 4, 8, 16, 32)
            or not 0 < wptr < 2**64
            or not 0 < lptr < 2**64
            or wptr % 4
            or lptr % 4
        ):
            raise ValueError("Invalid original per-part wire geometry or alignment")
        count = ((n // 16 + warps - 1) // warps) * s
        descriptors.append([wptr, lptr, n, s, ctas, count, offset])
        ctas += count
        offset += rows * s * n
    if ctas > 2**31 - 1 or offset * 4 > MAX_PARTIAL_BYTES:
        raise ValueError("Bounded transient workspace/grid exceeded")
    return descriptors, ctas, offset


@triton.jit
def _group_power(
    Partial,
    Ptrs,
    Y,
    NS: tl.constexpr,
    PS: tl.constexpr,
    KS: tl.constexpr,
    SS: tl.constexpr,
    OFFSETS: tl.constexpr,
    COLUMNS: tl.constexpr,
    FACTORS: tl.constexpr,
    GLOBAL64: tl.constexpr,
    TOTAL: tl.constexpr,
    FINAL: tl.constexpr,
):
    row, program = tl.program_id(0), tl.program_id(1)
    for g in tl.static_range(len(NS)):
        if (program >= FACTORS[g]) & (program < FACTORS[g] + KS[g]):
            factor = program - FACTORS[g]
            i = tl.arange(0, PS[g])
            col = factor * PS[g] + i
            base = OFFSETS[g] + row * SS[g] * NS[g] + col
            value = tl.load(Partial + base).to(tl.float32)
            for split in tl.static_range(1, SS[g]):
                value = (value + tl.load(Partial + base + split * NS[g])).to(tl.float32)
            for level in tl.static_range(PS[g].bit_length() - 1):
                step = 1 << level
                other = tl.gather(value, i ^ step, 0)
                value = tl.where((i & step) == 0, value + other, other - value)
            value = (value * NS[g] ** -0.5).to(tl.float32)
            if FINAL:
                scale = tl.load(Ptrs + g * 3 + 1).to(tl.pointer_type(tl.float32))
                value = (value * tl.load(scale + col)).to(tl.float32)
                if GLOBAL64[g]:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float64))
                else:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float32))
                value = value * tl.load(gs)
            tl.store(Y + row * TOTAL + COLUMNS[g] + col, value)


@triton.jit
def _group_factor_post(
    X,
    Ptrs,
    Y,
    NS: tl.constexpr,
    PS: tl.constexpr,
    KS: tl.constexpr,
    COLUMNS: tl.constexpr,
    WORK: tl.constexpr,
    WORK_COUNTS: tl.constexpr,
    H64: tl.constexpr,
    GLOBAL64: tl.constexpr,
    TOTAL: tl.constexpr,
):
    row, program = tl.program_id(0), tl.program_id(1)
    for g in tl.static_range(len(NS)):
        if (program >= WORK[g]) & (program < WORK[g] + WORK_COUNTS[g]):
            local = program - WORK[g]
            scale = tl.load(Ptrs + g * 3 + 1).to(tl.pointer_type(tl.float32))
            if KS[g] == 1:
                column = local * 256 + tl.arange(0, 256)
                active = column < NS[g]
                result = tl.load(X + row * TOTAL + COLUMNS[g] + column, active, 0)
                result = (result * tl.load(scale + column, active, 0)).to(tl.float32)
                if GLOBAL64[g]:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float64))
                else:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float32))
                result = result * tl.load(gs)
                tl.store(Y + row * TOTAL + COLUMNS[g] + column, result, active)
            else:
                a = (local // tl.cdiv(PS[g], 32)) * 16 + tl.arange(0, 16)
                b = tl.arange(0, max(16, triton.next_power_of_2(KS[g])))
                cols = (local % tl.cdiv(PS[g], 32)) * 32 + tl.arange(0, 32)
                if H64[g]:
                    hp = tl.load(Ptrs + g * 3).to(tl.pointer_type(tl.float64))
                else:
                    hp = tl.load(Ptrs + g * 3).to(tl.pointer_type(tl.float32))
                h = tl.load(
                    hp + a[:, None] * KS[g] + b[None, :],
                    (a[:, None] < KS[g]) & (b[None, :] < KS[g]),
                    0,
                ).to(tl.float32)
                value = tl.load(
                    X + row * TOTAL + COLUMNS[g] + b[:, None] * PS[g] + cols[None, :],
                    (b[:, None] < KS[g]) & (cols[None, :] < PS[g]),
                    0,
                )
                result = tl.dot(h, value, input_precision="tf32x3")
                offset = a[:, None] * PS[g] + cols[None, :]
                active = (a[:, None] < KS[g]) & (cols[None, :] < PS[g])
                result = result * tl.load(scale + offset, active, 0)
                if GLOBAL64[g]:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float64))
                else:
                    gs = tl.load(Ptrs + g * 3 + 2).to(tl.pointer_type(tl.float32))
                result = result * tl.load(gs)
                tl.store(Y + row * TOTAL + COLUMNS[g] + offset, result, active)


class PreparedGroupedQ4:
    def __init__(
        self,
        parts,
        rows,
        library,
        library_sha256,
        *,
        warps=4,
        prefetch=4,
        shared=1,
        splits_override=None,
    ):
        from vllm_gguf_plugin.quantization.yaqa_cuda_k34 import tile_splits

        if not parts or not 0 < len(parts) <= 65535:
            raise ValueError("Positive bounded group required")
        self.parts = tuple(parts)
        self.rows = rows
        device = parts[0].trellis.device
        if tuple(torch.cuda.get_device_capability(device)) not in ((8, 6), (12, 0)):
            raise ValueError("Grouped core requires SM86 or SM120")
        self.shapes = tuple(tuple(p.shape) for p in parts)
        self.splits = tuple(tile_splits(*shape) for shape in self.shapes)
        if splits_override is not None:
            if len(splits_override) != len(self.shapes) or any(
                type(v) is not int or not 1 <= v <= 32 for v in splits_override
            ):
                raise ValueError("Positive bounded per-part split policy required")
            self.splits = tuple(splits_override)
        for p, (n, k) in zip(parts, self.shapes, strict=True):
            if (
                p.codec != "qtip"
                or p.bits != 4
                or not p.permutation
                or p.trellis.dtype != torch.int16
                or p.trellis.shape != (n * k // 256, 64)
                or p.tlut.dtype != torch.bfloat16
                or p.tlut.shape != (512, 2)
                or any(
                    t.device != device or not t.is_contiguous()
                    for t in (
                        p.trellis,
                        p.tlut,
                        p.hadamard_output,
                        p.output_scale,
                        p.Wscale,
                    )
                )
                or p.output_scale.dtype != torch.float32
                or p.output_scale.shape != (n,)
                or p.hadamard_output.ndim != 2
                or p.hadamard_output.shape[0] < 1
                or p.hadamard_output.shape
                != (p.hadamard_output.shape[0], p.hadamard_output.shape[0])
                or p.Wscale.numel() != 1
                or p.Wscale.dtype not in (torch.float32, torch.float64)
                or p.hadamard_output.dtype not in (torch.float32, torch.float64)
            ):
                raise ValueError("Original QTIP4 part contract required")
        self.pointers = tuple((p.trellis.data_ptr(), p.tlut.data_ptr()) for p in parts)
        desc, self.blocks, self.elements = descriptor_rows(
            self.shapes, self.splits, rows, warps, self.pointers
        )
        self.descriptors = torch.tensor(desc, device=device, dtype=torch.int64)
        self.ptrs = torch.tensor(
            [
                [
                    p.hadamard_output.data_ptr(),
                    p.output_scale.data_ptr(),
                    p.Wscale.data_ptr(),
                ]
                for p in parts
            ],
            device=device,
            dtype=torch.int64,
        )
        self.ns = tuple(n for n, k in self.shapes)
        self.width = self.shapes[0][1]
        self.ks = tuple(p.hadamard_output.shape[0] for p in parts)
        if any(n % k for n, k in zip(self.ns, self.ks, strict=True)):
            raise ValueError("Original output factor must divide its width")
        self.ps = tuple(n // k for n, k in zip(self.ns, self.ks, strict=True))
        if any(p < 1 or p & (p - 1) for p in self.ps):
            raise ValueError("Originalpowerfactorrequired")
        self.columns = tuple(sum(self.ns[:g]) for g in range(len(parts)))
        self.factors = tuple(sum(self.ks[:g]) for g in range(len(parts)))
        self.offsets = tuple(d[6] for d in desc)
        self.total = sum(self.ns)
        self.h64 = tuple(p.hadamard_output.dtype == torch.float64 for p in parts)
        self.global64 = tuple(p.Wscale.dtype == torch.float64 for p in parts)
        self.work_counts = tuple(
            triton.cdiv(n, 256) if k == 1 else triton.cdiv(k, 16) * triton.cdiv(p, 32)
            for n, k, p in zip(self.ns, self.ks, self.ps, strict=True)
        )
        self.work = tuple(sum(self.work_counts[:g]) for g in range(len(parts)))
        self.policy = (warps, prefetch, shared)
        self.device = device
        if hashlib.sha256(Path(library).read_bytes()).hexdigest() != library_sha256:
            raise ValueError("Unbound grouped native library")
        self.library = ctypes.CDLL(str(library))
        self.run = self.library.yaqa_k4_grouped_rows
        self.run.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 7 + [ctypes.c_size_t]
        )
        self.run.restype = ctypes.c_int
        self.library_sha256 = library_sha256

    def core(self, x):
        if (
            x.shape != (len(self.parts), self.rows, self.width)
            or x.dtype != torch.float32
            or x.device != self.device
            or not x.is_contiguous()
            or x.data_ptr() % 8
        ):
            raise ValueError(
                "Prepared contiguous grouped F32 transformed rows required"
            )
        out = torch.empty((self.elements,), device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            rc = self.run(
                x.data_ptr(),
                self.descriptors.data_ptr(),
                out.data_ptr(),
                len(self.parts),
                self.width,
                self.rows,
                self.blocks,
                *self.policy,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if rc:
            raise RuntimeError(f"GroupedoriginalK4launchfailed:{rc}")
        return out

    def post(self, partial, dtype, *, fuse_k1=True):
        if (
            partial.shape != (self.elements,)
            or partial.device != self.device
            or partial.dtype != torch.float32
            or not partial.is_contiguous()
            or dtype not in (torch.float32, torch.bfloat16)
        ):
            raise ValueError("Prepared partial/F32-or-BF16 output required")
        final = fuse_k1 and all(k == 1 for k in self.ks)
        out = torch.empty((self.rows, self.total), device=self.device, dtype=dtype)
        temp = out if final else torch.empty_like(out, dtype=torch.float32)
        _group_power[(self.rows, sum(self.ks))](
            partial,
            self.ptrs,
            temp,
            NS=self.ns,
            PS=self.ps,
            KS=self.ks,
            SS=self.splits,
            OFFSETS=self.offsets,
            COLUMNS=self.columns,
            FACTORS=self.factors,
            GLOBAL64=self.global64,
            TOTAL=self.total,
            FINAL=final,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if not final:
            _group_factor_post[(self.rows, sum(self.work_counts))](
                temp,
                self.ptrs,
                out,
                NS=self.ns,
                PS=self.ps,
                KS=self.ks,
                COLUMNS=self.columns,
                WORK=self.work,
                WORK_COUNTS=self.work_counts,
                H64=self.h64,
                GLOBAL64=self.global64,
                TOTAL=self.total,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out
