# SPDX-License-Identifier: Apache-2.0
"""Prepared original K2/K3/K4 core with the qualified grouped postprocessor.

This host-only candidate does not select policies, compile a library or install
an execution route. The caller supplies each original (warps, prefetch, splits,
shared-LUT) policy and an authenticated native library. ``core`` returns a flat
allocation whose per-part views have shape [rows, splits, output_columns].
"""

import ctypes
import hashlib
from pathlib import Path

import torch
import triton

from .yaqa_grouped_k4 import PreparedGroupedQ4
from .yaqa_grouped_k234_metadata import descriptor_rows


def _signed_storage(value):
    """Preserve uint64 pointer bits in a Torch int64 descriptor allocation."""
    return value if value < 1 << 63 else value - (1 << 64)


class PreparedGroupedK234:
    # Keep the original function object, globals, reduction order and kernels.
    post = PreparedGroupedQ4.post

    def __init__(self, parts, rows, library, library_sha256, *, policies):
        self.parts = tuple(parts)
        self.policies = tuple(tuple(policy) for policy in policies)
        if not self.parts or len(self.parts) != len(self.policies):
            raise ValueError("Matching nonempty parts and original policies required")
        self.shapes = tuple(tuple(part.shape) for part in self.parts)
        self.bits = tuple(part.bits for part in self.parts)
        self.pointers = tuple(
            (part.trellis.data_ptr(), part.tlut.data_ptr()) for part in self.parts
        )
        desc, self.blocks, self.elements, self.max_warps = descriptor_rows(
            self.shapes, self.bits, self.policies, rows, self.pointers
        )
        self.rows = rows
        self.width = self.shapes[0][1]
        self.device = self.parts[0].trellis.device
        if self.device.type != "cuda":
            raise ValueError("CUDA original packed parts required")
        for part, (n, k), bit in zip(self.parts, self.shapes, self.bits, strict=True):
            if (
                part.codec != "qtip"
                or not part.permutation
                or part.dtype not in (torch.float32, torch.bfloat16)
                or part.trellis.dtype != torch.int16
                or part.trellis.shape != (n * k // 256, bit * 16)
                or part.tlut.dtype != torch.bfloat16
                or part.tlut.shape != (512, 2)
                or part.input_scale.dtype != torch.float32
                or part.input_scale.shape != (k,)
                or part.output_scale.dtype != torch.float32
                or part.output_scale.shape != (n,)
                or part.hadamard_output.ndim != 2
                or part.hadamard_output.shape[0] < 1
                or part.hadamard_output.shape
                != (part.hadamard_output.shape[0], part.hadamard_output.shape[0])
                or part.hadamard_output.dtype not in (torch.float32, torch.float64)
                or part.Wscale.numel() != 1
                or part.Wscale.dtype not in (torch.float32, torch.float64)
                or any(
                    tensor.device != self.device or not tensor.is_contiguous()
                    for tensor in (
                        part.trellis,
                        part.tlut,
                        part.input_scale,
                        part.output_scale,
                        part.hadamard_output,
                        part.Wscale,
                    )
                )
            ):
                raise ValueError("Original permuted QTIP K2/K3/K4 metadata required")
        self.ns = tuple(n for n, _ in self.shapes)
        self.ks = tuple(part.hadamard_output.shape[0] for part in self.parts)
        if any(n % k for n, k in zip(self.ns, self.ks, strict=True)):
            raise ValueError("Original output factor must divide its width")
        self.ps = tuple(n // k for n, k in zip(self.ns, self.ks, strict=True))
        if any(power < 1 or power & (power - 1) for power in self.ps):
            raise ValueError("Original output power factor required")
        if tuple(torch.cuda.get_device_capability(self.device)) not in (
            (8, 6),
            (12, 0),
        ):
            raise ValueError("Grouped core requires an authenticated SM86/SM120 target")
        library = Path(library).absolute()
        if hashlib.sha256(library.read_bytes()).hexdigest() != library_sha256:
            raise ValueError("Unbound grouped K234 native library")
        self.library_path = str(library)
        self.library_sha256 = library_sha256
        self.splits = tuple(policy[2] for policy in self.policies)
        self.policy = self.policies
        self.columns = tuple(sum(self.ns[:group]) for group in range(len(self.parts)))
        self.factors = tuple(sum(self.ks[:group]) for group in range(len(self.parts)))
        self.offsets = tuple(row[6] for row in desc)
        self.total = sum(self.ns)
        self.h64 = tuple(
            part.hadamard_output.dtype == torch.float64 for part in self.parts
        )
        self.global64 = tuple(part.Wscale.dtype == torch.float64 for part in self.parts)
        self.work_counts = tuple(
            triton.cdiv(n, 256)
            if k == 1
            else triton.cdiv(k, 16) * triton.cdiv(power, 32)
            for n, k, power in zip(self.ns, self.ks, self.ps, strict=True)
        )
        self.work = tuple(
            sum(self.work_counts[:group]) for group in range(len(self.parts))
        )
        self.partial_shapes = tuple(
            (rows, splits, n) for n, splits in zip(self.ns, self.splits, strict=True)
        )
        # Retain original tensors, including views, independently of part fields.
        # No wire, LUT, output metadata or dense weights are cloned or repacked.
        self.wire_tensors = tuple((part.trellis, part.tlut) for part in self.parts)
        self.input_scales = tuple(part.input_scale for part in self.parts)
        self.post_tensors = tuple(
            (part.hadamard_output, part.output_scale, part.Wscale)
            for part in self.parts
        )
        self.descriptor_values = tuple(tuple(row) for row in desc)
        self.descriptors = torch.tensor(
            [[_signed_storage(value) for value in row] for row in desc],
            device=self.device,
            dtype=torch.int64,
        )
        self.ptrs = torch.tensor(
            [
                [_signed_storage(tensor.data_ptr()) for tensor in tensors]
                for tensors in self.post_tensors
            ],
            device=self.device,
            dtype=torch.int64,
        )
        self.library = ctypes.CDLL(str(library))
        self.run = self.library.yaqa_k234_grouped_rows
        self.run.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5 + [ctypes.c_size_t]
        )
        self.run.restype = ctypes.c_int

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
            result = self.run(
                x.data_ptr(),
                self.descriptors.data_ptr(),
                out.data_ptr(),
                len(self.parts),
                self.width,
                self.rows,
                self.blocks,
                self.max_warps,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if result:
            raise RuntimeError(f"Grouped original K234 launch failed: {result}")
        return out

    def buffers(self):
        return [self.descriptors, self.ptrs, *self.input_scales] + [
            tensor
            for tensors in (*self.wire_tensors, *self.post_tensors)
            for tensor in tensors
        ]
