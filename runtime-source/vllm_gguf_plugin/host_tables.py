# SPDX-License-Identifier: Apache-2.0
"""Lookup tables in pinned host memory, read by the GPU directly (zero-copy over PCIe, unified addressing).

For tables that a decode step only gathers a few rows from (an untied embedding, DSpine's TSV / TG): the GPU reads
the gathered rows over PCIe instead of keeping the whole table in VRAM.  The result is bit-identical (same bytes,
same kernels); a gather of 16-512 rows costs ~2-30 us more than from VRAM (RTX 3060, WSL2), inside CUDA graphs too.
"""
from __future__ import annotations

import torch

_TYPESTR = {torch.int8: "|i1", torch.uint8: "|u1", torch.int16: "<i2", torch.int32: "<i4", torch.int64: "<i8",
            torch.float16: "<f2", torch.float32: "<f4"}


class _HostView:
    """Exposes a pinned host tensor through __cuda_array_interface__ (same pointer under unified addressing)."""

    __slots__ = ("host",)

    def __init__(self, host: torch.Tensor):
        self.host = host

    @property
    def __cuda_array_interface__(self):
        return {"shape": tuple(self.host.shape), "typestr": _TYPESTR[self.host.dtype],
                "data": (self.host.data_ptr(), False), "version": 2, "strides": None}


_KEEP: list = []      # the pinned host tensors stay alive for the process


def host_alias(t: torch.Tensor) -> torch.Tensor:
    """A CUDA tensor whose storage is a pinned host copy of ``t`` (``t`` may be on any device)."""
    if t.dtype not in _TYPESTR:
        raise TypeError(f"host_alias: unsupported dtype {t.dtype}")
    host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)   # explicit: vLLM may set a CUDA default device
    host.copy_(t.detach().contiguous())
    view = _HostView(host)
    _KEEP.append(view)
    alias = torch.as_tensor(view, device=torch.device("cuda", torch.cuda.current_device()))
    if alias.data_ptr() != host.data_ptr():
        raise RuntimeError("host_alias: the CUDA view does not alias the pinned host block (no unified addressing?)")
    return alias


def host_bytes() -> int:
    return sum(v.host.numel() * v.host.element_size() for v in _KEEP)
