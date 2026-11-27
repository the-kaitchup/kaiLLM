# SPDX-License-Identifier: Apache-2.0
"""Hy4 Engram table kept in host memory (memory-mapped from the GGUF file).

Each layer reads 16 packed rows per token. Full CUDA graphs gather them directly from
the stage-local, registered host mapping. Ordinary serving can gather on the CPU and
transfer the selected rows. Both paths use the same on-device GGUF dequantization.
"""

from __future__ import annotations

import mmap
import weakref

import numpy as np
import torch

_PLAIN = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}
_BLOCKS = {"Q4_0": (32, 18, 2), "Q4_1": (32, 20, 3), "Q5_0": (32, 22, 6), "Q5_1": (32, 24, 7), "Q8_0": (32, 34, 8),
           "IQ4_NL": (32, 18, 20), "Q4_K": (256, 144, 12), "Q6_K": (256, 210, 14)}


def _pinning() -> bool:
    """vLLM's rule (off under WSL, where pinned host memory is scarce)."""
    try:
        from vllm.utils.platform_utils import is_pin_memory_available
    except ImportError:  # pragma: no cover
        return torch.cuda.is_available()
    return is_pin_memory_available()


class HostEngramTable:
    def __init__(self, path: str, offset: int, nbytes: int, ggml_type: str, rows: int, dim: int):
        self.type, self.rows, self.dim = ggml_type, rows, dim
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,))
        if rows < 1 or nbytes % rows:
            raise ValueError("Invalid Engram packed row layout")
        self.row_bytes = nbytes // rows
        self._graph_rows = None
        self._first_graph_row = 0
        self.packed = torch.from_numpy(raw).view(rows, self.row_bytes)
        if ggml_type not in _PLAIN and ggml_type not in _BLOCKS:
            raise ValueError(f"Unsupported Engram table type {ggml_type}")

    @classmethod
    def from_config(cls, config):
        info = getattr(config, "hy4_engram_table", None)
        return None if not info else cls(**info)

    def prepare_graph_rows(self, start, end):
        """Pin this pipeline stage's GGUF file pages once and expose them via CUDA UVA."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        if not 0 <= start < end <= self.rows or self._graph_rows is not None:
            raise ValueError("Invalid or repeated Engram stage mapping")
        stage = self.packed[start:end]
        address = stage.data_ptr()
        aligned = address - address % mmap.PAGESIZE
        length = ((address + stage.nbytes - aligned + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        api = torch.cuda.cudart()
        # cudaHostRegisterMapped | cudaHostRegisterReadOnly; the source is a read-only GGUF mmap.
        ret = api.cudaHostRegister(aligned, length, 2 | 8)
        if ret.value != 0:
            raise RuntimeError(f"Could not register {length} bytes of read-only host Engram pages: {ret}")
        try:
            if not stage.is_pinned():
                raise RuntimeError("CUDA did not register the mapped Engram rows")
            self._graph_rows = get_accelerator_view_from_cpu_tensor(stage)
        except Exception:
            api.cudaHostUnregister(aligned)
            raise
        self._first_graph_row = start
        # Retain the mapped file until the device alias and graph captures have been released.
        finalizer = weakref.finalize(self, self._unregister, api, aligned, self.packed)
        finalizer.atexit = False

    @staticmethod
    def _unregister(api, aligned, backing):
        api.cudaHostUnregister(aligned)

    def gather(self, rows: torch.Tensor, dim: int, device=None, dtype=torch.bfloat16) -> torch.Tensor:
        device = rows.device if device is None else device
        if self._graph_rows is not None:
            if rows.device != self._graph_rows.device:
                raise ValueError("Engram row IDs must share the mapped stage's CUDA device")
            picked = self._graph_rows.index_select(0, rows.reshape(-1) - self._first_graph_row)
        else:
            if rows.is_cuda:
                from .attention import graph_backend_enabled
                if graph_backend_enabled():
                    raise RuntimeError("Hy4 graph gather requires the stage's registered host Engram rows")
            picked = self.packed.index_select(0, rows.reshape(-1).cpu())
            if torch.device(device).type == "cuda" and _pinning():
                picked = picked.pin_memory().to(device, non_blocking=True)
            else:
                picked = picked.to(device)
        if self.type in _PLAIN:
            return picked.view(_PLAIN[self.type]).view(-1, dim).to(dtype)
        from .. import ops
        _, _, type_id = _BLOCKS[self.type]
        return ops.ggml_dequantize(picked.contiguous(), type_id, picked.shape[0], dim, dtype)
