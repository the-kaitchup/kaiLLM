# SPDX-License-Identifier: Apache-2.0
"""Hy4 Engram table kept in host memory (memory-mapped from the GGUF file).

Only 16 rows per layer and token are read, so the 35-127 GB table never goes to the GPU: rows are
gathered on the CPU from the mmap (packed bytes), copied to the GPU, and dequantized there
(F8_E4M3 / BF16 / F16 / F32 by view, block types with ggml's dequant kernels).
"""

from __future__ import annotations

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
        self.row_bytes = nbytes // rows
        self.packed = torch.from_numpy(raw).view(rows, self.row_bytes)
        if ggml_type not in _PLAIN and ggml_type not in _BLOCKS:
            raise ValueError(f"Unsupported Engram table type {ggml_type}")

    @classmethod
    def from_config(cls, config):
        info = getattr(config, "hy4_engram_table", None)
        return None if not info else cls(**info)

    def gather(self, rows: torch.Tensor, dim: int, device=None, dtype=torch.bfloat16) -> torch.Tensor:
        device = rows.device if device is None else device
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
