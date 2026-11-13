# SPDX-License-Identifier: Apache-2.0
"""Merge native beta/alpha MMVF rows while preserving MMF/cuBLAS shapes."""

import torch

from . import current_ggml_direct as direct


@torch.library.custom_op("gguf_current::beta_alpha", mutates_args=())
def beta_alpha(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != 32
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.float32
        or weight.dtype not in (torch.float32, torch.float16)
        or not x.is_cuda
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA F32[M,K] and original [32,K]")
    m, k = x.shape
    small = 1 <= m <= (3 if weight.dtype == torch.float32 else 1)
    if small and torch.cuda.get_device_capability(x.device) == (8, 6):
        lib = direct._library()
        route = (
            lib.current_float_route
            if weight.dtype == torch.float32
            else lib.current_half_route
        )
        if (
            route(x.device.index, 16, k, m) == 1
            and route(x.device.index, 32, k, m) == 1
        ):
            # MMVF row count only changes its grid; per-row arithmetic and the
            # block size selected from K remain those of each N16 projection.
            return direct.current_ggml_float(x, weight)
    return torch.cat(
        [direct.current_ggml_float(x, part) for part in weight.split(16)], dim=-1
    )


@beta_alpha.register_fake
def _fake_beta_alpha(x, weight):
    return x.new_empty((x.shape[0], 32))
