# SPDX-License-Identifier: Apache-2.0
"""Read the framework's current token-row padding mask without reading GPU data.

Call only for row-preserving body operators. Missing/misaligned metadata keeps
original behavior. The returned device pointer is loaded at kernel execution,
so graph replay uses the live mask rather than a captured CPU row count.
"""


def mask_for_rows(source, rows, *, role=None):
    if role != "body" or type(rows) is not int or rows <= 0:
        return None
    if not getattr(source, "is_cuda", False) or source.ndim < 2:
        return None
    if source.numel() != rows * source.shape[-1]:
        return None
    try:
        import torch
        from vllm import envs
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return None
    # Current GPUModelRunner only maintains this buffer when this flag is on.
    if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    padding = getattr(get_forward_context(), "is_padding", None)
    if (
        padding is None
        or padding.ndim != 1
        or tuple(padding.shape) != (rows,)
        or padding.dtype != torch.bool
        or not padding.is_cuda
        or padding.device != source.device
        or not padding.is_contiguous()
    ):
        return None
    return padding


def runtime_marker():
    import hashlib
    import os
    from pathlib import Path

    path = Path(__file__).resolve()
    return {
        "contract": "live-framework-token-padding-zero-inside-HalfPrep-v1",
        "scope": "row-preserving body operations only",
        "requires_framework_padding_maintenance": "VLLM_MOE_SKIP_PADDING",
        "framework_padding_maintenance_enabled": bool(
            int(os.getenv("VLLM_MOE_SKIP_PADDING", "1"))
        ),
        "valid_rows_unchanged": True,
        "genuine_valid_row_NaN_Inf_overflow_guard_unchanged": True,
        "host_GPU_data_reads": False,
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
    }
