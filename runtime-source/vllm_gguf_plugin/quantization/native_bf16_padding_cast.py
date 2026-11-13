# SPDX-License-Identifier: Apache-2.0
"""Exact BF16-to-F32 conversion, zeroing only official invalid padding rows."""

import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

ROWS = (512, 514, 516, 1024, 2048)
TYPES = (8, 12, 13, 14, 23)
_WARMED = set()


@triton.jit
def _cast_padding_rows(X, Padding, Raw, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + cols, cols < K, 0).to(tl.float32)
    is_padding = tl.load(Padding + row)
    value = tl.where(is_padding, 0.0, value)
    tl.store(Raw + row * K + cols, value, cols < K)


@torch.library.custom_op("vllm_gguf::native_bf16_padding_cast", mutates_args=())
def padding_cast(x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] not in ROWS
        or x.shape[1] not in (2048, 7680)
        or torch.cuda.get_device_capability(x.device) != (8, 6)
    ):
        raise ValueError("Padding cast requires contiguous SM86 BF16 body prefill")
    if (
        padding.dtype != torch.bool
        or padding.ndim != 1
        or padding.shape[0] != x.shape[0]
        or padding.device != x.device
        or not padding.is_contiguous()
    ):
        raise ValueError("Require official same-device contiguous GPU bool row mask")
    raw = torch.empty_like(x, dtype=torch.float32)
    _cast_padding_rows[(x.shape[0],)](
        x,
        padding,
        raw,
        K=x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return raw


@padding_cast.register_fake
def _fake(x, padding):
    return x.new_empty(x.shape, dtype=torch.float32)


def apply(x, prepared, padding):
    if (
        prepared.qtype not in TYPES
        or not 128 <= prepared.n <= 7680
        or prepared.k not in (2048, 7680)
        or x.shape[-1] != prepared.k
        or x.device != prepared.packed.device
    ):
        raise ValueError("Padding cast is restricted to the original native body scope")
    return padding_cast(x.contiguous(), padding)


def warm(prepared, rows):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare padding conversion before graph capture")
    if any(row not in ROWS for row in rows):
        raise ValueError("Warm only qualified prefill rows")
    with torch.inference_mode():
        for row in rows:
            key = (prepared.packed.device, prepared.k, row)
            if key in _WARMED:
                continue
            x = torch.ones(
                (row, prepared.k), dtype=torch.bfloat16, device=prepared.packed.device
            )
            padding = torch.zeros(row, dtype=torch.bool, device=x.device)
            apply(x, prepared, padding)
            torch.cuda.synchronize(x.device)
            _WARMED.add(key)


def runtime_marker():
    path = Path(__file__).resolve()
    return {
        "contract": "official-padding-only-BF16-to-F32-cast-v1",
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
        "SM": 86,
        "activation_rows": list(ROWS),
        "qtypes": list(TYPES),
        "activation_dtype": "bfloat16",
        "output_dtype": "float32",
        "mask": "ForwardContext.is_padding; GPU bool, True means invalid",
        "valid_rows": "unchanged original BF16-to-F32 values",
        "invalid_rows": "zero before existing native/P2 sanitization",
        "native_P2_guard_and_repair_unchanged": True,
        "head_or_decode_admitted": False,
        "extra_persistent_weight_bytes": 0,
        "GPU_D2H_per_forward": False,
    }
