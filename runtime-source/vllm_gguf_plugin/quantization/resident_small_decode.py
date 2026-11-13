# SPDX-License-Identifier: Apache-2.0
"""Default-off reuse of small matrices in the complete physical BF16 cache."""

import hashlib
import os
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_RESIDENT_SMALL_DECODE"


def enabled():
    value = os.getenv(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(ENV + " must be0or1")
    return value == "1"


def geometry(rows, shape):
    """Low-row scalar reuse; large prefills keep their existing MMA path."""
    if type(rows) is not int or not isinstance(shape, (tuple, list)) or len(shape) != 2:
        return False
    n, k = shape
    return (
        type(n) is int
        and type(k) is int
        and 0 < rows < 128
        and 0 < n <= 512
        and n % 16 == 0
        and k == 2048
    )


def marker():
    directory = Path(__file__).resolve().parent
    return {
        "enabled": enabled(),
        "contract": "SM86-small-resident-canonical-BF16-F32-reduction-v1",
        "physical_rows": "positive below128; existing larger-prefill MMA unchanged",
        "weight_geometry": "N<=512/N%16=0/K2048",
        "additional_weight_bytes": 0,
        "weight_source": "Complete existing physical BF16 cache only",
        "checkpoint_or_logical_batch_selection": False,
        "SM": 86,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                directory / "resident_small_decode.py",
                directory / "resident_small_decode_kernel.py",
            )
        },
    }


def warm(layer):
    """Compile both operator families before any graph captures the layer."""
    if not geometry(1, layer.shape):
        return
    import torch

    weight = layer.prefill_physical_weight
    if weight.device.type != "cuda" or tuple(
        torch.cuda.get_device_capability(weight.device)
    ) != (8, 6):
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Small resident cache must prepare before graph capture")
    from .resident_small_decode_kernel import cached_linear

    for dtype in (torch.float32, torch.bfloat16):
        for rows in (1, 4):
            source = torch.zeros(
                (rows, weight.shape[1]), device=weight.device, dtype=dtype
            )
            cached_linear(source, weight, False)
    layer.resident_small_decode_warmed = True
